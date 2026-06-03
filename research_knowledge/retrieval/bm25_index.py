"""SQLite FTS5-based BM25 index.

Uses the porter + unicode61 tokenizer, optimized for English academic text.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from ..models import Chunk

logger = logging.getLogger(__name__)


def create_index(db_path: Path) -> sqlite3.Connection:
    """Create (or open) a SQLite connection with an FTS5 index."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            paper_id UNINDEXED,
            raw_text,
            contextualized_text,
            tokenize='porter unicode61'
        )
        """
    )
    conn.commit()
    return conn


def clear_index(conn: sqlite3.Connection) -> None:
    """Delete all rows from the index (call before a rebuild)."""
    conn.execute("DELETE FROM chunks_fts")
    conn.commit()


def add_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Insert a list of chunks into the FTS5 index."""
    conn.executemany(
        "INSERT INTO chunks_fts(chunk_id, paper_id, raw_text, contextualized_text) "
        "VALUES (?, ?, ?, ?)",
        [
            (c.chunk_id, c.paper_id, c.raw_text, c.contextualized_text)
            for c in chunks
        ],
    )
    conn.commit()
    logger.info("Added %d chunks to BM25 index", len(chunks))


def _sanitize_query(query: str) -> str:
    """Escape special characters in an FTS5 MATCH query.

    FTS5 treats AND, OR, NOT, NEAR, double-quotes, and parentheses as
    operators. This function strips them so raw user input is handled safely.
    """
    # Remove special characters (keep alphanumerics, whitespace, and hyphens)
    sanitized = re.sub(r'[^\w\s\-]', ' ', query)
    # Collapse consecutive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # An empty result (or all-special-char input) is returned as-is;
    # the caller is responsible for handling an empty return value.
    return sanitized


def search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 150,
    paper_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """BM25 search. Returns a list of (chunk_id, score) tuples.

    Args:
        conn: SQLite connection
        query: Search query string
        top_k: Maximum number of results to return
        paper_ids: Restrict results to these paper IDs (None means all)

    Returns:
        [(chunk_id, bm25_score), ...] — scores are positive (higher = more relevant)
    """
    sanitized = _sanitize_query(query)
    if not sanitized:
        return []

    try:
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            cur = conn.execute(
                f"""
                SELECT chunk_id, bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND paper_id IN ({placeholders})
                ORDER BY score
                LIMIT ?
                """,
                [sanitized] + paper_ids + [top_k],
            )
        else:
            cur = conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) as score
                FROM chunks_fts WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (sanitized, top_k),
            )
        # bm25() returns negative values (lower = more relevant); negate to get positive scores
        return [(row[0], -row[1]) for row in cur]
    except Exception as e:
        logger.warning("BM25 search failed (query=%r): %s", query, e)
        return []
