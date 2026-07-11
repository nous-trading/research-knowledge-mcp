"""SQLite-backed chunk store.

Replaces the former ``chunks.json`` in-RAM dict (119K chunks ≈ 3.9 GB RSS on
the Discord corpus). A search only ever reads the rerank candidates (~150
rows) and the final top-k, so chunk bodies live on disk and are fetched
on demand. Row order (``pos``) preserves the faiss vector order.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ..models import Chunk

logger = logging.getLogger(__name__)


def create(db_path: Path) -> sqlite3.Connection:
    """Create (or reset) a chunk store at db_path."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # Remove on rebuild
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE chunks (
            pos INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            paper_id TEXT NOT NULL,
            section_path TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            raw_text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            contextualized_text TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_chunks_paper ON chunks(paper_id)")
    conn.commit()
    return conn


def open_store(db_path: Path) -> sqlite3.Connection:
    """Open an existing chunk store.

    Never creates the file: a bare ``sqlite3.connect`` on a missing path
    would leave a 0-byte db behind, which then defeats both the migrate-index
    completion check and the legacy-format guidance in ``HybridIndex.load``.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Chunk store not found: {db_path}. Run "
            f"`RESEARCH_KNOWLEDGE_DATA_DIR={db_path.parent.parent} "
            "python -m research_knowledge.cli migrate-index` once."
        )
    conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
    # Schema guarantee for stores created before the paper_id index existed.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id)")
    conn.commit()
    return conn


def add_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Bulk-insert chunks in list order (pos = faiss vector order)."""
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            (
                pos,
                c.chunk_id,
                c.paper_id,
                json.dumps(c.section_path, ensure_ascii=False),
                c.page_start,
                c.page_end,
                c.raw_text,
                c.token_count,
                c.contextualized_text,
            )
            for pos, c in enumerate(chunks)
        ),
    )
    conn.commit()


def _row_to_chunk(row: tuple) -> Chunk:
    return Chunk(
        chunk_id=row[0],
        paper_id=row[1],
        section_path=json.loads(row[2]),
        page_start=row[3],
        page_end=row[4],
        raw_text=row[5],
        token_count=row[6],
        contextualized_text=row[7],
    )


_CHUNK_COLS = (
    "chunk_id, paper_id, section_path, page_start, page_end, "
    "raw_text, token_count, contextualized_text"
)


def get_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, Chunk]:
    """Fetch full Chunk objects for the given ids."""
    if not chunk_ids:
        return {}
    ph = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT {_CHUNK_COLS} FROM chunks WHERE chunk_id IN ({ph})",
        chunk_ids,
    ).fetchall()
    return {row[0]: _row_to_chunk(row) for row in rows}


def texts_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, str]:
    """Fetch contextualized_text only (rerank input) for the given ids."""
    if not chunk_ids:
        return {}
    ph = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, contextualized_text FROM chunks WHERE chunk_id IN ({ph})",
        chunk_ids,
    ).fetchall()
    return dict(rows)


def paper_ids_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, str]:
    """Fetch chunk_id → paper_id for the given ids."""
    if not chunk_ids:
        return {}
    ph = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, paper_id FROM chunks WHERE chunk_id IN ({ph})",
        chunk_ids,
    ).fetchall()
    return dict(rows)


def transcript_by_paper(conn: sqlite3.Connection, paper_id: str) -> str:
    """Reconstruct a document body from its chunks, in order.

    Chunking uses token overlap, so adjacent chunks may repeat a few lines
    around each seam; seams are marked explicitly rather than hidden.
    """
    rows = conn.execute(
        "SELECT raw_text FROM chunks WHERE paper_id = ? ORDER BY pos",
        (paper_id,),
    ).fetchall()
    return "\n\n[--- chunk seam (may repeat a few overlapping lines) ---]\n\n".join(
        r[0] for r in rows
    )


def id_order(conn: sqlite3.Connection) -> list[str]:
    """All chunk_ids in faiss vector order."""
    return [r[0] for r in conn.execute("SELECT chunk_id FROM chunks ORDER BY pos")]


def count(conn: sqlite3.Connection) -> int:
    """Number of chunks in the store."""
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def migrate_from_json(meta_path: Path, db_path: Path) -> int:
    """One-off migration: legacy ``chunks.json`` → ``chunks.db``.

    Reads the legacy meta (chunk_id_order + chunks dict), writes the SQLite
    store in faiss order, and renames the JSON to ``chunks.json.bak`` so the
    migration is reversible until the operator deletes the backup.

    Returns the number of migrated chunks.
    """
    logger.info("Migrating %s → %s ...", meta_path.name, db_path.name)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    order: list[str] = meta.get("chunk_id_order", [])
    raw: dict[str, dict] = meta.get("chunks", {})

    chunks = [Chunk.model_validate(raw[cid]) for cid in order]
    # Build into a tmp file and rename atomically: chunks.db existing on disk
    # is the completion marker (a killed migration must not look migrated).
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    conn = create(tmp_path)
    add_chunks(conn, chunks)
    conn.close()
    tmp_path.replace(db_path)

    meta_path.rename(meta_path.with_suffix(".json.bak"))
    logger.info("Migrated %d chunks. Legacy meta kept as %s.bak", len(chunks), meta_path.name)
    return len(chunks)
