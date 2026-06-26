"""Ingest orchestration.

PDF → Docling parsing → chunking → Contextual Retrieval → JSONL storage → manifest update.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm.auth import get_async_anthropic_client
from ..models import Chunk, Manifest, PaperMeta
from ..paths import (
    CHUNKS_DIR,
    MANIFEST_PATH,
    PAPERS_MARKDOWN_DIR,
    PAPERS_PROCESSED_DIR,
    ensure_dirs,
)
from .chunking import chunk_document
from .contextualizer import contextualize_chunks, estimate_cost
from .metadata_extractor import extract_metadata_haiku, is_metadata_complete
from .parsing import (
    extract_authors_from_markdown,
    extract_title_from_markdown,
    parse_pdf,
)

logger = logging.getLogger(__name__)


def _compute_paper_id(pdf_path: Path) -> str:
    """Return the first 16 hex characters of the SHA-256 hash of the PDF bytes."""
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return h[:16]


def _extract_year(markdown: str) -> int | None:
    """Extract the publication year from markdown using heuristics."""
    # Find all 4-digit numbers in the 2000-2099 range
    years = re.findall(r"\b(20\d{2})\b", markdown[:5000])
    if years:
        # Return the most frequent year (likely near the title/abstract)
        return int(max(set(years), key=years.count))
    return None


async def _extract_metadata_with_fallback(markdown: str) -> dict:
    """Extract title, authors, and year — Haiku first, heuristic fallback.

    Haiku is preferred because the LLM can understand academic PDF edge cases
    such as endorsement pages, cover pages, and mixed metadata, which heuristics
    cannot handle reliably. Falls back to heuristics on API error or parse failure.
    """
    # First attempt: Haiku
    try:
        client = get_async_anthropic_client()
        haiku_meta = await extract_metadata_haiku(markdown, client)
    except Exception as e:
        logger.warning("Haiku metadata extraction failed, falling back to heuristics: %s", e)
        haiku_meta = None

    if is_metadata_complete(haiku_meta):
        # Supplement year with heuristics if Haiku returned None
        year = haiku_meta.get("year") or _extract_year(markdown)
        return {
            "title": haiku_meta["title"],
            "authors": haiku_meta.get("authors") or [],
            "year": year,
        }

    # Fallback: heuristics (parsing.py)
    logger.info("Haiku metadata incomplete — falling back to heuristics")
    return {
        "title": extract_title_from_markdown(markdown),
        "authors": extract_authors_from_markdown(markdown),
        "year": _extract_year(markdown),
    }


def _save_chunks_jsonl(paper_id: str, chunks: list[Chunk]) -> Path:
    """Serialize chunks to a JSONL file."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = CHUNKS_DIR / f"{paper_id}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    return jsonl_path


def load_all_chunks() -> list[Chunk]:
    """Load chunks from all JSONL files."""
    chunks: list[Chunk] = []
    if not CHUNKS_DIR.exists():
        return chunks
    for jsonl_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk.model_validate_json(line))
    return chunks


async def ingest_one_pdf(
    pdf_path: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest a single PDF.

    Args:
        pdf_path: Path to the PDF file.
        force: Re-process even if the paper has already been ingested.
        dry_run: If True, return only a cost estimate without making API calls.

    Returns:
        {paper_id, title, chunk_count, token_usage, cost_estimate, ...}
    """
    ensure_dirs()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    paper_id = _compute_paper_id(pdf_path)

    # Deduplication check
    manifest = Manifest.load(MANIFEST_PATH)
    if paper_id in manifest.papers and not force:
        logger.info("Already ingested: %s (%s)", pdf_path.name, paper_id)
        return {
            "paper_id": paper_id,
            "status": "already_ingested",
            "title": manifest.papers[paper_id].title,
        }

    # 1) Docling parsing
    logger.info("[1/5] Parsing PDF: %s", pdf_path.name)
    markdown, blocks = parse_pdf(pdf_path)

    # Extract title/authors/year/DOI — Haiku first, heuristic fallback
    meta = await _extract_metadata_with_fallback(markdown)
    title = meta["title"]
    authors = meta["authors"]
    year = meta["year"]

    # 2) Chunking
    logger.info("[2/5] Chunking...")
    raw_chunks = chunk_document(paper_id, blocks)
    logger.info("  → %d chunks created", len(raw_chunks))

    # 3) dry-run: cost estimate only
    if dry_run:
        cost = estimate_cost(markdown, raw_chunks)
        return {
            "paper_id": paper_id,
            "status": "dry_run",
            "title": title,
            "authors": authors,
            "year": year,
            "chunk_count": len(raw_chunks),
            "cost_estimate": cost,
        }

    # 4) Contextual Retrieval
    logger.info("[3/5] Contextual Retrieval (Haiku + prompt caching)...")
    contextualized, token_usage = await contextualize_chunks(
        paper_meta=PaperMeta(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            source_pdf_path="",
            markdown_path="",
            ingested_at=datetime.now(),
        ),
        full_markdown=markdown,
        raw_chunks=raw_chunks,
    )

    # 5) Persist
    logger.info("[4/5] Saving...")

    # Save markdown
    md_path = PAPERS_MARKDOWN_DIR / f"{paper_id}.md"
    md_path.write_text(markdown, encoding="utf-8")

    # Save JSONL (반환 경로는 미사용 — 저장 부작용만 필요)
    _save_chunks_jsonl(paper_id, contextualized)

    # Move PDF to processed directory
    processed_path = PAPERS_PROCESSED_DIR / pdf_path.name
    shutil.move(str(pdf_path), str(processed_path))

    # 6) Update manifest
    logger.info("[5/5] Updating manifest...")
    paper_meta = PaperMeta(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        source_pdf_path=str(processed_path.relative_to(processed_path.parents[2])),
        markdown_path=str(md_path.relative_to(md_path.parents[2])),
        ingested_at=datetime.now(),
        token_usage=token_usage,
    )
    manifest.papers[paper_id] = paper_meta
    chunks_count = 0
    for p in CHUNKS_DIR.glob("*.jsonl"):
        with open(p) as f:
            chunks_count += sum(1 for _ in f)
    manifest.chunks_count = chunks_count
    manifest.last_updated = datetime.now()
    manifest.save(MANIFEST_PATH)

    logger.info(
        "Ingest complete: %s (%s) — %d chunks, cost: input=%d, cache_read=%d",
        title,
        paper_id,
        len(contextualized),
        token_usage.get("input", 0),
        token_usage.get("cache_read", 0),
    )

    return {
        "paper_id": paper_id,
        "status": "ingested",
        "title": title,
        "authors": authors,
        "year": year,
        "chunk_count": len(contextualized),
        "token_usage": token_usage,
        "cost_estimate": estimate_cost(markdown, raw_chunks),
    }


async def ingest_directory(
    dir_path: Path,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Ingest all PDF files found in the given directory."""
    results: list[dict[str, Any]] = []
    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found: %s", dir_path)
        return results

    for pdf_path in pdf_files:
        try:
            result = await ingest_one_pdf(pdf_path, force=force, dry_run=dry_run)
            results.append(result)
        except Exception as e:
            logger.error("Ingest failed: %s — %s", pdf_path.name, e)
            results.append({
                "paper_id": None,
                "status": "error",
                "file": pdf_path.name,
                "error": str(e),
            })

    return results
