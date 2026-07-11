"""research-knowledge CLI.

Usage:
    python -m research_knowledge.cli ingest <pdf|dir> [--force] [--dry-run]
    python -m research_knowledge.cli list
    python -m research_knowledge.cli status
    python -m research_knowledge.cli rebuild-index
    python -m research_knowledge.cli search <query> [--top-k 20] [--debug]
    python -m research_knowledge.cli summarize <paper_id> [--focus ...]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

# Avoid a faiss + torch OpenMP clash (macOS ARM64 SEGFAULT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import typer

app = typer.Typer(name="research-knowledge", help="Academic paper RAG system CLI")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to a PDF file or directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-process already ingested papers"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Estimate cost only (no API calls)"),
) -> None:
    """Ingest a PDF file or directory into the knowledge base."""
    from .ingest.pipeline import ingest_directory, ingest_one_pdf

    p = Path(path)
    if not p.exists():
        typer.echo(f"File or directory not found: {path}", err=True)
        raise typer.Exit(1)

    if p.is_dir():
        results = asyncio.run(ingest_directory(p, force=force, dry_run=dry_run))
    else:
        result = asyncio.run(ingest_one_pdf(p, force=force, dry_run=dry_run))
        results = [result]

    for r in results:
        typer.echo(json.dumps(r, indent=2, ensure_ascii=False, default=str))


@app.command("list")
def list_papers() -> None:
    """List all ingested papers."""
    from .models import Manifest
    from .paths import MANIFEST_PATH

    manifest = Manifest.load(MANIFEST_PATH)
    if not manifest.papers:
        typer.echo("No ingested papers found.")
        return

    for pid, meta in manifest.papers.items():
        typer.echo(
            f"[{pid}] {meta.title} ({meta.year or '?'}) "
            f"— authors: {', '.join(meta.authors) or 'unknown'}"
        )
    typer.echo(f"\nTotal: {len(manifest.papers)} papers, {manifest.chunks_count} chunks")


@app.command()
def status() -> None:
    """Show system status."""
    from .models import Manifest
    from .paths import CHUNKS_DIR, INDEX_DIR, MANIFEST_PATH, PAPERS_INBOX_DIR

    manifest = Manifest.load(MANIFEST_PATH)

    inbox_count = len(list(PAPERS_INBOX_DIR.glob("*.pdf"))) if PAPERS_INBOX_DIR.exists() else 0
    chunks_files = len(list(CHUNKS_DIR.glob("*.jsonl"))) if CHUNKS_DIR.exists() else 0
    index_exists = (INDEX_DIR / "dense.faiss").exists() and (INDEX_DIR / "bm25.db").exists()

    typer.echo(f"Papers: {len(manifest.papers)}")
    typer.echo(f"Chunks: {manifest.chunks_count}")
    typer.echo(f"JSONL files: {chunks_files}")
    typer.echo(f"Inbox pending: {inbox_count}")
    typer.echo(f"Index: {'ready' if index_exists else 'missing (run rebuild-index)'}")
    typer.echo(f"Embedding model: {manifest.embedding_model}")
    typer.echo(f"Reranker model: {manifest.reranker_model}")
    typer.echo(f"Last updated: {manifest.last_updated}")


@app.command("rebuild-index")
def rebuild_index_cmd() -> None:
    """Rebuild the full search index from all ingested chunks."""
    from .retrieval.index import rebuild_index

    typer.echo("Rebuilding index...")
    idx = rebuild_index()
    typer.echo(
        f"Index rebuild complete: {idx.chunks_count} chunks, "
        f"faiss={idx.faiss_index is not None}, bm25={idx.bm25_db is not None}"
    )


@app.command("migrate-index")
def migrate_index_cmd() -> None:
    """One-off migration: legacy index/chunks.json → index/chunks.db.

    Converts the in-RAM chunk metadata format to the SQLite chunk store.
    The legacy JSON is kept as chunks.json.bak until you delete it.
    """
    from .paths import INDEX_DIR
    from .retrieval import chunk_store

    meta_path = INDEX_DIR / "chunks.json"
    db_path = INDEX_DIR / "chunks.db"
    if db_path.exists():
        typer.echo(f"Already migrated: {db_path} exists.")
        raise typer.Exit(0)
    if not meta_path.exists():
        typer.echo(f"Nothing to migrate: {meta_path} not found.", err=True)
        raise typer.Exit(1)

    n = chunk_store.migrate_from_json(meta_path, db_path)
    typer.echo(f"Migrated {n} chunks → {db_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Number of results to return"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable per-stage debug logging"),
) -> None:
    """Run a hybrid search query against ingested papers."""
    from .retrieval.embedding import DenseEmbedder
    from .retrieval.index import HybridIndex
    from .retrieval.reranker import Reranker
    from .retrieval.search import HybridSearcher

    if debug:
        logging.getLogger("research_knowledge").setLevel(logging.DEBUG)

    idx = HybridIndex.load()
    if not idx.is_ready:
        typer.echo("Index not found. Run rebuild-index first.", err=True)
        raise typer.Exit(1)

    embedder = DenseEmbedder()
    reranker = Reranker()
    searcher = HybridSearcher(idx, embedder, reranker)

    results = searcher.search(query, top_k=top_k, debug=debug)

    if not results:
        typer.echo("No results found.")
        return

    for i, sc in enumerate(results, 1):
        chunk = sc.chunk
        snippet = chunk.raw_text[:200].replace("\n", " ")
        typer.echo(
            f"\n[{i}] score={sc.score:.4f} | paper={chunk.paper_id} "
            f"| section={'/'.join(chunk.section_path) or 'root'}"
        )
        typer.echo(f"    {snippet}...")


@app.command()
def summarize(
    paper_id: str = typer.Argument(..., help="Paper ID"),
    focus: str = typer.Option(None, "--focus", "-f", help="Summary focus topic"),
) -> None:
    """Summarize a paper using Claude Haiku."""
    from .llm.summarizer import summarize_paper

    result = asyncio.run(summarize_paper(paper_id, focus=focus))
    typer.echo(result)


if __name__ == "__main__":
    app()
