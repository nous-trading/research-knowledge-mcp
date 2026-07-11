"""
Research Knowledge MCP Server — academic paper RAG search.

Contextual Retrieval + Hybrid Search (BM25 + Dense + Cross-Encoder Reranking)
over an ingested corpus of academic papers, exposed as an MCP server.

Tools:
    - search_papers: hybrid search (BM25 + dense + rerank)
    - get_paper: full-text markdown or BibTeX
    - list_papers: list ingested papers
    - cite: APA / BibTeX citation
    - summarize_paper: Haiku-based paper summary
    - get_concept: concept definition lookup

Usage:
    python -m research_knowledge.server   # or: research-knowledge-mcp

Port: 8014 (override with MCP_PORT env var)
"""

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

# Avoid a faiss + torch OpenMP clash (macOS ARM64 SEGFAULT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Singletons (lazy-initialized in lifespan)
_index = None
_embedder = None
_reranker = None
_searcher = None


@asynccontextmanager
async def _lifespan(app):
    """Server start/stop lifecycle. Models are lazy-loaded on first search."""
    logger.info("Research Knowledge MCP server starting")
    yield
    logger.info("Research Knowledge MCP server stopped")


app = FastMCP("research-knowledge", lifespan=_lifespan)


# Serializes model use in a worker thread; searches run via asyncio.to_thread
# so the shared HTTP daemon's event loop stays responsive during a cold model
# load (~20 s) or rerank. The TTL watchdog takes the same lock non-blockingly,
# so it can never unload models mid-search.
_search_lock = threading.Lock()


def _ensure_searcher():
    """Initialize the searcher singleton (lazy).

    An index that failed to load (missing/empty) is NOT pinned: the warm
    daemon retries the load on the next search, so deploying data after the
    first search does not require a service restart.
    """
    global _index, _embedder, _reranker, _searcher
    if _searcher is not None and _index.is_ready:
        return

    from research_knowledge.retrieval.embedding import DenseEmbedder
    from research_knowledge.retrieval.index import HybridIndex
    from research_knowledge.retrieval.reranker import Reranker
    from research_knowledge.retrieval.search import HybridSearcher

    _index = HybridIndex.load()
    if _embedder is None:
        _embedder = DenseEmbedder()
        _reranker = Reranker()
    _searcher = HybridSearcher(_index, _embedder, _reranker)


def _run_search(query: str, top_k: int, paper_ids: list[str] | None):
    """Blocking search body — runs in a worker thread under _search_lock.

    Returns None when the index is not ready (caller renders the error).
    """
    with _search_lock:
        _ensure_searcher()
        if not _index.is_ready:
            return None
        return _searcher.search(query, top_k=top_k, paper_ids=paper_ids)


# =============================================================================
# Model idle TTL — searches load the two bge models (~2.5 GB accelerator
# memory); a watchdog unloads them after RESEARCH_KNOWLEDGE_MODEL_TTL_SECONDS
# of no searches (default 600, 0 disables) so an idle server holds ~0.
# =============================================================================
_MODEL_TTL_SECONDS = int(os.environ.get("RESEARCH_KNOWLEDGE_MODEL_TTL_SECONDS", "600"))
_last_search_ts: float = 0.0
_ttl_watchdog: asyncio.Task | None = None


def _touch_models() -> None:
    """Record model use and arm the idle watchdog (call from async context)."""
    global _last_search_ts, _ttl_watchdog
    _last_search_ts = time.monotonic()
    if _MODEL_TTL_SECONDS <= 0:
        return
    if _ttl_watchdog is None or _ttl_watchdog.done():
        _ttl_watchdog = asyncio.get_running_loop().create_task(_ttl_loop())


async def _ttl_loop() -> None:
    from research_knowledge.retrieval.embedding import DenseEmbedder
    from research_knowledge.retrieval.reranker import Reranker

    while True:
        await asyncio.sleep(min(_MODEL_TTL_SECONDS, 60))
        if time.monotonic() - _last_search_ts >= _MODEL_TTL_SECONDS:
            # Non-blocking: a search in flight holds the lock — skip and retry.
            if not _search_lock.acquire(blocking=False):
                continue
            try:
                DenseEmbedder.unload()
                Reranker.unload()
            finally:
                _search_lock.release()
            return  # re-armed by the next search


# =============================================================================
# Manifest brief cache — search results only need title/citation fields, and
# Manifest.load() parses the full manifest (45 MB on the Discord corpus) on
# every call. Cache a slim projection, invalidated by file mtime.
# =============================================================================
_briefs: dict | None = None
_briefs_mtime: float | None = None


def _manifest_briefs() -> dict:
    """paper_id → {title, citation_key} projection of the manifest."""
    global _briefs, _briefs_mtime
    from research_knowledge.models import Manifest
    from research_knowledge.paths import MANIFEST_PATH

    mtime = MANIFEST_PATH.stat().st_mtime if MANIFEST_PATH.exists() else None
    if _briefs is not None and mtime == _briefs_mtime:
        return _briefs

    manifest = Manifest.load(MANIFEST_PATH)
    _briefs = {
        pid: {
            "title": p.title,
            "citation_key": (
                f"{p.authors[0].split()[-1] if p.authors else 'unknown'}{p.year or ''}"
            ),
        }
        for pid, p in manifest.papers.items()
    }
    _briefs_mtime = mtime
    return _briefs


# =============================================================================
# Tool 1: search_papers
# =============================================================================
@app.tool(annotations={"readOnlyHint": True})
async def search_papers(
    query: str,
    top_k: int = 20,
    paper_ids: list[str] | None = None,
) -> str:
    """Perform hybrid search over ingested academic papers.

    BM25 + Dense retrieval → RRF fusion → Cross-Encoder reranking.

    Args:
        query: Search query (e.g. "market microstructure", "order flow toxicity")
        top_k: Maximum number of results to return (default 20)
        paper_ids: Restrict search to specific paper IDs (None = search all)
    """
    try:
        _touch_models()
        results = await asyncio.to_thread(_run_search, query, top_k, paper_ids)
        if results is None:
            return json.dumps(
                {"error": "Index not found. Run rebuild-index first."},
                ensure_ascii=False,
            )

        briefs = _manifest_briefs()

        output = []
        for sc in results:
            chunk = sc.chunk
            brief = briefs.get(chunk.paper_id)
            output.append(
                {
                    "paper_id": chunk.paper_id,
                    "title": brief["title"] if brief else "unknown",
                    "section": "/".join(chunk.section_path) or "root",
                    "page_range": f"{chunk.page_start}-{chunk.page_end}",
                    "snippet": chunk.raw_text[:500],
                    "score": round(sc.score, 4),
                    "citation_key": brief["citation_key"] if brief else chunk.paper_id,
                }
            )

        return json.dumps(
            {"query": query, "results": output, "count": len(output)},
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("search_papers failed: %s", e, exc_info=True)
        return json.dumps({"error": str(e), "tool": "search_papers"}, indent=2)


# =============================================================================
# Tool 2: get_paper
# =============================================================================
@app.tool(annotations={"readOnlyHint": True})
async def get_paper(
    paper_id: str,
    format: str = "markdown",
) -> str:
    """Return the full text of a paper.

    Args:
        paper_id: Paper ID
        format: Return format — "markdown" (full text) or "bibtex" (BibTeX entry)
    """
    try:
        from research_knowledge.models import Manifest
        from research_knowledge.paths import MANIFEST_PATH, PAPERS_MARKDOWN_DIR

        manifest = Manifest.load(MANIFEST_PATH)
        if paper_id not in manifest.papers:
            return json.dumps({"error": f"Paper not found: {paper_id}"}, ensure_ascii=False)

        paper = manifest.papers[paper_id]

        if format == "bibtex":
            first_author = paper.authors[0].split()[-1] if paper.authors else "unknown"
            bibtex = (
                f"@article{{{first_author}{paper.year or ''},\n"
                f"  title = {{{paper.title}}},\n"
                f"  author = {{{' and '.join(paper.authors)}}},\n"
                f"  year = {{{paper.year or '?'}}},\n"
                f"}}"
            )
            return bibtex

        # markdown
        md_path = PAPERS_MARKDOWN_DIR / f"{paper_id}.md"
        if not md_path.exists():
            return json.dumps({"error": f"Markdown file not found: {paper_id}"}, ensure_ascii=False)

        return md_path.read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "get_paper"}, indent=2)


# =============================================================================
# Tool 3: list_papers
# =============================================================================
@app.tool(annotations={"readOnlyHint": True})
async def list_papers(
    year_min: int | None = None,
) -> str:
    """Return a list of ingested papers.

    Args:
        year_min: Minimum publication year filter (None = all years)
    """
    try:
        from research_knowledge.models import Manifest
        from research_knowledge.paths import MANIFEST_PATH

        manifest = Manifest.load(MANIFEST_PATH)
        papers = []
        for pid, meta in manifest.papers.items():
            if year_min and meta.year and meta.year < year_min:
                continue
            papers.append(
                {
                    "paper_id": pid,
                    "title": meta.title,
                    "authors": meta.authors,
                    "year": meta.year,
                    "ingested_at": meta.ingested_at.isoformat(),
                }
            )

        return json.dumps(
            {"papers": papers, "count": len(papers), "total": len(manifest.papers)},
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "list_papers"}, indent=2)


# =============================================================================
# Tool 4: cite
# =============================================================================
@app.tool(annotations={"readOnlyHint": True})
async def cite(
    paper_id: str,
    style: str = "apa",
) -> str:
    """Generate a citation string for a paper.

    Args:
        paper_id: Paper ID
        style: Citation style — "apa" or "bibtex"
    """
    try:
        from research_knowledge.models import Manifest
        from research_knowledge.paths import MANIFEST_PATH

        manifest = Manifest.load(MANIFEST_PATH)
        if paper_id not in manifest.papers:
            return json.dumps({"error": f"Paper not found: {paper_id}"}, ensure_ascii=False)

        paper = manifest.papers[paper_id]
        authors_str = ", ".join(paper.authors) if paper.authors else "Unknown"
        year = paper.year or "n.d."

        if style == "bibtex":
            first_author = paper.authors[0].split()[-1] if paper.authors else "unknown"
            return (
                f"@article{{{first_author}{paper.year or ''},\n"
                f"  title = {{{paper.title}}},\n"
                f"  author = {{{' and '.join(paper.authors)}}},\n"
                f"  year = {{{year}}},\n"
                f"}}"
            )

        # APA
        return f"{authors_str} ({year}). {paper.title}."
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "cite"}, indent=2)


# =============================================================================
# Tool 5: summarize_paper
# =============================================================================
@app.tool(name="summarize_paper", annotations={"readOnlyHint": True})
async def summarize_paper_tool(
    paper_id: str,
    focus: str | None = None,
) -> str:
    """Summarize a paper (Haiku-based, with caching).

    Args:
        paper_id: Paper ID
        focus: Summary focus topic (None defaults to 'algorithmic trading')
    """
    try:
        from research_knowledge.llm.summarizer import summarize_paper

        return await summarize_paper(paper_id, focus=focus)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "summarize_paper"}, indent=2)


# =============================================================================
# Tool 6: get_concept
# =============================================================================
@app.tool(name="get_concept", annotations={"readOnlyHint": True})
async def get_concept_tool(
    concept_name: str,
) -> str:
    """Look up a concept definition (docs/research/concepts/).

    Args:
        concept_name: Concept name (e.g. "vpin", "kyle-lambda", "adverse-selection")
    """
    try:
        from research_knowledge.llm.concept_resolver import get_concept

        result = get_concept(concept_name)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "get_concept"}, indent=2)


# =============================================================================
# Health Check
# =============================================================================
@app.tool()
async def health_check() -> str:
    """Check server health status."""
    return json.dumps(
        {
            "status": "healthy",
            "server": "research-knowledge",
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )


def main() -> None:
    """Console entry point (``research-knowledge-mcp``).

    Transport is selected via ``MCP_TRANSPORT`` (``stdio`` default, or
    ``streamable-http`` on ``MCP_PORT``, default 8014).
    """
    # journalctl 관측성: 모델 로드/언로드·인덱스 로드 INFO 가 systemd 저널에 남도록
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        app.run(transport="stdio", show_banner=False)
    else:
        app.run(
            transport="streamable-http",
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", 8014)),
            show_banner=False,
            stateless_http=True,
        )


if __name__ == "__main__":
    main()
