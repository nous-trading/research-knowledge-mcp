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

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

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


def _ensure_searcher():
    """Initialize the searcher singleton (lazy)."""
    global _index, _embedder, _reranker, _searcher
    if _searcher is not None:
        return

    from research_knowledge.embedding import DenseEmbedder
    from research_knowledge.index import HybridIndex
    from research_knowledge.reranker import Reranker
    from research_knowledge.search import HybridSearcher

    _index = HybridIndex.load()
    _embedder = DenseEmbedder()
    _reranker = Reranker()
    _searcher = HybridSearcher(_index, _embedder, _reranker)


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
        _ensure_searcher()
        if not _index.is_ready:
            return json.dumps(
                {"error": "Index not found. Run rebuild-index first."},
                ensure_ascii=False,
            )

        results = _searcher.search(query, top_k=top_k, paper_ids=paper_ids)

        from research_knowledge.models import Manifest
        from research_knowledge.paths import MANIFEST_PATH

        manifest = Manifest.load(MANIFEST_PATH)

        output = []
        for sc in results:
            chunk = sc.chunk
            paper = manifest.papers.get(chunk.paper_id)
            output.append(
                {
                    "paper_id": chunk.paper_id,
                    "title": paper.title if paper else "unknown",
                    "section": "/".join(chunk.section_path) or "root",
                    "page_range": f"{chunk.page_start}-{chunk.page_end}",
                    "snippet": chunk.raw_text[:500],
                    "score": round(sc.score, 4),
                    "citation_key": f"{(paper.authors[0].split()[-1] if paper and paper.authors else 'unknown')}{paper.year or ''}" if paper else chunk.paper_id,
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
        from research_knowledge.summarizer import summarize_paper

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
        from research_knowledge.concept_resolver import get_concept

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def main() -> None:
    """Console entry point (``research-knowledge-mcp``).

    Transport is selected via ``MCP_TRANSPORT`` (``stdio`` default, or
    ``streamable-http`` on ``MCP_PORT``, default 8014).
    """
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        app.run(transport="stdio", show_banner=False)
    else:
        app.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=int(os.environ.get("MCP_PORT", 8014)),
            show_banner=False,
            stateless_http=True,
        )


if __name__ == "__main__":
    main()
