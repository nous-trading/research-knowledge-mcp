"""Filesystem layout for papers and the search index.

All locations derive from a single data root, resolved in this order:

1. ``RESEARCH_KNOWLEDGE_DATA_DIR`` environment variable, if set.
2. ``./research-knowledge-data`` under the current working directory.

Override the data root to point at an existing corpus, or to share one index
across processes. Layout under the data root::

    <data>/papers/{inbox,processed,markdown}   source PDFs + parsed markdown
    <data>/chunks/{paper_id}.jsonl             contextualized chunks
    <data>/manifest.json                       ingest catalog
    <data>/index/{bm25.db,dense.faiss,chunks.json}   search index
"""

from __future__ import annotations

import os
from pathlib import Path


def _data_root() -> Path:
    env = os.getenv("RESEARCH_KNOWLEDGE_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "research-knowledge-data").resolve()


DATA_ROOT = _data_root()

# Paper directories
PAPERS_INBOX_DIR = DATA_ROOT / "papers" / "inbox"
PAPERS_PROCESSED_DIR = DATA_ROOT / "papers" / "processed"
PAPERS_MARKDOWN_DIR = DATA_ROOT / "papers" / "markdown"

# Index / runtime artifacts
CHUNKS_DIR = DATA_ROOT / "chunks"
INDEX_DIR = DATA_ROOT / "index"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

# Optional concept definitions (markdown with frontmatter)
CONCEPTS_DIR = DATA_ROOT / "concepts"


def ensure_dirs() -> None:
    """Create the directories the pipeline writes to."""
    for d in [
        PAPERS_INBOX_DIR,
        PAPERS_PROCESSED_DIR,
        PAPERS_MARKDOWN_DIR,
        CHUNKS_DIR,
        INDEX_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
