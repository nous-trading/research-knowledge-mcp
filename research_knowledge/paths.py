"""Filesystem layout for papers and the search index.

All locations derive from a single data root, resolved in this order:

1. ``RESEARCH_KNOWLEDGE_DATA_DIR`` environment variable, if set.
2. ``./research-knowledge-data`` under the current working directory.

Override the data root to point at an existing corpus, or to share one index
across processes. Papers live under ``<data>/papers/{inbox,processed,markdown}``
and the index under ``<data>/index``.
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
RESEARCH_INDEX_DIR = DATA_ROOT / "index"
CHUNKS_DIR = RESEARCH_INDEX_DIR / "chunks"
INDEX_DIR = RESEARCH_INDEX_DIR / "store"
MANIFEST_PATH = RESEARCH_INDEX_DIR / "manifest.json"

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
