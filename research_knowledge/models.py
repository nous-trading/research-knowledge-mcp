"""Pydantic models — Paper, Chunk, and Manifest."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PaperMeta(BaseModel):
    """Metadata for an ingested paper."""

    paper_id: str  # sha256 of pdf bytes (16 char prefix)
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    source_pdf_path: str  # papers/processed/...
    markdown_path: str  # papers/markdown/...
    ingested_at: datetime
    token_usage: dict[str, int] = Field(default_factory=dict)
    summary_cache: str | None = None
    summary_focus: str | None = None


class RawChunk(BaseModel):
    """Raw chunk immediately after parsing."""

    chunk_id: str  # f"{paper_id}_{idx:04d}"
    paper_id: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    raw_text: str
    token_count: int


class Chunk(RawChunk):
    """Chunk with injected context."""

    contextualized_text: str  # context prepend + raw_text


class ScoredChunk(BaseModel):
    """Search result chunk with relevance score."""

    chunk: Chunk
    score: float


class Manifest(BaseModel):
    """Ingestion state manifest."""

    papers: dict[str, PaperMeta] = Field(default_factory=dict)
    chunks_count: int = 0
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    last_updated: datetime = Field(default_factory=lambda: datetime.now())

    def save(self, path: Any) -> None:
        """Serialize the manifest to JSON and write it to disk."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> Manifest:
        """Load the manifest from JSON. Returns an empty instance if the file does not exist."""
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return cls()
        return cls.model_validate_json(p.read_text(encoding="utf-8"))
