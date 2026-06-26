"""HybridIndex — combined faiss IndexFlatIP + SQLite FTS5 index.

faiss supports CPU only (MPS not supported). Embedding runs on MPS while the
index runs on CPU, keeping the two devices separate.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import faiss

from ..ingest.pipeline import load_all_chunks
from ..models import Chunk
from ..paths import INDEX_DIR
from . import bm25_index
from .embedding import DenseEmbedder

logger = logging.getLogger(__name__)

_BM25_DB_NAME = "bm25.db"
_FAISS_INDEX_NAME = "dense.faiss"
_CHUNKS_META_NAME = "chunks.json"


class HybridIndex:
    """Hybrid index combining faiss dense search and SQLite FTS5 BM25."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.bm25_db: sqlite3.Connection | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.chunk_id_order: list[str] = []
        self.chunks: dict[str, Chunk] = {}

    @classmethod
    def build_from_chunks(
        cls,
        chunks: list[Chunk],
        index_dir: Path,
        embedder: DenseEmbedder,
    ) -> HybridIndex:
        """Build an index from a list of chunks.

        Args:
            chunks: Full list of chunks to index
            index_dir: Directory to store the index
            embedder: Dense embedder

        Returns:
            The built HybridIndex
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        instance = cls(index_dir)

        if not chunks:
            logger.warning("Empty chunk list — creating empty index")
            instance._save_empty()
            return instance

        # 1) BM25 index
        logger.info("[1/3] Building BM25 index (%d chunks)...", len(chunks))
        bm25_path = index_dir / _BM25_DB_NAME
        if bm25_path.exists():
            bm25_path.unlink()  # Remove on rebuild
        instance.bm25_db = bm25_index.create_index(bm25_path)
        bm25_index.add_chunks(instance.bm25_db, chunks)

        # 2) Dense embedding + faiss
        logger.info("[2/3] Generating dense embeddings...")
        texts = [c.contextualized_text for c in chunks]
        embeddings = embedder.encode(texts)  # (N, 1024)

        logger.info("[3/3] Building faiss IndexFlatIP...")
        dim = embeddings.shape[1]
        instance.faiss_index = faiss.IndexFlatIP(dim)
        instance.faiss_index.add(embeddings)

        # Save metadata
        instance.chunk_id_order = [c.chunk_id for c in chunks]
        instance.chunks = {c.chunk_id: c for c in chunks}
        instance._save_meta()

        logger.info(
            "Index build complete: %d chunks, %dd vectors, BM25 + faiss",
            len(chunks),
            dim,
        )
        return instance

    def _save_meta(self) -> None:
        """Persist the faiss index and chunk metadata to disk."""
        # Save faiss index
        if self.faiss_index is not None:
            faiss.write_index(
                self.faiss_index, str(self.index_dir / _FAISS_INDEX_NAME)
            )

        # Save chunk metadata
        meta = {
            "chunk_id_order": self.chunk_id_order,
            "chunks": {
                cid: chunk.model_dump() for cid, chunk in self.chunks.items()
            },
        }
        meta_path = self.index_dir / _CHUNKS_META_NAME
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def _save_empty(self) -> None:
        """Persist an empty index to disk."""
        self.chunk_id_order = []
        self.chunks = {}
        self.faiss_index = faiss.IndexFlatIP(1024)
        self.bm25_db = bm25_index.create_index(self.index_dir / _BM25_DB_NAME)
        self._save_meta()

    @classmethod
    def load(cls, index_dir: Path | None = None) -> HybridIndex:
        """Load an index from disk.

        Args:
            index_dir: Index directory (uses the default path when None)
        """
        if index_dir is None:
            index_dir = INDEX_DIR

        instance = cls(index_dir)
        meta_path = index_dir / _CHUNKS_META_NAME

        if not meta_path.exists():
            logger.warning("Index metadata not found: %s — returning empty index", meta_path)
            return instance

        # Load metadata
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instance.chunk_id_order = meta.get("chunk_id_order", [])
        instance.chunks = {
            cid: Chunk.model_validate(data)
            for cid, data in meta.get("chunks", {}).items()
        }

        # Load faiss index
        faiss_path = index_dir / _FAISS_INDEX_NAME
        if faiss_path.exists():
            instance.faiss_index = faiss.read_index(str(faiss_path))

        # Load BM25 index
        bm25_path = index_dir / _BM25_DB_NAME
        if bm25_path.exists():
            instance.bm25_db = sqlite3.connect(str(bm25_path))

        logger.info(
            "Index loaded: %d chunks, faiss=%s, bm25=%s",
            len(instance.chunks),
            instance.faiss_index is not None,
            instance.bm25_db is not None,
        )
        return instance

    @property
    def is_ready(self) -> bool:
        """Whether the index is ready for search."""
        return (
            self.faiss_index is not None
            and self.bm25_db is not None
            and len(self.chunks) > 0
        )


def rebuild_index(embedder: DenseEmbedder | None = None) -> HybridIndex:
    """Rebuild the index from all available chunks.

    Args:
        embedder: Dense embedder (creates a new one when None)

    Returns:
        The built HybridIndex
    """
    if embedder is None:
        embedder = DenseEmbedder()

    chunks = load_all_chunks()
    logger.info("Loaded %d chunks in total", len(chunks))

    return HybridIndex.build_from_chunks(chunks, INDEX_DIR, embedder)
