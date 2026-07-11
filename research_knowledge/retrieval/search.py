"""HybridSearcher — BM25 + Dense + RRF Fusion + Cross-Encoder Reranking.

Search pipeline:
1. BM25 top-150
2. Dense top-150
3. RRF fusion (k=60) → rerank candidates (RESEARCH_KNOWLEDGE_RERANK_CANDIDATES)
4. bge-reranker-v2-m3 cross-encoder → top-N

The cross-encoder is linear in candidate count and dominates latency
(~270 ms/doc on Apple-Silicon CPU, ~27 ms/doc on RTX 2070), so the
candidate cap is the main speed knob.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

from ..models import ScoredChunk
from . import bm25_index
from .embedding import DenseEmbedder
from .index import HybridIndex
from .reranker import Reranker

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Hybrid searcher combining BM25, dense retrieval, RRF fusion, and reranking."""

    def __init__(
        self,
        index: HybridIndex,
        embedder: DenseEmbedder,
        reranker: Reranker,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 20,
        paper_ids: list[str] | None = None,
        debug: bool = False,
        rerank_candidates: int | None = None,
    ) -> list[ScoredChunk]:
        """Run a hybrid search.

        Args:
            query: Search query string
            top_k: Number of final results to return
            paper_ids: Restrict results to these paper IDs (None means all)
            debug: When True, log intermediate results at each stage
            rerank_candidates: RRF-fused candidates passed to the
                cross-encoder (None → RESEARCH_KNOWLEDGE_RERANK_CANDIDATES
                env, default 150)

        Returns:
            List of ScoredChunk sorted by score descending
        """
        if rerank_candidates is None:
            rerank_candidates = int(
                os.environ.get("RESEARCH_KNOWLEDGE_RERANK_CANDIDATES", "150")
            )
        if rerank_candidates <= 0:
            raise ValueError(
                f"rerank_candidates must be positive, got {rerank_candidates} "
                "(check RESEARCH_KNOWLEDGE_RERANK_CANDIDATES)"
            )
        if not self.index.is_ready:
            logger.warning("Index is not ready. Run rebuild-index first.")
            return []

        # 1) BM25 top-150
        bm25_results = bm25_index.search(
            self.index.bm25_db, query, top_k=150, paper_ids=paper_ids
        )
        if debug:
            logger.info("BM25 results: %d", len(bm25_results))
            for cid, score in bm25_results[:5]:
                logger.info("  BM25: %s → %.4f", cid, score)

        # 2) Dense top-150
        q_emb = self.embedder.encode([query])  # (1, 1024)
        search_k = min(150, self.index.faiss_index.ntotal)
        if search_k == 0:
            dense_results: list[tuple[str, float]] = []
        else:
            distances, indices = self.index.faiss_index.search(q_emb, search_k)
            hits = [
                (self.index.chunk_id_order[i], float(d))
                for d, i in zip(distances[0], indices[0], strict=True)
                if 0 <= i < len(self.index.chunk_id_order)
            ]
            if paper_ids:
                hit_paper_ids = self.index.get_paper_ids([cid for cid, _ in hits])
                hits = [
                    (cid, d) for cid, d in hits
                    if hit_paper_ids.get(cid) in paper_ids
                ]
            dense_results = hits

        if debug:
            logger.info("Dense results: %d", len(dense_results))
            for cid, score in dense_results[:5]:
                logger.info("  Dense: %s → %.4f", cid, score)

        # 3) RRF fusion (k=60)
        rrf_scores: dict[str, float] = defaultdict(float)
        for rank, (cid, _) in enumerate(bm25_results):
            rrf_scores[cid] += 1.0 / (60 + rank)
        for rank, (cid, _) in enumerate(dense_results):
            rrf_scores[cid] += 1.0 / (60 + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :rerank_candidates
        ]

        if debug:
            logger.info("RRF fusion results: %d", len(fused))
            for cid, score in fused[:5]:
                logger.info("  RRF: %s → %.4f", cid, score)

        if not fused:
            return []

        # 4) Rerank → top-N (candidate texts fetched on demand from SQLite).
        # Direct indexing: a fused cid missing from chunks.db means the BM25/
        # faiss indexes and the chunk store desynced — KeyError is honest.
        fused_texts = self.index.get_texts([cid for cid, _ in fused])
        docs = [fused_texts[cid] for cid, _ in fused]

        reranked = self.reranker.rerank(query, docs, top_n=top_k)

        if debug:
            logger.info("Rerank results: %d", len(reranked))
            for orig_idx, score in reranked[:5]:
                logger.info("  Rerank: %s → %.4f", fused[orig_idx][0], score)

        top_ids = [fused[orig_idx][0] for orig_idx, _ in reranked]
        top_chunks = self.index.get_chunks(top_ids)
        return [
            ScoredChunk(chunk=top_chunks[cid], score=score)
            for (_, score), cid in zip(reranked, top_ids, strict=True)
        ]
