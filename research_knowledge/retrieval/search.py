"""HybridSearcher — BM25 + Dense + RRF Fusion + Cross-Encoder Reranking.

Search pipeline:
1. BM25 top-150
2. Dense top-150
3. RRF fusion (k=60) → top-150
4. bge-reranker-v2-m3 cross-encoder → top-N
"""

from __future__ import annotations

import logging
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
    ) -> list[ScoredChunk]:
        """Run a hybrid search.

        Args:
            query: Search query string
            top_k: Number of final results to return
            paper_ids: Restrict results to these paper IDs (None means all)
            debug: When True, log intermediate results at each stage

        Returns:
            List of ScoredChunk sorted by score descending
        """
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
            dense_results = []
            for d, i in zip(distances[0], indices[0], strict=True):
                if i >= 0 and i < len(self.index.chunk_id_order):
                    cid = self.index.chunk_id_order[i]
                    # paper_ids filter
                    if paper_ids and self.index.chunks[cid].paper_id not in paper_ids:
                        continue
                    dense_results.append((cid, float(d)))

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

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:150]

        if debug:
            logger.info("RRF fusion results: %d", len(fused))
            for cid, score in fused[:5]:
                logger.info("  RRF: %s → %.4f", cid, score)

        if not fused:
            return []

        # 4) Rerank → top-N
        docs = []
        valid_fused: list[tuple[str, float]] = []
        for cid, score in fused:
            if cid in self.index.chunks:
                docs.append(self.index.chunks[cid].contextualized_text)
                valid_fused.append((cid, score))

        if not docs:
            return []

        reranked = self.reranker.rerank(query, docs, top_n=top_k)

        if debug:
            logger.info("Rerank results: %d", len(reranked))
            for orig_idx, score in reranked[:5]:
                cid = valid_fused[orig_idx][0]
                logger.info("  Rerank: %s → %.4f", cid, score)

        results: list[ScoredChunk] = []
        for orig_idx, score in reranked:
            cid = valid_fused[orig_idx][0]
            if cid in self.index.chunks:
                results.append(
                    ScoredChunk(
                        chunk=self.index.chunks[cid],
                        score=score,
                    )
                )

        return results
