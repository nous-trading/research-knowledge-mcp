"""bge-reranker-v2-m3 Cross-Encoder Reranker.

Uses the sentence-transformers CrossEncoder to re-score query-document pairs.
FlagReranker has a compatibility issue with transformers 5.x
(prepare_for_model AttributeError), so the stable CrossEncoder API from
sentence-transformers is used instead.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Reranker:
    """bge-reranker-v2-m3-based cross-encoder. Uses lazy loading."""

    _model = None

    def _load(self) -> None:
        if Reranker._model is not None:
            return
        logger.info("Loading bge-reranker-v2-m3 model (CrossEncoder)...")
        from sentence_transformers import CrossEncoder

        Reranker._model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        logger.info("bge-reranker-v2-m3 model loaded.")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 20,
    ) -> list[tuple[int, float]]:
        """Re-score query-document pairs for relevance.

        Args:
            query: Search query string
            documents: List of document texts to re-score
            top_n: Number of top results to return

        Returns:
            [(original_index, score), ...] -- sorted by score descending
        """
        if not documents:
            return []

        self._load()
        pairs = [[query, doc] for doc in documents]
        scores = Reranker._model.predict(pairs)
        # scores is a numpy array
        scores_list = scores.tolist()
        if not isinstance(scores_list, list):
            scores_list = [scores_list]

        ranked = sorted(enumerate(scores_list), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
