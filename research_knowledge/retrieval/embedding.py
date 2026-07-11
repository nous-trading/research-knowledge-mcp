"""bge-m3 Dense Embedder.

Uses FlagEmbedding's BGEM3FlagModel to produce 1024-dimensional normalized
embeddings. Supports MPS (Apple Silicon). Because faiss is CPU-only, the
embedding device and the index device are kept separate.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DenseEmbedder:
    """bge-m3-based dense embedder. Uses lazy loading to conserve memory."""

    _model = None

    def __init__(self, device: str | None = None):
        if device is None:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                # bge-m3 causes a SEGFAULT on MPS (macOS ARM64).
                # Fall back to CPU safely. Takes <1 s for ~3 chunks.
                device = "cpu"
        self.device = device

    def _load(self) -> None:
        if DenseEmbedder._model is not None:
            return
        logger.info("Loading bge-m3 model (device=%s)...", self.device)
        from FlagEmbedding import BGEM3FlagModel

        DenseEmbedder._model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=(self.device != "cpu"),
            device=self.device,
        )
        logger.info("bge-m3 model loaded.")

    @classmethod
    def unload(cls) -> None:
        """Release the model and its accelerator memory (idle TTL)."""
        if cls._model is None:
            return
        cls._model = None
        _release_accelerator_memory()
        logger.info("bge-m3 model unloaded.")

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: Texts to encode
            batch_size: Batch size (auto-selected based on device when None)

        Returns:
            Normalized vector array of shape (N, 1024)
        """
        self._load()
        if batch_size is None:
            batch_size = 16 if self.device != "cpu" else 4

        out = DenseEmbedder._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
        )
        vecs = out["dense_vecs"]  # (N, 1024), already normalized
        return np.array(vecs, dtype=np.float32)


def _release_accelerator_memory() -> None:
    """Return cached accelerator memory to the OS after a model unload.

    On Apple Silicon the MPS caching allocator otherwise keeps multi-GB
    IOAccelerator residency alive for the process lifetime.
    """
    import gc

    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
