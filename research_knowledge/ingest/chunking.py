"""Heading-boundary + token-window chunking.

First splits at heading boundaries, then further splits each section by token window.
Token counting uses tiktoken cl100k_base (Claude-compatible).
"""

from __future__ import annotations

import logging

import tiktoken

from ..models import RawChunk
from .parsing import StructuredBlock

logger = logging.getLogger(__name__)

ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(ENCODER.encode(text))


def chunk_document(
    paper_id: str,
    blocks: list[StructuredBlock],
    target_tokens: int = 800,
    min_tokens: int = 200,
    overlap_tokens: int = 100,
) -> list[RawChunk]:
    """Split structural blocks into token-based chunks.

    Args:
        paper_id: Paper identifier.
        blocks: Parsed structural blocks.
        target_tokens: Target token count per chunk.
        min_tokens: Minimum token count; blocks below this threshold are merged with the next.
        overlap_tokens: Number of overlap tokens between consecutive chunks.

    Returns:
        List of RawChunk objects.
    """
    if not blocks:
        return []

    chunks: list[RawChunk] = []
    idx = 0

    # Iterate blocks and chunk while respecting heading boundaries
    pending_text = ""
    pending_section_path: list[str] = []
    pending_page_start = 0
    pending_page_end = 0

    for block in blocks:
        block_tokens = _count_tokens(block.text)

        # Flush pending text when a heading block is encountered
        if block.level > 0 and pending_text:
            pending_tokens = _count_tokens(pending_text)
            if pending_tokens >= min_tokens:
                # Large chunk: split by token window
                sub_chunks = _split_by_tokens(
                    pending_text, target_tokens, overlap_tokens
                )
                for sc in sub_chunks:
                    chunks.append(
                        RawChunk(
                            chunk_id=f"{paper_id}_{idx:04d}",
                            paper_id=paper_id,
                            section_path=list(pending_section_path),
                            page_start=pending_page_start,
                            page_end=pending_page_end,
                            raw_text=sc,
                            token_count=_count_tokens(sc),
                        )
                    )
                    idx += 1
                pending_text = ""

        # Append the current block to pending text
        if not pending_text:
            pending_section_path = list(block.section_path)
            pending_page_start = block.page
            pending_page_end = block.page
        pending_text += ("\n\n" if pending_text else "") + block.text
        pending_page_end = max(pending_page_end, block.page)

    # Flush the final pending text
    if pending_text.strip():
        pending_tokens = _count_tokens(pending_text)
        if pending_tokens > 0:
            sub_chunks = _split_by_tokens(
                pending_text, target_tokens, overlap_tokens
            )
            for sc in sub_chunks:
                chunks.append(
                    RawChunk(
                        chunk_id=f"{paper_id}_{idx:04d}",
                        paper_id=paper_id,
                        section_path=list(pending_section_path),
                        page_start=pending_page_start,
                        page_end=pending_page_end,
                        raw_text=sc,
                        token_count=_count_tokens(sc),
                    )
                )
                idx += 1

    logger.info("Chunking complete: %d blocks → %d chunks", len(blocks), len(chunks))
    return chunks


def _split_by_tokens(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into token-bounded segments, respecting sentence boundaries."""
    total_tokens = _count_tokens(text)
    if total_tokens <= target_tokens:
        return [text]

    # Split into sentence-level units and combine within the token limit
    sentences = _split_sentences(text)
    result: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        if current_tokens + sent_tokens > target_tokens and current_sentences:
            # Finalise the current chunk
            result.append("\n".join(current_sentences))

            # Overlap: carry the last few sentences into the next chunk
            overlap_sents: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tokens = _count_tokens(s)
                if overlap_count + s_tokens > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_tokens

            current_sentences = overlap_sents + [sent]
            current_tokens = sum(_count_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sent)
            current_tokens += sent_tokens

    if current_sentences:
        chunk_text = "\n".join(current_sentences)
        # Only append if the last chunk is not a duplicate of the previous one
        if not result or chunk_text != result[-1]:
            result.append(chunk_text)

    return result if result else [text]


def _split_sentences(text: str) -> list[str]:
    """Split text into line/paragraph units, treating blank lines as boundaries."""
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            result.append(stripped)
    return result if result else [text]
