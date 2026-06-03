"""Anthropic Contextual Retrieval implementation.

Following the Anthropic Cookbook pattern, each chunk is situated within the
context of the whole document. Prompt caching cuts the cost of processing
repeated chunks of the same document by ~90%.

Reference: https://www.anthropic.com/news/contextual-retrieval
Authentication is resolved by :mod:`research_knowledge.auth`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tqdm import tqdm

from .auth import get_async_anthropic_client
from .models import Chunk, PaperMeta, RawChunk

logger = logging.getLogger(__name__)

# Default model for contextualization. Override via CONTEXTUALIZER_MODEL env.
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Anthropic Cookbook Contextual Retrieval prompts (verbatim from the cookbook)
DOCUMENT_CONTEXT_PROMPT = """\
<document>
{doc_content}
</document>"""

CHUNK_CONTEXT_PROMPT = """\
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document \
for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""




async def contextualize_chunks(
    paper_meta: PaperMeta,
    full_markdown: str,
    raw_chunks: list[RawChunk],
    concurrency: int = 5,
) -> tuple[list[Chunk], dict[str, int]]:
    """Inject document-level context into each chunk.

    Args:
        paper_meta: Paper metadata.
        full_markdown: Full markdown text of the document.
        raw_chunks: List of raw chunks to contextualize.
        concurrency: Maximum number of concurrent API requests (respects rate limits).

    Returns:
        (contextualized_chunks, token_usage_dict)
    """
    client = get_async_anthropic_client()
    token_usage: dict[str, int] = {
        "input": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "output": 0,
    }

    contextualized: list[Chunk] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_one(chunk: RawChunk) -> Chunk:
        async with semaphore:
            response = await client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=300,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": DOCUMENT_CONTEXT_PROMPT.format(
                                    doc_content=full_markdown
                                ),
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": CHUNK_CONTEXT_PROMPT.format(
                                    chunk_content=chunk.raw_text
                                ),
                            },
                        ],
                    }
                ],
                # Prompt caching was promoted to GA in August 2024 — no beta header needed.
                # extra_headers is intentionally omitted to avoid overwriting the OAT beta header
                # (claude-code-..., oauth-...). cache_control: ephemeral is sufficient.
            )

            context = response.content[0].text.strip()
            contextualized_text = f"{context}\n\n{chunk.raw_text}"

            # Accumulate usage counters (safe: asyncio is single-threaded)
            token_usage["input"] += response.usage.input_tokens
            token_usage["cache_read"] += getattr(
                response.usage, "cache_read_input_tokens", 0
            ) or 0
            token_usage["cache_creation"] += getattr(
                response.usage, "cache_creation_input_tokens", 0
            ) or 0
            token_usage["output"] += response.usage.output_tokens

            return Chunk(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                raw_text=chunk.raw_text,
                token_count=chunk.token_count,
                contextualized_text=contextualized_text,
            )

    # Sequential processing with a progress bar (order preserved to maximise prompt cache hits)
    for chunk in tqdm(raw_chunks, desc="Contextualizing"):
        result = await _process_one(chunk)
        contextualized.append(result)

    logger.info(
        "Contextualization complete: %d chunks, usage: input=%d, cache_read=%d, cache_creation=%d, output=%d",
        len(contextualized),
        token_usage["input"],
        token_usage["cache_read"],
        token_usage["cache_creation"],
        token_usage["output"],
    )

    return contextualized, token_usage


def estimate_cost(
    full_markdown: str,
    raw_chunks: list[RawChunk],
) -> dict[str, Any]:
    """Estimate the ingest cost for a dry run.

    Returns:
        {chunk_count, doc_tokens, estimated_cost_usd, breakdown}
    """
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    doc_tokens = len(encoder.encode(full_markdown))

    chunk_count = len(raw_chunks)
    avg_chunk_tokens = (
        sum(c.token_count for c in raw_chunks) / chunk_count if chunk_count else 0
    )

    # Haiku 4.5 pricing (as of 2026-04)
    # Input: $0.80/MTok, Output: $4.00/MTok
    # Cache read: $0.08/MTok (90% discount)
    # Cache creation: $1.00/MTok (25% surcharge)
    input_per_mtok = 0.80
    cache_read_per_mtok = 0.08
    cache_creation_per_mtok = 1.00
    output_per_mtok = 4.00

    # First chunk: cache_creation (entire document); remaining chunks: cache_read
    first_chunk_input_cost = (doc_tokens / 1_000_000) * cache_creation_per_mtok
    remaining_input_cost = (
        (chunk_count - 1) * (doc_tokens / 1_000_000) * cache_read_per_mtok
        if chunk_count > 1
        else 0
    )
    # Per-chunk prompt portion (cache miss)
    chunk_prompt_cost = (
        chunk_count * (avg_chunk_tokens + 50) / 1_000_000 * input_per_mtok
    )
    # Output: ~50 tokens per chunk
    output_cost = chunk_count * 50 / 1_000_000 * output_per_mtok

    total = first_chunk_input_cost + remaining_input_cost + chunk_prompt_cost + output_cost

    return {
        "chunk_count": chunk_count,
        "doc_tokens": doc_tokens,
        "avg_chunk_tokens": round(avg_chunk_tokens),
        "estimated_cost_usd": round(total, 4),
        "breakdown": {
            "cache_creation": round(first_chunk_input_cost, 4),
            "cache_read": round(remaining_input_cost, 4),
            "chunk_prompts": round(chunk_prompt_cost, 4),
            "output": round(output_cost, 4),
        },
    }
