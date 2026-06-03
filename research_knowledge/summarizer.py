"""Haiku-based paper summarization.

Caches the full markdown via prompt caching to reduce cost.
Re-summarization for the same focus is skipped via the summary_cache field.
"""

from __future__ import annotations

import logging


from .auth import get_async_anthropic_client
from .contextualizer import HAIKU_MODEL
from .models import Manifest
from .paths import MANIFEST_PATH, PAPERS_MARKDOWN_DIR

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """\
Summarize this academic paper. Focus on: abstract, key claims, \
formulas, and relevance to {focus}.

Output as:
## Abstract
## Key Claims
## Formulas
## Our Engine Relevance"""


async def summarize_paper(
    paper_id: str,
    focus: str | None = None,
) -> str:
    """Summarize a paper, returning a cached summary if one already exists.

    Args:
        paper_id: Paper identifier.
        focus: Summary focus topic (defaults to 'algorithmic trading').

    Returns:
        Summary text in markdown format.
    """
    focus = focus or "algorithmic trading"
    manifest = Manifest.load(MANIFEST_PATH)

    if paper_id not in manifest.papers:
        return f"Paper not found: {paper_id}"

    paper = manifest.papers[paper_id]

    # Return cached summary if available
    if paper.summary_cache and paper.summary_focus == focus:
        logger.info("Returning cached summary: %s (focus=%s)", paper_id, focus)
        return paper.summary_cache

    # Load markdown
    md_path = PAPERS_MARKDOWN_DIR / f"{paper_id}.md"
    if not md_path.exists():
        return f"Markdown file not found: {md_path}"

    markdown = md_path.read_text(encoding="utf-8")

    # Summarize with Haiku (prompt caching)
    client = get_async_anthropic_client()
    response = await client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=2000,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<paper>\n{markdown}\n</paper>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": SUMMARIZE_PROMPT.format(focus=focus),
                    },
                ],
            }
        ],
        # Prompt caching is GA — no beta header needed (avoids overwriting the OAT beta header).
    )

    summary = response.content[0].text.strip()

    # Persist the summary to cache
    paper.summary_cache = summary
    paper.summary_focus = focus
    manifest.save(MANIFEST_PATH)

    logger.info("Summary generated: %s (focus=%s)", paper_id, focus)
    return summary
