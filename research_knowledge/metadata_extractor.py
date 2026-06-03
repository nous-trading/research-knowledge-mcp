"""Academic PDF metadata extraction — Claude Haiku-based.

Overcomes the limitations of pure heuristics:
- Endorsement pages (e.g., "Praise for ...")
- Cover pages containing only a page number ("1")
- Author names mixed with affiliations and titles
- Meta lines such as "First version: April 15" appearing where authors would be

The LLM reads the first 5 pages of the academic PDF as markdown and extracts
the title, authors, year, and DOI accurately.
Fallback: heuristics in parsing.py remain available when the LLM call fails.
"""

from __future__ import annotations

import json
import logging
import re

import anthropic

from .contextualizer import HAIKU_MODEL

logger = logging.getLogger(__name__)


METADATA_EXTRACTION_PROMPT = """\
You extract metadata from academic PDFs. Return STRICT JSON ONLY.

Academic PDFs sometimes start with non-paper pages — skip them and find the real title page:
- Praise/endorsement pages (e.g., "Praise for [book title]") — skip, real title is later
- Cover pages with just a number (e.g., "1") — skip
- Table of contents — skip
- Author affiliations mixed with names — separate cleanly (drop "Professor", "University", emails, "First version: ...")

Return this exact JSON shape:
{{"title": "...", "authors": ["...", "..."], "year": NNNN_or_null, "doi": "..."_or_null}}

Rules:
1. NEVER hallucinate. If a field is genuinely unclear, use null.
2. Title: the paper/book/article's main title (not section headers like "ABSTRACT").
3. Authors: each entry is "First Last" or "First M. Last". Drop affiliations, emails, dates, "First version: ..." text.
4. Year: 4-digit publication year if explicitly stated; else null.
5. DOI: pattern "10.NNNN/..." if present in first pages; else null.

<paper_first_pages>
{first_pages_md}
</paper_first_pages>

JSON:"""


async def extract_metadata_haiku(
    first_pages_md: str,
    client: anthropic.AsyncAnthropic,
) -> dict | None:
    """Extract metadata from the first 5 pages of markdown using Claude Haiku.

    Args:
        first_pages_md: First ~10,000 characters of markdown (approximately 5 pages).
        client: AsyncAnthropic client authenticated via OAT.

    Returns:
        {title, authors, year, doi} dict, or None on parse failure.
    """
    # Truncate input to reduce Haiku cost and improve cache efficiency
    truncated = first_pages_md[:10000]
    prompt = METADATA_EXTRACTION_PROMPT.format(first_pages_md=truncated)

    try:
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=600,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        logger.warning("Haiku metadata extraction call failed: %s", e)
        return None

    # Guard against empty responses or non-text blocks (e.g. when Haiku stops at max_tokens
    # without producing text, or returns only a tool_use block — prevents IndexError/AttributeError)
    if not response.content or not hasattr(response.content[0], "text"):
        logger.warning("Haiku response contains no text block (stop_reason=%s)", getattr(response, "stop_reason", "unknown"))
        return None

    raw = response.content[0].text.strip()
    # Extract JSON only (handles cases where the LLM adds a ```json ... ``` block or a text prefix)
    json_text = _strip_json_envelope(raw)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON from Haiku response: %s\nResponse: %s", e, raw[:300])
        return None

    # Schema validation
    if not isinstance(data, dict):
        return None
    title = data.get("title")
    authors = data.get("authors")
    year = data.get("year")
    doi = data.get("doi")

    # Type normalisation
    if title is not None and not isinstance(title, str):
        title = str(title)
    if not isinstance(authors, list):
        authors = []
    else:
        authors = [str(a).strip() for a in authors if a and isinstance(a, (str, int))]
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None
    if doi is not None and not isinstance(doi, str):
        doi = None

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
    }


def _strip_json_envelope(text: str) -> str:
    """Extract the raw JSON body from an LLM response.

    Handles:
    - ```json ... ``` code blocks
    - Plain text prefix/suffix
    - Extraction from the first '{' to the last '}'
    """
    # Strip code block wrapper
    m = re.search(r"```(?:json)?\s*\n?(.+?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back to the substring from the first '{' to the last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def is_metadata_complete(meta: dict | None) -> bool:
    """Return True if the metadata is minimally usable."""
    if not meta:
        return False
    # A title of at least 5 characters is sufficient; authors and year are optional
    title = meta.get("title")
    return bool(title and isinstance(title, str) and len(title.strip()) >= 5)
