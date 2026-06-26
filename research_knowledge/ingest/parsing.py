"""Docling PDF parser wrapper.

Converts a PDF into markdown and structured blocks using the Docling v2.x API.
Reference: https://github.com/DS4SD/docling (v2.x)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StructuredBlock:
    """A structural block from a document — heading level, text, and page information."""

    level: int  # 0 = body text, 1 = h1, 2 = h2, ...
    text: str
    page: int = 0
    heading_text: str = ""  # Heading text when this block is a heading
    section_path: list[str] = field(default_factory=list)


def parse_pdf(pdf_path: Path) -> tuple[str, list[StructuredBlock]]:
    """Parse a PDF with Docling and return (markdown, blocks).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        (full_markdown, structured_blocks)
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Convert to markdown
    markdown = doc.export_to_markdown()

    # Extract structural blocks
    blocks = _extract_blocks(markdown)

    logger.info("PDF parsed: %s → %d blocks, %d chars of markdown", pdf_path.name, len(blocks), len(markdown))
    return markdown, blocks


def _extract_blocks(markdown: str) -> list[StructuredBlock]:
    """Extract heading-based structural blocks from markdown.

    Recognises heading boundaries (#, ##, ###, ...) and splits the document into per-section blocks.
    """
    lines = markdown.split("\n")
    blocks: list[StructuredBlock] = []
    current_heading_stack: list[str] = []
    current_text_lines: list[str] = []
    current_level = 0

    def _flush() -> None:
        text = "\n".join(current_text_lines).strip()
        if text:
            blocks.append(
                StructuredBlock(
                    level=current_level,
                    text=text,
                    section_path=list(current_heading_stack),
                )
            )
        current_text_lines.clear()

    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        m = heading_re.match(line)
        if m:
            _flush()
            level = len(m.group(1))
            heading_text = m.group(2).strip()

            # Manage the heading stack: pop entries at the same or deeper level
            while current_heading_stack and len(current_heading_stack) >= level:
                current_heading_stack.pop()
            current_heading_stack.append(heading_text)
            current_level = level

            # Include the heading line itself in the block text
            current_text_lines.append(line)
        else:
            current_text_lines.append(line)

    _flush()
    return blocks


def extract_title_from_markdown(markdown: str) -> str:
    """Extract the title from the first heading in markdown, or fall back to the first line.

    Docling often emits the paper title as ## (h2), so all heading levels are checked.
    """
    for line in markdown.split("\n"):
        line = line.strip()
        # Use the first heading as the title (h1–h6)
        if re.match(r"^#{1,6}\s+", line):
            return re.sub(r"^#{1,6}\s+", "", line).strip()
    # No heading found — return the first non-empty line
    for line in markdown.split("\n"):
        line = line.strip()
        if line:
            return line[:200]
    return "Untitled"


def extract_authors_from_markdown(markdown: str) -> list[str]:
    """Attempt to extract authors from the early part of the markdown using heuristics."""
    # The lines immediately after the title often contain the authors
    lines = [line.strip() for line in markdown.split("\n") if line.strip()]
    if len(lines) < 2:
        return []

    # Skip the first heading (h1–h6)
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+", line):
            start = i + 1
            break

    # Look for author patterns in the 2–3 lines immediately after the title
    authors: list[str] = []
    for line in lines[start : start + 5]:
        if line.startswith("#"):
            break
        # Patterns: "Name1, Name2, Name3" or "Name1 and Name2"
        if "," in line or " and " in line:
            # Lines that are too long are likely body text
            if len(line) < 500:
                # Split on commas and "and"
                parts = re.split(r",\s*|\s+and\s+", line)
                for p in parts:
                    p = p.strip()
                    # Check if this looks like a name (1–6 words, starts with uppercase)
                    words = p.split()
                    if 1 <= len(words) <= 6 and words[0][0].isupper():
                        # Strip trailing year (e.g. "Bob Johnson 2024" → "Bob Johnson")
                        p = re.sub(r"\s+\d{4}$", "", p).strip()
                        if p:
                            authors.append(p)
                if authors:
                    break

    return authors
