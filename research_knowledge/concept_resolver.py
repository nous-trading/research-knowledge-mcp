"""Concept definition lookup.

Reads concept definitions from markdown files (with YAML frontmatter) under the
data root's ``concepts/`` directory. When a concept is missing, returns a
fallback hint pointing at ``search_papers``.
"""

from __future__ import annotations

import logging
import re

from .paths import CONCEPTS_DIR

logger = logging.getLogger(__name__)


def get_concept(name: str) -> dict:
    """Look up a concept definition.

    Args:
        name: Concept name (filename = slugified name)

    Returns:
        {concept, origin, papers, related_concepts, applies_to_modules, status, body}
        or {fallback: "search", message: ...}
    """
    # Normalize filename + guard against path traversal
    slug = _slugify(name)
    if not slug or ".." in slug or "/" in slug or "\\" in slug or slug.startswith("_"):
        return {
            "fallback": "search",
            "message": (
                f"'{name}' is not a valid concept name. "
                f"Use only letters, numbers, and hyphens."
            ),
        }
    concept_path = CONCEPTS_DIR / f"{slug}.md"
    # Extra safety: ensure the resolved path stays inside CONCEPTS_DIR
    try:
        concept_path.resolve().relative_to(CONCEPTS_DIR.resolve())
    except ValueError:
        return {
            "fallback": "search",
            "message": f"'{name}' is not a valid concept name.",
        }

    if not concept_path.exists():
        # Search for similar files
        candidates = _find_similar(name)
        if candidates:
            return {
                "fallback": "similar",
                "message": f"No concept file found for '{name}'. Similar files: {candidates}",
                "similar": candidates,
            }
        return {
            "fallback": "search",
            "message": (
                f"No concept file found for '{name}'. "
                f"Use the search_papers tool to find related papers."
            ),
        }

    # Parse frontmatter
    content = concept_path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content)

    return {
        "concept": frontmatter.get("concept", name),
        "origin": frontmatter.get("origin", ""),
        "papers": frontmatter.get("papers", []),
        "related_concepts": frontmatter.get("related_concepts", []),
        "applies_to_modules": frontmatter.get("applies_to_modules", []),
        "status": frontmatter.get("status", "draft"),
        "body": body.strip(),
    }


def _slugify(name: str) -> str:
    """Convert a concept name to a filename slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s\-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    return slug


def _find_similar(name: str) -> list[str]:
    """Find concept files with names similar to the given name."""
    if not CONCEPTS_DIR.exists():
        return []
    name_lower = name.lower()
    results: list[str] = []
    for f in CONCEPTS_DIR.glob("*.md"):
        if f.name.startswith("_"):
            continue
        stem = f.stem.replace("-", " ")
        if name_lower in stem or stem in name_lower:
            results.append(f.stem)
    return results[:5]


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter (lightweight implementation, no PyYAML dependency)."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    fm_text = parts[1].strip()
    body = parts[2]

    # Simple YAML-like parsing
    result: dict = {}
    current_key = ""
    current_list: list[str] | None = None

    for line in fm_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # list item
        if stripped.startswith("- "):
            if current_list is not None:
                val = stripped[2:].strip().strip('"').strip("'")
                current_list.append(val)
            continue

        # key: value
        if ":" in stripped:
            if current_list is not None and current_key:
                result[current_key] = current_list
                current_list = None

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not value:
                # The next lines may be a list
                current_key = key
                current_list = []
            else:
                result[key] = value
                current_key = key

    # Flush the last list
    if current_list is not None and current_key:
        result[current_key] = current_list

    return result, body
