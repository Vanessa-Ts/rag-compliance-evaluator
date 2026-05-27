"""Golden evaluation set loading and validation.

Loads the curated question/answer golden set from ``data/golden/qa.yaml`` and
validates that every ``expected_doc_ids`` value refers to a real document in the
corpus (``data/corpus/*.md``), as derived from each file's YAML frontmatter.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

# Repo root: this file lives at <repo>/src/app/eval/dataset.py, so four parents up.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_GOLDEN_PATH: Path = _REPO_ROOT / "data" / "golden" / "qa.yaml"
_CORPUS_DIR: Path = _REPO_ROOT / "data" / "corpus"


class GoldenItem(BaseModel):
    """A single golden evaluation question and its grounded expectations."""

    id: str
    question: str
    jurisdiction: str
    expected_doc_ids: list[str]
    reference_answer: str


def _parse_frontmatter_doc_id(text: str) -> str | None:
    """Extract the ``doc_id`` from a markdown file's leading YAML frontmatter.

    The frontmatter is the block delimited by the first pair of ``---`` fences at
    the start of the file. Returns ``None`` if no frontmatter or no ``doc_id``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return None
    frontmatter_text = "\n".join(lines[1:closing_index])
    parsed = yaml.safe_load(frontmatter_text)
    if not isinstance(parsed, dict):
        return None
    doc_id = parsed.get("doc_id")
    return doc_id if isinstance(doc_id, str) else None


def _valid_doc_ids(corpus_dir: Path = _CORPUS_DIR) -> set[str]:
    """Build the set of valid corpus ``doc_id`` values from frontmatter."""
    doc_ids: set[str] = set()
    for md_path in sorted(corpus_dir.glob("*.md")):
        doc_id = _parse_frontmatter_doc_id(md_path.read_text(encoding="utf-8"))
        if doc_id is not None:
            doc_ids.add(doc_id)
    return doc_ids


def load_golden(subset: list[str] | None = None) -> list[GoldenItem]:
    """Load and validate the golden evaluation set.

    Parses ``data/golden/qa.yaml`` into :class:`GoldenItem` instances and verifies
    that every ``expected_doc_ids`` value exists in the corpus. If ``subset`` is
    provided, only items whose ``id`` is in ``subset`` are returned.

    Raises:
        ValueError: if any ``expected_doc_ids`` value is not a known corpus doc_id.
    """
    raw = yaml.safe_load(_GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Golden set at {_GOLDEN_PATH} must be a YAML list of items.")

    items: list[GoldenItem] = [GoldenItem(**entry) for entry in raw]

    valid_ids = _valid_doc_ids()
    unknown: set[str] = set()
    for item in items:
        for doc_id in item.expected_doc_ids:
            if doc_id not in valid_ids:
                unknown.add(doc_id)
    if unknown:
        raise ValueError(
            "Unknown expected_doc_ids not present in the corpus: "
            f"{sorted(unknown)}. Valid doc_ids are: {sorted(valid_ids)}."
        )

    if subset is not None:
        wanted = set(subset)
        items = [item for item in items if item.id in wanted]

    return items
