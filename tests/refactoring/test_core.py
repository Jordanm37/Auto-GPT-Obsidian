"""Tests for the refactoring core module."""

import pytest
from pathlib import Path
from typing import Dict, Any, Optional

from bidian.refactoring.core import (
    analyze_markdown_structure,
    _slugify,
    generate_toc,
    restructure_headings,
    update_front_matter,
    refactor_note
)
from bidian.indexing.discovery import ParsedNote

# --- Test Data ---

MD_BASIC = """
# Title
Intro.
## Section 1
Content 1.
### Sub 1.1
Content 1.1.
## Section 2
Content 2.
"""

MD_MULTI_H1 = """
# First H1
Text.
# Second H1
More text.
## Section A
# Third H1
Final text.
"""

MD_NO_HEADINGS = "Just plain text.\nNo headings here."

MD_WITH_TOC_SPOT = """
Some content before.

Maybe more.

# First Actual Heading

More content.
"""

FM_BASIC = {"key": "value", "tags": ["a"]}

NOTE_BASIC = ParsedNote(file_path=Path("/vault/basic.md"),
                        content=MD_BASIC, front_matter=FM_BASIC)
NOTE_MULTI_H1 = ParsedNote(file_path=Path("/vault/multi.md"),
                           content=MD_MULTI_H1, front_matter={})
NOTE_NO_HEADINGS = ParsedNote(file_path=Path(
    "/vault/no_headings.md"), content=MD_NO_HEADINGS, front_matter={})
NOTE_TOC_SPOT = ParsedNote(file_path=Path("/vault/toc_spot.md"),
                           content=MD_WITH_TOC_SPOT, front_matter={})

# --- Tests for analyze_markdown_structure ---


def test_analyze_structure_basic():
    structure = analyze_markdown_structure(MD_BASIC)
    assert len(structure["headings"]) == 4
    assert structure["headings"][0] == {"level": 1, "content": "Title"}
    assert structure["headings"][1] == {"level": 2, "content": "Section 1"}
    assert structure["headings"][2] == {"level": 3, "content": "Subsection 1.1"}
    assert structure["headings"][3] == {"level": 2, "content": "Section 2"}


def test_analyze_structure_multi_h1():
    structure = analyze_markdown_structure(MD_MULTI_H1)
    assert len(structure["headings"]) == 4
    assert structure["headings"][0]["level"] == 1
    assert structure["headings"][1]["level"] == 1
    assert structure["headings"][2]["level"] == 2
    assert structure["headings"][3]["level"] == 1


def test_analyze_structure_no_headings():
    structure = analyze_markdown_structure(MD_NO_HEADINGS)
    assert len(structure["headings"]) == 0


def test_analyze_structure_empty():
    structure = analyze_markdown_structure("")
    assert len(structure["headings"]) == 0

# --- Tests for _slugify ---


@pytest.mark.parametrize("text, expected_slug", [
    ("Simple Title", "simple-title"),
    ("Title with spaces and CAPITALS", "title-with-spaces-and-capitals"),
    ("Title with !@#$%^&*() characters?", "title-with--characters"),  # Special chars removed
    ("   Leading/Trailing Spaces/Hyphens--- ", "leadingtrailing-spaceshyphens"),
    ("Unicode Ćhårâctêrs", "unicode-characters"),
    ("Duplicate--Hyphens", "duplicate-hyphens"),
    ("Already-slugified", "already-slugified"),
    ("", ""),
])
def test_slugify(text, expected_slug):
    assert _slugify(text) == expected_slug

# --- Tests for generate_toc ---


def test_generate_toc_basic():
    structure = analyze_markdown_structure(MD_BASIC)
    toc = generate_toc(structure)
    expected = (
        "- [Title](#title)\n"
        "  - [Section 1](#section-1)\n"
        "    - [Subsection 1.1](#subsection-11)"
        # Section 2 is level 2, should be included by default
    )
    # Only check included levels
    assert toc.startswith(expected)
    # Checking the last line separately as max_level default is 3
    assert toc.strip().endswith("  - [Section 2](#section-2)")


def test_generate_toc_max_level():
    structure = analyze_markdown_structure(MD_BASIC)
    toc_l2 = generate_toc(structure, max_level=2)
    expected_l2 = (
        "- [Title](#title)\n"
        "  - [Section 1](#section-1)\n"
        "  - [Section 2](#section-2)"
    )
    assert toc_l2 == expected_l2

    toc_l1 = generate_toc(structure, max_level=1)
    expected_l1 = "- [Title](#title)"
    assert toc_l1 == expected_l1


def test_generate_toc_no_headings():
    structure = analyze_markdown_structure(MD_NO_HEADINGS)
    toc = generate_toc(structure)
    assert toc == ""


def test_generate_toc_duplicate_headings():
    md_dup = "# Title\n## Section\n## Section"
    structure = analyze_markdown_structure(md_dup)
    toc = generate_toc(structure)
    expected = (
        "- [Title](#title)\n"
        "  - [Section](#section)\n"
        "  - [Section](#section-1)"  # Duplicate slug gets -1
    )
    assert toc == expected

# --- Tests for restructure_headings ---


def test_restructure_headings_multiple_h1():
    structure = analyze_markdown_structure(MD_MULTI_H1)
    refactored = restructure_headings(MD_MULTI_H1, structure)
    expected = (
        "# First H1\n"
        "Text.\n"
        "## Second H1\n"
        "More text.\n"
        "## Section A\n"
        "## Third H1\n"
        "Final text."
    ).strip()
    assert refactored.strip() == expected


def test_restructure_headings_single_h1():
    structure = analyze_markdown_structure(MD_BASIC)
    refactored = restructure_headings(MD_BASIC, structure)
    assert refactored == MD_BASIC  # No change expected


def test_restructure_headings_no_h1():
    md_no_h1 = "## Section 1\n### Sub 1"
    structure = analyze_markdown_structure(md_no_h1)
    refactored = restructure_headings(md_no_h1, structure)
    assert refactored == md_no_h1  # No change expected

# --- Tests for update_front_matter ---


def test_update_front_matter_add_and_modify():
    original = {"a": 1, "b": 2}
    updates = {"b": 3, "c": 4}
    result = update_front_matter(original, updates)
    assert result == {"a": 1, "b": 3, "c": 4}
    assert original == {"a": 1, "b": 2}  # Original should not be modified


def test_update_front_matter_empty_original():
    result = update_front_matter({}, {"a": 1})
    assert result == {"a": 1}


def test_update_front_matter_empty_updates():
    result = update_front_matter({"a": 1}, {})
    assert result == {"a": 1}

# --- Tests for refactor_note ---


def test_refactor_note_no_changes():
    """Test refactor when no operations are requested/cause changes."""
    result = refactor_note(NOTE_BASIC)  # No flags set
    assert result is None


def test_refactor_note_update_fm_only():
    """Test refactoring only front matter."""
    updates = {"status": "done", "new_key": True}
    result = refactor_note(NOTE_BASIC, update_fm=updates)
    assert result is not None
    assert "status: done" in result
    assert "new_key: true" in result
    assert "key: value" in result  # Original FM key still there
    assert "# Title" in result  # Content unchanged


def test_refactor_note_add_toc_only():
    """Test refactoring only adding a ToC."""
    result = refactor_note(NOTE_BASIC, add_toc=True)
    assert result is not None
    assert "## Table of Contents" in result
    assert "[Section 1](#section-1)" in result
    assert "[Subsection 1.1](#subsection-11)" in result
    assert "key: value" in result  # Original FM
    assert "Intro." in result  # Original Content


def test_refactor_note_restructure_only():
    """Test refactoring only restructuring headings."""
    result = refactor_note(NOTE_MULTI_H1, run_heading_restructure=True)
    assert result is not None
    assert result.count("\n# ") == 1  # Only one H1 should remain
    assert result.count("\n## ") == 3  # Two demoted H1s + original H2


def test_refactor_note_all_operations():
    """Test applying multiple refactoring operations."""
    fm_updates = {"status": "reviewed", "tags": ["a", "b"]}
    result = refactor_note(
        NOTE_BASIC,
        update_fm=fm_updates,
        add_toc=True,
        toc_max_level=2,  # Limit ToC depth
        run_heading_restructure=True  # Should have no effect here
    )
    assert result is not None
    # Check FM
    assert "status: reviewed" in result
    assert "tags:\n- a\n- b" in result  # Assumes list format
    # Check ToC
    assert "## Table of Contents" in result
    assert "[Title](#title)" in result
    assert "[Section 1](#section-1)" in result
    assert "[Section 2](#section-2)" in result
    assert "Subsection 1.1" not in result  # Excluded by max_level
    # Check content
    assert "Intro." in result


def test_refactor_note_toc_insertion_point(tmp_path):
    """Test where the ToC gets inserted."""
    # Test with content that has leading whitespace/newlines
    note_leading_ws = ParsedNote(file_path=tmp_path / "ws.md",
                                 content="\n  \n# Heading 1\nText", front_matter={})
    result = refactor_note(note_leading_ws, add_toc=True)
    assert result is not None
    expected_start = "---\n# Heading 1"
    toc_section = "## Table of Contents\n\n- [Heading 1](#heading-1)\n\n"
    # ToC should come *before* the first heading
    assert result.strip().startswith(toc_section + "# Heading 1")
