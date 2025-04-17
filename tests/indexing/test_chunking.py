"""Tests for the chunking module."""

import pytest
from pathlib import Path

from bidian.indexing.discovery import ParsedNote
from bidian.indexing.chunking import chunk_by_paragraph, TextChunk


# --- Test Data ---

TEST_NOTE_CONTENT = """
First paragraph.
This should be one chunk.

Second paragraph, a bit longer.
It has multiple lines.



Third paragraph, separated by multiple newlines.

Short.

Final paragraph that meets the minimum length requirement.

"""

# Create a dummy ParsedNote for testing
dummy_path = Path("/path/to/dummy/note.md")
dummy_note = ParsedNote(file_path=dummy_path, content=TEST_NOTE_CONTENT)
empty_note = ParsedNote(file_path=dummy_path, content="")
short_note = ParsedNote(file_path=dummy_path, content="Too short.")
no_double_newline_note = ParsedNote(
    file_path=dummy_path, content="Line 1\nLine 2\nLine 3")

# --- Tests for chunk_by_paragraph ---


def test_chunk_by_paragraph_basic():
    """Test basic paragraph splitting with default min length."""
    chunks = list(chunk_by_paragraph(dummy_note))

    assert len(chunks) == 3

    # Check content (stripping is done by the function)
    assert chunks[0].content == "First paragraph.\nThis should be one chunk."
    assert chunks[1].content == "Second paragraph, a bit longer.\nIt has multiple lines."
    assert chunks[2].content == "Final paragraph that meets the minimum length requirement."

    # Check indices and source path
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2
    assert all(c.source_path == dummy_path for c in chunks)


def test_chunk_by_paragraph_min_length():
    """Test filtering chunks based on min_chunk_length."""
    # Set min length high enough to exclude the first paragraph
    chunks = list(chunk_by_paragraph(dummy_note, min_chunk_length=40))

    assert len(chunks) == 2
    assert chunks[0].content == "Second paragraph, a bit longer.\nIt has multiple lines."
    assert chunks[1].content == "Final paragraph that meets the minimum length requirement."
    assert chunks[0].chunk_index == 0  # Index resets after skipping
    assert chunks[1].chunk_index == 1


def test_chunk_by_paragraph_very_high_min_length():
    """Test when min_chunk_length excludes all chunks."""
    chunks = list(chunk_by_paragraph(dummy_note, min_chunk_length=1000))
    assert len(chunks) == 0


def test_chunk_by_paragraph_empty_content():
    """Test chunking with empty note content."""
    chunks = list(chunk_by_paragraph(empty_note))
    assert len(chunks) == 0


def test_chunk_by_paragraph_short_content():
    """Test chunking content shorter than default min length."""
    chunks = list(chunk_by_paragraph(short_note))
    assert len(chunks) == 0

    chunks_allow_short = list(chunk_by_paragraph(short_note, min_chunk_length=5))
    assert len(chunks_allow_short) == 1
    assert chunks_allow_short[0].content == "Too short."


def test_chunk_by_paragraph_no_double_newlines():
    """Test content without double newlines (should be one chunk)."""
    chunks = list(chunk_by_paragraph(no_double_newline_note))
    assert len(chunks) == 1
    assert chunks[0].content == "Line 1\nLine 2\nLine 3"
    assert chunks[0].chunk_index == 0
