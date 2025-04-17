"""Tests for the backlinker module."""

import pytest
import json
import re
from pathlib import Path
from typing import List

from bidian.core.retriever import RelatedNoteResult
from bidian.core.backlinker import (
    update_backlinks,
    _format_backlinks,
    _insert_backlinks_idempotent,
    BACKLINK_START_COMMENT,
    BACKLINK_END_COMMENT,
    DEFAULT_BACKLINK_HEADING,
    DEFAULT_LOG_PATH  # To check default log location if needed
)

# --- Fixtures ---


@pytest.fixture
def mock_vault_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Creates paths for a mock vault and some notes within tmp_path."""
    vault_dir = tmp_path / "test_vault"
    vault_dir.mkdir()
    note_a = vault_dir / "Note A.md"
    note_b = vault_dir / "sub" / "Note B.md"
    note_c = vault_dir / "Note C.md"
    # Ensure subdirectory exists
    (vault_dir / "sub").mkdir()
    return vault_dir, note_a, note_b, note_c


@pytest.fixture
def sample_related_notes(mock_vault_paths) -> List[RelatedNoteResult]:  # type: ignore
    """Creates a list of sample RelatedNoteResult objects."""
    _, note_a, note_b, note_c = mock_vault_paths
    return [
        RelatedNoteResult(file_path=note_b, score=0.1,
                          best_chunk_index=0, best_chunk_content="..."),
        RelatedNoteResult(file_path=note_c, score=0.2,
                          best_chunk_index=0, best_chunk_content="..."),
    ]


@pytest.fixture
def temp_log_path(tmp_path: Path) -> Path:
    """Returns a path for a temporary log file."""
    return tmp_path / "test_backlinks.jsonl"

# --- Tests for _format_backlinks ---


def test_format_backlinks_basic(mock_vault_paths, sample_related_notes):  # type: ignore
    """Test basic formatting of backlinks."""
    vault_dir, note_a, note_b, note_c = mock_vault_paths
    # Format for Note A
    formatted = _format_backlinks(sample_related_notes, vault_dir, note_a)
    expected = (
        "- [[sub/Note B|Note B]]\n"
        "- [[Note C|Note C]]"
    )
    assert formatted == expected


def test_format_backlinks_skips_self(mock_vault_paths, sample_related_notes):  # type: ignore
    """Test that the target note itself is excluded from the list."""
    vault_dir, note_a, note_b, note_c = mock_vault_paths
    notes_with_self = sample_related_notes + \
        [RelatedNoteResult(file_path=note_a, score=0.01)]
    # Format for Note A, should exclude itself
    formatted = _format_backlinks(notes_with_self, vault_dir, note_a)
    expected = (
        "- [[sub/Note B|Note B]]\n"
        "- [[Note C|Note C]]"
    )
    assert formatted == expected


def test_format_backlinks_empty_list(mock_vault_paths):  # type: ignore
    """Test formatting an empty list of related notes."""
    vault_dir, note_a, _, _ = mock_vault_paths
    formatted = _format_backlinks([], vault_dir, note_a)
    assert formatted == ""

# --- Tests for _insert_backlinks_idempotent ---


BASE_CONTENT = "# Test Note\n\nSome initial content."
LINKS_MD = "- [[Link 1|Link 1]]\n- [[sub/Link 2|Link 2]]"
EXPECTED_BLOCK = f"\n\n{DEFAULT_BACKLINK_HEADING}\n{BACKLINK_START_COMMENT}\n{LINKS_MD}\n{BACKLINK_END_COMMENT}\n"


@pytest.fixture
def target_file(tmp_path: Path) -> Path:
    """Creates a temporary target file with base content."""
    file = tmp_path / "target.md"
    file.write_text(BASE_CONTENT)
    return file


def test_insert_new_block(target_file: Path):
    """Test inserting a new backlink block into a file."""
    updated = _insert_backlinks_idempotent(target_file, LINKS_MD)
    assert updated is not None
    assert updated.strip() == (BASE_CONTENT + EXPECTED_BLOCK).strip()
    # Check trailing newline consistency
    assert updated.endswith('\n')
    assert not updated.endswith('\n\n')


def test_insert_into_empty_file(tmp_path: Path):
    """Test inserting a block into an empty file."""
    empty_file = tmp_path / "empty.md"
    empty_file.touch()
    updated = _insert_backlinks_idempotent(empty_file, LINKS_MD)
    assert updated is not None
    # Should not have leading newlines if file was empty
    expected = f"{DEFAULT_BACKLINK_HEADING}\n{BACKLINK_START_COMMENT}\n{LINKS_MD}\n{BACKLINK_END_COMMENT}\n"
    assert updated == expected


def test_update_existing_block(target_file: Path):
    """Test updating an existing backlink block."""
    existing_block_content = "- [[Old Link]]"
    existing_block = f"\n\n{DEFAULT_BACKLINK_HEADING}\n{BACKLINK_START_COMMENT}\n{existing_block_content}\n{BACKLINK_END_COMMENT}\n"
    target_file.write_text(BASE_CONTENT + existing_block + "\nMore content.")

    updated = _insert_backlinks_idempotent(target_file, LINKS_MD)
    assert updated is not None
    # Check that the old block is fully replaced
    assert existing_block_content not in updated
    assert LINKS_MD in updated
    # Check that content outside the block is preserved
    assert BASE_CONTENT in updated
    assert "\nMore content." in updated
    # Check overall structure
    # Normalize newlines just in case
    expected_content = (BASE_CONTENT + EXPECTED_BLOCK +
                        "More content.\n").replace("\n\n\n", "\n\n")
    # Normalize test output too for comparison, strip might be too aggressive
    assert updated.replace("\n\n\n", "\n\n") == expected_content


def test_no_change_needed(target_file: Path):
    """Test when the existing block matches the new links."""
    target_file.write_text(BASE_CONTENT + EXPECTED_BLOCK)
    updated = _insert_backlinks_idempotent(target_file, LINKS_MD)
    assert updated is None  # Should return None indicating no change


def test_remove_existing_block_when_empty(target_file: Path):
    """Test removing the block when the new link list is empty."""
    target_file.write_text(BASE_CONTENT + EXPECTED_BLOCK)
    updated = _insert_backlinks_idempotent(target_file, "")  # Empty links
    assert updated is not None
    # Block should be removed
    assert BACKLINK_START_COMMENT not in updated
    assert DEFAULT_BACKLINK_HEADING not in updated
    assert updated.strip() == BASE_CONTENT.strip()


def test_no_change_needed_when_empty_and_no_block(target_file: Path):
    """Test case where new links are empty and no block exists."""
    # File only has base content
    updated = _insert_backlinks_idempotent(target_file, "")  # Empty links
    assert updated is None  # No changes needed

# --- Tests for update_backlinks (integration) ---


# type: ignore
def test_update_backlinks_writes_file_and_logs(mock_vault_paths, sample_related_notes, temp_log_path):
    """Test the main function writing the file and logging."""
    vault_dir, note_a, note_b, note_c = mock_vault_paths
    note_a.write_text(BASE_CONTENT)  # Initial content for Note A

    modified = update_backlinks(
        target_file_path=note_a,
        related_notes=sample_related_notes,
        vault_base_path=vault_dir,
        log_file_path_str=str(temp_log_path)
    )

    assert modified is True
    # Check file content
    content = note_a.read_text()
    assert DEFAULT_BACKLINK_HEADING in content
    assert BACKLINK_START_COMMENT in content
    assert "[[sub/Note B|Note B]]" in content
    assert "[[Note C|Note C]]" in content
    assert BASE_CONTENT in content

    # Check log file content
    assert temp_log_path.exists()
    log_content = temp_log_path.read_text()
    assert len(log_content.strip().split('\n')) == 1  # Should be one JSON line
    log_data = json.loads(log_content)
    assert log_data["target_file"] == str(note_a)
    assert log_data["added_links"] == [
        "- [[sub/Note B|Note B]]",
        "- [[Note C|Note C]]"
    ]
    assert log_data["removed_links"] == []  # Placeholder


# type: ignore
def test_update_backlinks_no_change(mock_vault_paths, sample_related_notes, temp_log_path):
    """Test the main function when no file modification is needed."""
    vault_dir, note_a, note_b, note_c = mock_vault_paths

    # Pre-populate file with the correct content
    links_md = _format_backlinks(sample_related_notes, vault_dir, note_a)
    block = f"\n\n{DEFAULT_BACKLINK_HEADING}\n{BACKLINK_START_COMMENT}\n{links_md}\n{BACKLINK_END_COMMENT}\n"
    note_a.write_text(BASE_CONTENT + block)
    initial_content = note_a.read_text()
    log_exists_before = temp_log_path.exists()

    modified = update_backlinks(
        target_file_path=note_a,
        related_notes=sample_related_notes,
        vault_base_path=vault_dir,
        log_file_path_str=str(temp_log_path)
    )

    assert modified is False
    # Verify file content hasn't changed
    assert note_a.read_text() == initial_content
    # Verify log file wasn't written to (or created if it didn't exist)
    if log_exists_before:
        # Assuming it was empty or check length didn't change
        assert temp_log_path.read_text() == ""
    else:
        assert not temp_log_path.exists()
