"""Tests for the indexer module."""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

from bidian.indexing.indexer import Indexer, DEFAULT_INDEX_STATE_PATH, NO_CURATE_KEY
from bidian.indexing.discovery import ParsedNote
from bidian.indexing.chunking import TextChunk
from bidian.indexing.embedding import EmbeddingGenerator
from bidian.indexing.vector_store import ChromaVectorStore

# --- Mocks and Fixtures ---

# Mock data
MOCK_VAULT_PATH = Path("/test/vault")
MOCK_STATE_PATH = Path("/test/data/state.json")

FILE_A = MOCK_VAULT_PATH / "Note A.md"
FILE_B = MOCK_VAULT_PATH / "Note B.md"
FILE_C = MOCK_VAULT_PATH / "subdir" / "Note C.md"  # New file later
FILE_D = MOCK_VAULT_PATH / "Note D (exclude).md"  # Excluded by frontmatter
FILE_IGNORED = MOCK_VAULT_PATH / "ignored" / "Ignored.md"  # Ignored by gitignore


@pytest.fixture
def mock_embedder(mocker):  # type: ignore
    mock = MagicMock(spec=EmbeddingGenerator)
    mock.get_embedding_dim.return_value = 3
    mock.generate_embeddings.side_effect = lambda texts: [[0.1] * 3 for _ in texts]
    return mock


@pytest.fixture
def mock_vector_store(mocker):  # type: ignore
    mock = MagicMock(spec=ChromaVectorStore)
    mock.count.return_value = 0
    # Use dictionaries to simulate storage for add/delete/query checks if needed
    mock._storage = {}

    def _add(chunks, embeddings):
        for i, chunk in enumerate(chunks):
            mock._storage[f"{chunk.source_path}::{chunk.chunk_index}"] = (
                chunk, embeddings[i])

    def _delete(source_path):
        keys_to_del = [k for k in mock._storage if k.startswith(str(source_path))]
        for k in keys_to_del:
            del mock._storage[k]
    mock.add_chunks.side_effect = _add
    mock.delete_chunks_by_source.side_effect = _delete
    mock.count.side_effect = lambda: len(mock._storage)
    return mock


@pytest.fixture
def mock_discovery(mocker):  # type: ignore
    """Mocks discovery functions."""
    # Mock find_markdown_files
    mock_find = mocker.patch("bidian.indexing.indexer.find_markdown_files")
    # Mock parse_markdown_file
    mock_parse = mocker.patch("bidian.indexing.indexer.parse_markdown_file")
    # Mock chunk_by_paragraph
    mock_chunk = mocker.patch("bidian.indexing.indexer.chunk_by_paragraph")
    return mock_find, mock_parse, mock_chunk


@pytest.fixture
def indexer_instance(mock_embedder, mock_vector_store, tmp_path):  # type: ignore
    """Creates an Indexer instance with mocked dependencies and temp state file."""
    state_file = tmp_path / "test_state.json"
    # Mock Path object methods for state file handling
    with patch.object(Path, 'exists') as mock_exists, \
            patch.object(Path, 'read_text') as mock_read, \
            patch.object(Path, 'write_text') as mock_write, \
            patch.object(Path, 'mkdir') as mock_mkdir:

        mock_exists.return_value = False  # Start with no state file

        instance = Indexer(
            vault_path=MOCK_VAULT_PATH,
            embedding_generator=mock_embedder,
            vector_store=mock_vector_store,
            state_path=str(state_file)
        )
    return instance, state_file

# Helper to configure discovery mocks for a scenario


def configure_discovery_mocks(mock_discovery_funcs, current_files_mtimes, parsed_notes_map, chunks_map):
    mock_find, mock_parse, mock_chunk = mock_discovery_funcs

    # Configure find_markdown_files mock
    mock_find.return_value = iter(current_files_mtimes.keys())  # Yield paths

    # Configure parse_markdown_file mock
    def parse_side_effect(file_path):
        return parsed_notes_map.get(file_path)
    mock_parse.side_effect = parse_side_effect

    # Configure chunk_by_paragraph mock
    def chunk_side_effect(parsed_note):
        # Need to handle potential None if parse failed, though map avoids it here
        return chunks_map.get(parsed_note.file_path, [])
    mock_chunk.side_effect = chunk_side_effect

    # Mock stat().st_mtime for the paths
    # We need to patch Path.stat itself
    def mock_stat(self):
        mock_stat_result = MagicMock()
        mock_stat_result.st_mtime = current_files_mtimes.get(
            self, time.time())  # Path obj is self
        return mock_stat_result

    # Apply the patch using a context manager within the test function usually,
    # or apply globally if careful. Applying here for simplicity, but better in test.
    # This might need refinement based on how Path objects are created/used.
    # Patching __new__ or specific Path instances might be needed.
    # For now, assume we can patch the method globally for the test duration.
    return patch("pathlib.Path.stat", mock_stat)


# --- Tests for Indexer ---

def test_indexer_init_no_state(indexer_instance):  # type: ignore
    """Test initialization when no state file exists."""
    indexer, state_file = indexer_instance
    assert indexer.vault_path == MOCK_VAULT_PATH
    assert indexer._indexed_mtimes == {}
    # mkdir should have been called by Path(state_file).parent.mkdir()
    # This requires patching Path methods correctly during fixture setup.


def test_indexer_init_with_state(mock_embedder, mock_vector_store, tmp_path):  # type: ignore
    """Test initialization when a state file exists."""
    state_file = tmp_path / "existing_state.json"
    initial_state = {str(FILE_A): 1000.0}
    state_file.write_text(json.dumps(initial_state))

    indexer = Indexer(
        vault_path=MOCK_VAULT_PATH,
        embedding_generator=mock_embedder,
        vector_store=mock_vector_store,
        state_path=str(state_file)
    )
    assert indexer._indexed_mtimes == initial_state


# type: ignore
def test_run_incremental_update_first_run(indexer_instance, mock_discovery, mock_embedder, mock_vector_store):
    """Test the first incremental update when all files are new."""
    indexer, state_file = indexer_instance
    mock_find, mock_parse, mock_chunk = mock_discovery

    # Scenario: Two files exist initially
    current_files = {FILE_A: 100.0, FILE_B: 110.0}
    parsed_notes = {
        FILE_A: ParsedNote(file_path=FILE_A, content="Content A", front_matter={}),
        FILE_B: ParsedNote(file_path=FILE_B, content="Content B", front_matter={}),
    }
    chunks = {
        FILE_A: [TextChunk(source_path=FILE_A, chunk_index=0, content="Content A")],
        FILE_B: [TextChunk(source_path=FILE_B, chunk_index=0, content="Content B")]
    }

    with configure_discovery_mocks(mock_discovery, current_files, parsed_notes, chunks):
        stats = indexer.run_incremental_update(
            respect_gitignore=False)  # Easier first test

    # Assertions
    assert stats["added"] == 2
    assert stats["updated"] == 0
    assert stats["deleted"] == 0
    assert stats["excluded_fm"] == 0
    assert stats["processed_chunks"] == 2
    mock_vector_store.add_chunks.assert_called()
    assert mock_vector_store.add_chunks.call_count == 2  # Called once per file with chunks
    assert indexer._indexed_mtimes == {str(FILE_A): 100.0, str(FILE_B): 110.0}
    # Check if state was saved (requires patching Path.write_text)


# type: ignore
def test_run_incremental_update_no_changes(indexer_instance, mock_discovery, mock_vector_store):
    """Test incremental update when no files have changed."""
    indexer, state_file = indexer_instance
    mock_find, mock_parse, mock_chunk = mock_discovery

    # Scenario: Files A and B exist and are already indexed with same mtime
    initial_mtimes = {str(FILE_A): 100.0, str(FILE_B): 110.0}
    indexer._indexed_mtimes = initial_mtimes.copy()
    current_files = {FILE_A: 100.0, FILE_B: 110.0}  # Same mtimes
    parsed_notes = {}
    chunks = {}

    with configure_discovery_mocks(mock_discovery, current_files, parsed_notes, chunks):
        stats = indexer.run_incremental_update(respect_gitignore=False)

    assert stats["added"] == 0
    assert stats["updated"] == 0
    assert stats["deleted"] == 0
    assert stats["excluded_fm"] == 0
    assert stats["processed_chunks"] == 0
    mock_vector_store.add_chunks.assert_not_called()
    mock_vector_store.delete_chunks_by_source.assert_not_called()
    assert indexer._indexed_mtimes == initial_mtimes


# type: ignore
def test_run_incremental_update_modified_file(indexer_instance, mock_discovery, mock_vector_store, mock_embedder):
    """Test incremental update with one modified file."""
    indexer, state_file = indexer_instance
    mock_find, mock_parse, mock_chunk = mock_discovery

    # Scenario: File A modified, File B unchanged
    initial_mtimes = {str(FILE_A): 100.0, str(FILE_B): 110.0}
    indexer._indexed_mtimes = initial_mtimes.copy()
    current_files = {FILE_A: 120.0, FILE_B: 110.0}  # File A has new mtime
    parsed_notes = {
        FILE_A: ParsedNote(file_path=FILE_A, content="New Content A", front_matter={}),
        # FILE_B shouldn't be parsed if mtime matches
    }
    chunks = {
        FILE_A: [TextChunk(source_path=FILE_A, chunk_index=0, content="New Content A")]
    }

    with configure_discovery_mocks(mock_discovery, current_files, parsed_notes, chunks):
        stats = indexer.run_incremental_update(respect_gitignore=False)

    assert stats["added"] == 0
    assert stats["updated"] == 1
    assert stats["deleted"] == 0
    assert stats["excluded_fm"] == 0
    assert stats["processed_chunks"] == 1
    # Delete should be called for FILE_A before adding
    mock_vector_store.delete_chunks_by_source.assert_called_once_with(FILE_A)
    # Add should be called for FILE_A
    mock_vector_store.add_chunks.assert_called_once()
    assert indexer._indexed_mtimes == {str(FILE_A): 120.0, str(FILE_B): 110.0}


# type: ignore
def test_run_incremental_update_deleted_file(indexer_instance, mock_discovery, mock_vector_store):
    """Test incremental update with one deleted file."""
    indexer, state_file = indexer_instance
    mock_find, mock_parse, mock_chunk = mock_discovery

    # Scenario: File B deleted, File A unchanged
    initial_mtimes = {str(FILE_A): 100.0, str(FILE_B): 110.0}
    indexer._indexed_mtimes = initial_mtimes.copy()
    current_files = {FILE_A: 100.0}  # File B missing
    parsed_notes = {}
    chunks = {}

    with configure_discovery_mocks(mock_discovery, current_files, parsed_notes, chunks):
        stats = indexer.run_incremental_update(respect_gitignore=False)

    assert stats["added"] == 0
    assert stats["updated"] == 0
    assert stats["deleted"] == 1
    assert stats["excluded_fm"] == 0
    assert stats["processed_chunks"] == 0
    # Delete should be called for FILE_B
    mock_vector_store.delete_chunks_by_source.assert_called_once_with(FILE_B)
    mock_vector_store.add_chunks.assert_not_called()
    assert indexer._indexed_mtimes == {str(FILE_A): 100.0}  # File B removed


# type: ignore
def test_run_incremental_update_new_file(indexer_instance, mock_discovery, mock_vector_store, mock_embedder):
    """Test incremental update with one new file added."""
    indexer, state_file = indexer_instance
    mock_find, mock_parse, mock_chunk = mock_discovery

    # Scenario: File A exists, File C is new
    initial_mtimes = {str(FILE_A): 100.0}
    indexer._indexed_mtimes = initial_mtimes.copy()
    current_files = {FILE_A: 100.0, FILE_C: 130.0}  # File C added
    parsed_notes = {
        FILE_C: ParsedNote(file_path=FILE_C, content="Content C", front_matter={})
    }
    chunks = {
        FILE_C: [TextChunk(source_path=FILE_C, chunk_index=0, content="Content C")]
    }

    with configure_discovery_mocks(mock_discovery, current_files, parsed_notes, chunks):
        stats = indexer.run_incremental_update(respect_gitignore=False)

    assert stats["added"] == 1
    assert stats["updated"] == 0
    assert stats["deleted"] == 0
    assert stats["excluded_fm"] == 0
    assert stats["processed_chunks"] == 1
    mock_vector_store.delete_chunks_by_source.assert_not_called()
    # Add should be called for FILE_C
    mock_vector_store.add_chunks.assert_called_once()
    assert indexer._indexed_mtimes == {str(FILE_A): 100.0, str(FILE_C): 130.0}


# type: ignore
def test_run_incremental_update_frontmatter_exclude(indexer_instance, mock_discovery, mock_vector_store):
    """Test skipping a file due to no-curate frontmatter."""
    indexer, state_file = indexer_instance
    mock_find, mock_parse, mock_chunk = mock_discovery

    # Scenario: File D is new but has no-curate: true
    current_files = {FILE_D: 140.0}
    parsed_notes = {
        FILE_D: ParsedNote(file_path=FILE_D, content="Content D",
                           front_matter={NO_CURATE_KEY: True})
    }
    chunks = {}

    with configure_discovery_mocks(mock_discovery, current_files, parsed_notes, chunks):
        stats = indexer.run_incremental_update(respect_gitignore=False)

    assert stats["added"] == 0  # Not counted as added if excluded by FM
    assert stats["updated"] == 0
    assert stats["deleted"] == 0
    assert stats["excluded_fm"] == 1
    assert stats["processed_chunks"] == 0
    # Delete might be called if the file existed before (to clean up), but not add
    # mock_vector_store.delete_chunks_by_source.assert_called_once_with(FILE_D) # If it existed before
    mock_vector_store.add_chunks.assert_not_called()
    # mtime should still be recorded so we don't re-process unless it changes
    assert indexer._indexed_mtimes == {str(FILE_D): 140.0}


# type: ignore
def test_run_incremental_update_gitignore_exclude(indexer_instance, mock_discovery, mock_vector_store):
    """Test skipping a file due to gitignore (mocked via find_markdown_files)."""
    indexer, state_file = indexer_instance
    mock_find, mock_parse, mock_chunk = mock_discovery

    # Scenario: find_markdown_files is mocked to *not* return FILE_IGNORED
    current_files_on_disk = {FILE_A: 100.0, FILE_IGNORED: 150.0}
    # Mock find_markdown_files to only yield FILE_A when respect_gitignore=True

    def find_side_effect(vault_path, respect_gitignore=True):
        if respect_gitignore:
            yield FILE_A
        else:
            yield FILE_A
            yield FILE_IGNORED
    mock_find.side_effect = find_side_effect

    parsed_notes = {
        FILE_A: ParsedNote(file_path=FILE_A, content="Content A", front_matter={}),
        # FILE_IGNORED should not be parsed
    }
    chunks = {
        FILE_A: [TextChunk(source_path=FILE_A, chunk_index=0, content="Content A")],
    }

    # Use a context manager for stat patching here
    def mock_stat(self):
        mock_stat_result = MagicMock()
        mock_stat_result.st_mtime = current_files_on_disk.get(self, time.time())
        return mock_stat_result

    with patch("pathlib.Path.stat", mock_stat):
        # Run with respect_gitignore=True
        stats = indexer.run_incremental_update(respect_gitignore=True)

    assert stats["added"] == 1
    assert stats["updated"] == 0
    assert stats["deleted"] == 0
    assert stats["excluded_fm"] == 0
    assert stats["processed_chunks"] == 1
    # FILE_IGNORED should not have been processed or added to mtimes
    assert str(FILE_IGNORED) not in indexer._indexed_mtimes
    assert indexer._indexed_mtimes == {str(FILE_A): 100.0}

# TODO: Add tests for run_full_reindex
# TODO: Add tests for handling of specific errors during processing steps
# TODO: Refine mock setup for Path objects if needed for state saving/loading tests
