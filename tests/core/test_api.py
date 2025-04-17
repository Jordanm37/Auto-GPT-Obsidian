"""Tests for the main API module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bidian.core.api import BidianAgentAPI
from bidian.core.retriever import Retriever, RelatedNoteResult
# We also need parse_markdown_file from discovery for the API method
from bidian.indexing.discovery import ParsedNote

# --- Mocks and Fixtures ---

MOCK_VAULT_PATH = Path("/test/vault")
NOTE_A_PATH = MOCK_VAULT_PATH / "Note A.md"
NOTE_B_PATH = MOCK_VAULT_PATH / "Note B.md"


@pytest.fixture
def mock_retriever(mocker) -> MagicMock:  # type: ignore
    """Mocks the Retriever class."""
    mock = MagicMock(spec=Retriever)
    # Default: return one related note
    mock.find_related_notes.return_value = [
        RelatedNoteResult(file_path=NOTE_B_PATH, score=0.1)
    ]
    return mock


@pytest.fixture
@patch("bidian.core.api.parse_markdown_file")  # Patch where it's used
@patch("bidian.core.api.update_backlinks")   # Patch where it's used
def api_instance(mock_update_backlinks, mock_parse_md, mock_retriever) -> BidianAgentAPI:  # type: ignore
    """Creates a BidianAgentAPI instance with mocked dependencies."""
    # Configure mock for parse_markdown_file
    mock_parse_md.return_value = ParsedNote(
        file_path=NOTE_A_PATH,
        content="Content of Note A",
        front_matter={}
    )
    # Configure mock for update_backlinks
    mock_update_backlinks.return_value = True  # Simulate successful update

    api = BidianAgentAPI(retriever=mock_retriever, vault_path=MOCK_VAULT_PATH)
    return api, mock_parse_md, mock_update_backlinks  # Return mocks for assertions

# --- Tests for BidianAgentAPI ---


def test_curate_backlinks_success(api_instance, mock_retriever):  # type: ignore
    """Test successful backlink curation for a file."""
    api, mock_parse, mock_update = api_instance
    target_file = NOTE_A_PATH

    # Mock the target file check
    with patch.object(Path, 'is_file', return_value=True):
        success = api.curate_backlinks_for_file(target_file)

    assert success is True
    # Verify mocks were called correctly
    mock_parse.assert_called_once_with(target_file.resolve())
    mock_retriever.find_related_notes.assert_called_once_with(
        query_text="Content of Note A")
    mock_update.assert_called_once_with(
        target_file_path=target_file.resolve(),
        # From mock_retriever setup
        related_notes=[RelatedNoteResult(file_path=NOTE_B_PATH, score=0.1)],
        vault_base_path=MOCK_VAULT_PATH.resolve()
        # Check default log path/heading if necessary
    )


def test_curate_backlinks_file_not_found(api_instance):  # type: ignore
    """Test curation when the target file does not exist."""
    api, mock_parse, mock_update = api_instance
    target_file = MOCK_VAULT_PATH / "nonexistent.md"

    # Mock the target file check to return False
    with patch.object(Path, 'is_file', return_value=False):
        success = api.curate_backlinks_for_file(target_file)

    assert success is False
    mock_parse.assert_not_called()
    mock_update.assert_not_called()


def test_curate_backlinks_parse_fails(api_instance):  # type: ignore
    """Test curation when parsing the target file fails."""
    api, mock_parse, mock_update = api_instance
    target_file = NOTE_A_PATH
    mock_parse.return_value = None  # Simulate parsing failure

    with patch.object(Path, 'is_file', return_value=True):
        success = api.curate_backlinks_for_file(target_file)

    assert success is False
    mock_parse.assert_called_once_with(target_file.resolve())
    api.retriever.find_related_notes.assert_not_called()  # Access retriever via api instance
    mock_update.assert_not_called()


def test_curate_backlinks_no_content(api_instance, mock_retriever):  # type: ignore
    """Test curation when the target file has no content."""
    api, mock_parse, mock_update = api_instance
    target_file = NOTE_A_PATH
    # Simulate note with no content
    mock_parse.return_value = ParsedNote(
        file_path=target_file, content="  \n ", front_matter={})

    with patch.object(Path, 'is_file', return_value=True):
        success = api.curate_backlinks_for_file(target_file)

    # Should still proceed to update (potentially removing links)
    assert success is True
    mock_parse.assert_called_once_with(target_file.resolve())
    # Retriever should not be called if content is empty/whitespace
    mock_retriever.find_related_notes.assert_not_called()
    # Update should be called with an empty list of related notes
    mock_update.assert_called_once_with(
        target_file_path=target_file.resolve(),
        related_notes=[],
        vault_base_path=MOCK_VAULT_PATH.resolve()
    )


def test_curate_backlinks_retriever_error(api_instance, mock_retriever):  # type: ignore
    """Test curation when the retriever throws an exception."""
    api, mock_parse, mock_update = api_instance
    target_file = NOTE_A_PATH
    mock_retriever.find_related_notes.side_effect = Exception("Retriever failed")

    with patch.object(Path, 'is_file', return_value=True):
        success = api.curate_backlinks_for_file(target_file)

    assert success is False
    mock_parse.assert_called_once_with(target_file.resolve())
    mock_retriever.find_related_notes.assert_called_once()
    mock_update.assert_not_called()


def test_curate_backlinks_updater_error(api_instance, mock_retriever):  # type: ignore
    """Test curation when the backlink updater throws an exception."""
    api, mock_parse, mock_update = api_instance
    target_file = NOTE_A_PATH
    mock_update.side_effect = Exception("File write failed")

    with patch.object(Path, 'is_file', return_value=True):
        success = api.curate_backlinks_for_file(target_file)

    assert success is False
    mock_parse.assert_called_once()
    mock_retriever.find_related_notes.assert_called_once()
    mock_update.assert_called_once()  # Update is called, but throws error
