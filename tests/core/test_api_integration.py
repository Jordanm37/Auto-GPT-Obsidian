"""Integration tests for the BidianAgentAPI refactoring flow."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock

# Modules being tested/used
from bidian.core.api import BidianAgentAPI, RefactorProposal
from bidian.core.retriever import Retriever  # For mocking
from bidian.creation.templating import TemplateRenderer  # Need real one for API init
from bidian.common.patching import get_patch_path, PATCH_FILE_SUFFIX  # For verification

# --- Fixtures ---


@pytest.fixture
def mock_retriever(mocker) -> MagicMock:  # type: ignore
    """Mocks the Retriever for API initialization."""
    return MagicMock(spec=Retriever)


@pytest.fixture
def template_renderer(tmp_path: Path) -> TemplateRenderer:
    """Provides a TemplateRenderer instance (no templates needed for refactor)."""
    # Point to temporary dirs, though refactor doesn't use them directly
    custom_dir = tmp_path / "custom_templates"
    default_dir = tmp_path / "default_templates"
    custom_dir.mkdir()
    default_dir.mkdir()
    return TemplateRenderer(custom_template_dir=custom_dir, default_template_dir=default_dir)


@pytest.fixture
def api(mock_retriever, template_renderer, tmp_path: Path) -> BidianAgentAPI:
    """Provides a BidianAgentAPI instance for integration tests."""
    vault_path = tmp_path / "test_vault"
    vault_path.mkdir()
    return BidianAgentAPI(retriever=mock_retriever, template_renderer=template_renderer, vault_path=vault_path)


@pytest.fixture
def sample_note_for_refactor(api: BidianAgentAPI) -> Path:
    """Creates a sample note file in the API's vault path."""
    note_content = """
# Existing Title

Some intro.

## Section One

Content.

## Section Two

More content.

# Another H1

Should be demoted.

"""
    note_path = api.vault_path / "refactor_test.md"
    note_path.write_text(note_content, encoding="utf-8")
    return note_path

# --- Integration Tests ---


def test_refactor_propose_commit_rollback_flow(api: BidianAgentAPI, sample_note_for_refactor: Path):
    """Tests the full propose -> commit -> rollback cycle for refactoring."""
    target_file = sample_note_for_refactor
    original_content = target_file.read_text(encoding="utf-8")
    patch_path = get_patch_path(target_file)

    # --- 1. Propose Refactor ---
    # Request ToC add and heading restructure
    proposal = api.propose_refactor(
        target_file_path=target_file,
        add_toc=True,
        run_heading_restructure=True,
        update_fm={"status": "refactored"}  # Add a front matter change too
    )

    assert proposal is not None
    assert isinstance(proposal, RefactorProposal)
    assert proposal.target_file_path == target_file
    assert proposal.original_content == original_content
    assert proposal.refactored_content != original_content
    # Check refactored content for expected changes
    assert "status: refactored" in proposal.refactored_content
    assert "## Table of Contents" in proposal.refactored_content
    assert "[Section One](#section-one)" in proposal.refactored_content
    assert "\n## Another H1\n" in proposal.refactored_content  # Check demotion
    assert proposal.diff is not None
    assert len(proposal.diff) > 0

    # --- 2. Commit Refactor ---
    assert not patch_path.exists()  # Patch shouldn't exist yet
    commit_success = api.commit_refactor(proposal)
    assert commit_success is True

    # Verify file content was updated
    assert target_file.read_text(encoding="utf-8") == proposal.refactored_content
    # Verify patch file was created
    assert patch_path.exists()

    # Verify patch content (basic checks)
    patch_data = json.loads(patch_path.read_text(encoding="utf-8"))
    assert patch_data["action"] == "refactor"
    assert patch_data["target_path"] == str(target_file.resolve())
    assert patch_data["patch_format"] == "original_content"
    assert patch_data["original_content"] == original_content

    # --- 3. Rollback Patch ---
    rollback_success = api.rollback_patch(patch_path)
    assert rollback_success is True

    # Verify file content is restored
    assert target_file.read_text(encoding="utf-8") == original_content
    # Verify patch file is deleted
    assert not patch_path.exists()


def test_propose_refactor_no_changes(api: BidianAgentAPI, sample_note_for_refactor: Path):
    """Test proposing a refactor that results in no changes."""
    target_file = sample_note_for_refactor
    original_content = target_file.read_text(encoding="utf-8")

    # Propose refactor with options that won't change this specific file
    proposal = api.propose_refactor(
        target_file_path=target_file,
        add_toc=False,  # No ToC
        # Only one H1 initially (incorrect - test data has 2 H1s!) Let's fix test data or the test
        run_heading_restructure=False
        # Let's assume run_heading_restructure IS run, but the *only* change is H1->H2
        # And we request *only* that, and check it returns None if it was already like that.
        # Better: make a note that *won't* be changed by restructure.
    )

    # Let's make a file that won't change
    stable_content = "# Only H1\n\nText."
    stable_note_path = api.vault_path / "stable.md"
    stable_note_path.write_text(stable_content)

    proposal_stable = api.propose_refactor(
        stable_note_path, run_heading_restructure=True)

    assert proposal_stable is None  # Expect None as no changes should be made


def test_rollback_nonexistent_patch(api: BidianAgentAPI):
    """Test rolling back a patch file that does not exist."""
    non_existent_patch = api.vault_path / f"ghost.md{PATCH_FILE_SUFFIX}"
    assert not non_existent_patch.exists()

    rollback_success = api.rollback_patch(non_existent_patch)
    assert rollback_success is False  # Should fail as file doesn't exist


def test_rollback_patch_target_missing(api: BidianAgentAPI, sample_note_for_refactor: Path):
    """Test rolling back when the target file was deleted after commit."""
    target_file = sample_note_for_refactor
    original_content = target_file.read_text(encoding="utf-8")
    patch_path = get_patch_path(target_file)

    # Propose and commit
    proposal = api.propose_refactor(target_file, add_toc=True)
    assert proposal is not None
    commit_success = api.commit_refactor(proposal)
    assert commit_success is True
    assert patch_path.exists()

    # Delete the target file manually
    target_file.unlink()
    assert not target_file.exists()

    # Rollback should still proceed and delete the patch file
    rollback_success = api.rollback_patch(patch_path)
    assert rollback_success is True  # Rollback function considers this 'successful'
    assert not target_file.exists()  # Still deleted
    assert not patch_path.exists()  # Patch file should be gone
