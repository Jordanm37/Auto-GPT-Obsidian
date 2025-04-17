"""Tests for the patching module."""

import pytest
import json
from pathlib import Path

from bidian.common.patching import (
    get_patch_path,
    save_creation_patch,
    rollback_creation_patch,
    save_refactor_patch,
    rollback_refactor_patch,
    PATCH_FILE_SUFFIX,
    PatchError,
    PatchFormatError,
    PatchApplyError
)
# Need diff tool for refactor test
import diff_match_patch as dmp_module
dmp = dmp_module.diff_match_patch()

# --- Fixtures ---


@pytest.fixture
def target_file(tmp_path: Path) -> Path:
    """Create a dummy target file."""
    file = tmp_path / "note.md"
    file.write_text("Initial Content")
    return file


@pytest.fixture
def refactored_content() -> str:
    return "Refactored Content Here"


@pytest.fixture
def diff_list(target_file: Path, refactored_content: str) -> list:
    """Generate a sample diff list."""
    original = target_file.read_text()
    return dmp.diff_main(original, refactored_content)


# --- Tests ---

def test_get_patch_path(tmp_path: Path):
    """Test patch path generation."""
    p1 = tmp_path / "file.md"
    p2 = tmp_path / "file with spaces.md"
    p3 = tmp_path / "file.other.ext"

    assert get_patch_path(p1) == tmp_path / f"file.md{PATCH_FILE_SUFFIX}"
    assert get_patch_path(p2) == tmp_path / f"file with spaces.md{PATCH_FILE_SUFFIX}"
    assert get_patch_path(p3) == tmp_path / f"file.other.ext{PATCH_FILE_SUFFIX}"

# == Creation Patch Tests ==


def test_save_creation_patch(target_file: Path):
    """Test saving a creation patch file."""
    patch_path = save_creation_patch(target_file)
    assert patch_path is not None
    assert patch_path.exists()
    assert patch_path == get_patch_path(target_file)

    # Verify content
    data = json.loads(patch_path.read_text())
    assert data["action"] == "create"
    assert data["target_path"] == str(target_file.resolve())
    assert "timestamp_utc" in data
    assert data["patch_format"] == "metadata_only"


def test_rollback_creation_patch(target_file: Path):
    """Test rolling back a creation patch (deleting files)."""
    patch_path = save_creation_patch(target_file)
    assert patch_path is not None
    assert target_file.exists()
    assert patch_path.exists()

    rollback_creation_patch(patch_path)

    # Check files were deleted
    assert not target_file.exists()
    assert not patch_path.exists()


def test_rollback_creation_target_already_deleted(target_file: Path):
    """Test rollback when the target file is already gone."""
    patch_path = save_creation_patch(target_file)
    assert patch_path is not None
    target_file.unlink()  # Delete target manually
    assert not target_file.exists()
    assert patch_path.exists()

    rollback_creation_patch(patch_path)

    # Patch file should still be deleted
    assert not patch_path.exists()


def test_rollback_creation_invalid_patch(target_file: Path):
    """Test rollback with an invalid patch file format."""
    patch_path = get_patch_path(target_file)
    # Write invalid JSON
    patch_path.write_text("{\"action\": \"wrong\"}")

    with pytest.raises(PatchFormatError):
        rollback_creation_patch(patch_path)
    # Ensure files are not deleted on error
    assert target_file.exists()  # Original should still be there
    assert patch_path.exists()  # Invalid patch should remain


def test_rollback_creation_patch_not_found(tmp_path: Path):
    """Test rollback when the patch file doesn't exist."""
    non_existent_patch = tmp_path / "ghost.md.bidian-patch"
    with pytest.raises(FileNotFoundError):
        rollback_creation_patch(non_existent_patch)

# == Refactor Patch Tests ==


def test_save_refactor_patch(target_file: Path, diff_list: list):
    """Test saving a refactor patch file with original content."""
    original_content = target_file.read_text()
    patch_path = save_refactor_patch(target_file, original_content, diff_list)

    assert patch_path is not None
    assert patch_path.exists()
    assert patch_path == get_patch_path(target_file)

    # Verify content
    data = json.loads(patch_path.read_text())
    assert data["action"] == "refactor"
    assert data["target_path"] == str(target_file.resolve())
    assert data["patch_format"] == "original_content"
    assert data["original_content"] == original_content
    assert "timestamp_utc" in data
    assert "patch_diff_text" in data  # Check diff text was also saved (optional)


def test_rollback_refactor_patch(target_file: Path, refactored_content: str, diff_list: list):
    """Test rolling back a refactor patch by restoring original content."""
    original_content = target_file.read_text()
    # Simulate refactoring: update target file content
    target_file.write_text(refactored_content)
    # Save the patch
    patch_path = save_refactor_patch(target_file, original_content, diff_list)
    assert patch_path is not None
    assert target_file.read_text() == refactored_content  # Verify file was "refactored"

    # Rollback
    rollback_refactor_patch(patch_path)

    # Check file content restored and patch deleted
    assert target_file.read_text() == original_content
    assert not patch_path.exists()


def test_rollback_refactor_invalid_patch(target_file: Path):
    """Test rollback refactor with an invalid patch file."""
    patch_path = get_patch_path(target_file)
    # Write patch missing original_content
    invalid_data = {"action": "refactor", "target_path": str(target_file)}
    patch_path.write_text(json.dumps(invalid_data))

    with pytest.raises(PatchFormatError):
        rollback_refactor_patch(patch_path)
    assert patch_path.exists()


def test_rollback_refactor_target_not_found(target_file: Path, diff_list: list):
    """Test rollback refactor when the target file is missing."""
    original_content = target_file.read_text()
    patch_path = save_refactor_patch(target_file, original_content, diff_list)
    assert patch_path is not None

    target_file.unlink()  # Delete target file
    assert not target_file.exists()

    # Rollback should log a warning but not raise PatchApplyError for missing target,
    # and should delete the patch file.
    rollback_refactor_patch(patch_path)
    assert not patch_path.exists()
