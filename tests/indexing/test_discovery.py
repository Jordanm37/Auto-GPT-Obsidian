"""Tests for the discovery module."""

import pytest
from pathlib import Path

from bidian.indexing.discovery import (
    find_markdown_files,
    parse_markdown_file,
    find_and_parse_markdown_files,
    ParsedNote
)


# --- Fixtures ---

@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Creates a temporary directory structure simulating an Obsidian vault."""
    vault_dir = tmp_path / "test_vault"
    sub_dir = vault_dir / "Subfolder"
    ignored_dir = vault_dir / "Ignored Folder"
    vault_dir.mkdir()
    sub_dir.mkdir()
    ignored_dir.mkdir()

    # Create some markdown files
    (vault_dir /
     "Note A.md").write_text("---\ntitle: Note A\ntags: [tag1, tag2]\n---\nContent of Note A.")
    (vault_dir / "Note B.md").write_text("Content of Note B, no front matter.")
    (sub_dir /
     "Note C.md").write_text("---\naliases: [C]\nno-curate: true\n---\nContent of Note C.")
    (vault_dir / "Empty Note.md").write_text("")
    (ignored_dir / "Ignored Note.md").write_text("This should be ignored.")

    # Create other files/dirs
    (vault_dir / "image.png").touch()
    (vault_dir / "README.md").write_text("This is a readme, maybe ignored?")  # Case check
    (sub_dir / ".hidden_file.md").write_text("Hidden markdown file")
    (vault_dir / ".obsidian").mkdir()
    (vault_dir / ".obsidian" / "config").touch()

    # Create .gitignore
    (vault_dir / ".gitignore").write_text("Ignored Folder/\n.obsidian/\n*.png\nREADME.md")

    return vault_dir


# --- Tests for find_markdown_files ---

def test_find_markdown_files_basic(mock_vault: Path):
    """Test finding all markdown files without gitignore."""
    found_files = sorted(list(find_markdown_files(mock_vault, respect_gitignore=False)))
    assert len(found_files) == 5  # A, B, C, Empty, Hidden, Ignored, README?
    # Check expected paths (adjust based on actual filtering logic)
    # We expect Note A, Note B, Note C, Empty Note, .hidden_file.md, Ignored Note, README.md
    expected_names = sorted(["Note A.md", "Note B.md", "Note C.md",
                            "Empty Note.md", ".hidden_file.md", "Ignored Note.md", "README.md"])
    actual_names = sorted([p.name for p in found_files])
    # Correct the assertion based on actual expectation without gitignore
    # It should find all .md files regardless of location initially
    assert actual_names == expected_names


def test_find_markdown_files_with_gitignore(mock_vault: Path):
    """Test finding markdown files respecting .gitignore rules."""
    found_files = sorted(list(find_markdown_files(mock_vault, respect_gitignore=True)))
    # Should ignore: Ignored Note.md (in Ignored Folder/), README.md (explicitly), .obsidian/*
    # Should find: Note A, Note B, Note C, Empty Note, .hidden_file.md
    assert len(found_files) == 5
    expected_names = sorted(
        ["Note A.md", "Note B.md", "Note C.md", "Empty Note.md", ".hidden_file.md"])
    actual_names = sorted([p.name for p in found_files])
    assert actual_names == expected_names


def test_find_markdown_files_nonexistent_path(tmp_path: Path):
    """Test finding files in a path that doesn't exist."""
    non_existent_path = tmp_path / "non_existent_vault"
    with pytest.raises(FileNotFoundError):
        list(find_markdown_files(non_existent_path))

# --- Tests for parse_markdown_file ---


def test_parse_markdown_with_front_matter(mock_vault: Path):
    """Test parsing a file with valid YAML front-matter."""
    note_a_path = mock_vault / "Note A.md"
    parsed = parse_markdown_file(note_a_path)
    assert isinstance(parsed, ParsedNote)
    assert parsed.file_path == note_a_path
    assert parsed.front_matter == {"title": "Note A", "tags": ["tag1", "tag2"]}
    assert parsed.content == "Content of Note A."


def test_parse_markdown_without_front_matter(mock_vault: Path):
    """Test parsing a file with no front-matter."""
    note_b_path = mock_vault / "Note B.md"
    parsed = parse_markdown_file(note_b_path)
    assert isinstance(parsed, ParsedNote)
    assert parsed.file_path == note_b_path
    assert parsed.front_matter == {}
    assert parsed.content == "Content of Note B, no front matter."


def test_parse_markdown_with_no_curate(mock_vault: Path):
    """Test parsing front-matter containing the no-curate flag."""
    note_c_path = mock_vault / "Subfolder" / "Note C.md"
    parsed = parse_markdown_file(note_c_path)
    assert isinstance(parsed, ParsedNote)
    assert parsed.file_path == note_c_path
    assert parsed.front_matter == {"aliases": ["C"], "no-curate": True}
    assert parsed.content == "Content of Note C."


def test_parse_empty_markdown(mock_vault: Path):
    """Test parsing an empty markdown file."""
    empty_note_path = mock_vault / "Empty Note.md"
    parsed = parse_markdown_file(empty_note_path)
    assert isinstance(parsed, ParsedNote)
    assert parsed.file_path == empty_note_path
    assert parsed.front_matter == {}
    assert parsed.content == ""


def test_parse_nonexistent_file(tmp_path: Path):
    """Test parsing a file that doesn't exist."""
    non_existent_file = tmp_path / "ghost.md"
    parsed = parse_markdown_file(non_existent_file)
    assert parsed is None

# --- Tests for find_and_parse_markdown_files ---


def test_find_and_parse_with_gitignore(mock_vault: Path):
    """Test the combined finding and parsing, respecting gitignore."""
    parsed_notes = list(find_and_parse_markdown_files(
        mock_vault, respect_gitignore=True))
    assert len(parsed_notes) == 5  # Should match find_markdown_files_with_gitignore

    parsed_paths = sorted([note.file_path for note in parsed_notes])
    expected_paths = sorted([
        mock_vault / "Note A.md",
        mock_vault / "Note B.md",
        mock_vault / "Subfolder" / "Note C.md",
        mock_vault / "Empty Note.md",
        mock_vault / "Subfolder" / ".hidden_file.md"
    ])
    assert parsed_paths == expected_paths


def test_find_and_parse_no_gitignore(mock_vault: Path):
    """Test combined finding and parsing without gitignore."""
    parsed_notes = list(find_and_parse_markdown_files(
        mock_vault, respect_gitignore=False))
    assert len(parsed_notes) == 7  # Should match find_markdown_files_basic
    parsed_paths = sorted([note.file_path for note in parsed_notes])
    expected_paths = sorted([
        mock_vault / "Note A.md",
        mock_vault / "Note B.md",
        mock_vault / "Subfolder" / "Note C.md",
        mock_vault / "Empty Note.md",
        mock_vault / "Subfolder" / ".hidden_file.md",
        mock_vault / "Ignored Folder" / "Ignored Note.md",
        mock_vault / "README.md"
    ])
    assert parsed_paths == expected_paths
