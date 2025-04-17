"""Functions for discovering and parsing Markdown files in an Obsidian vault."""

import logging
from pathlib import Path
from typing import List, Generator, Dict, Any, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

# Configure logger for this module
logger = logging.getLogger(__name__)


class ParsedNote(BaseModel):
    """Represents a parsed Markdown note with its metadata and content."""
    file_path: Path = Field(..., description="The absolute path to the Markdown file.")
    front_matter: Dict[str, Any] = Field(
        default_factory=dict, description="Parsed YAML front-matter as a dictionary.")
    content: str = Field(...,
                         description="The main Markdown content, excluding front-matter.")


def _extract_front_matter(content: str) -> Tuple[Optional[str], str]:
    """Splits YAML front-matter from the main content.

    Assumes front-matter is delimited by '---' at the start and end.

    Args:
        content: The full content of the file.

    Returns:
        A tuple containing the front-matter string (or None if not found)
        and the remaining content.
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            # parts[0] is empty string before first '---'
            # parts[1] is the front-matter
            # parts[2] is the rest of the content
            front_matter_str = parts[1].strip()
            main_content = parts[2].lstrip('\n ')  # Remove leading whitespace/newlines
            return front_matter_str, main_content
    # No valid front-matter found
    return None, content


def parse_markdown_file(file_path: Path) -> Optional[ParsedNote]:
    """Parses a single Markdown file, extracting front-matter and content.

    Args:
        file_path: The path to the Markdown file.

    Returns:
        A ParsedNote object containing the path, front-matter, and content,
        or None if the file cannot be read or parsed.
    """
    try:
        full_content = file_path.read_text(encoding="utf-8")
        front_matter_str, main_content = _extract_front_matter(full_content)

        front_matter_dict: Dict[str, Any] = {}
        if front_matter_str:
            try:
                # Use safe_load to prevent arbitrary code execution
                loaded_yaml = yaml.safe_load(front_matter_str)
                if isinstance(loaded_yaml, dict):
                    front_matter_dict = loaded_yaml
                else:
                    logger.warning(
                        f"Front-matter in {file_path} is not a dictionary. Ignoring. "
                        f"Content: {front_matter_str[:100]}..."
                    )
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML front-matter in {file_path}: {e}")
                # Decide if we should return None or proceed without front_matter
                # Proceeding without front_matter for now

        return ParsedNote(
            file_path=file_path,
            front_matter=front_matter_dict,
            content=main_content
        )

    except OSError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing file {file_path}: {e}")
        return None


def find_and_parse_markdown_files(vault_path: Path) -> Generator[ParsedNote, None, None]:
    """Finds and parses all Markdown files in the vault.

    Combines finding files and parsing them into a single generator.

    Args:
        vault_path: The absolute path to the Obsidian vault directory.

    Yields:
        ParsedNote objects for each successfully parsed Markdown file.
    """
    # Reuse the find_markdown_files logic for discovery
    for file_path in find_markdown_files(vault_path):
        parsed_note = parse_markdown_file(file_path)
        if parsed_note:
            yield parsed_note


def find_markdown_files(vault_path: Path) -> Generator[Path, None, None]:
    """Recursively finds all Markdown (.md) files in the given directory.

    Args:
        vault_path: The absolute path to the Obsidian vault directory.

    Yields:
        Path objects representing the found Markdown files.

    Raises:
        FileNotFoundError: If the vault_path does not exist or is not a directory.
    """
    if not vault_path.is_dir():
        msg = f"Vault path does not exist or is not a directory: {vault_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Starting Markdown file discovery in: {vault_path}")
    count = 0
    for file_path in vault_path.rglob("*.md"):
        if file_path.is_file():
            logger.debug(f"Found Markdown file: {file_path}")
            yield file_path
            count += 1
        else:
            logger.warning(
                f"Found item matching *.md pattern, but it's not a file: {file_path}")

    logger.info(f"Finished discovery. Found {count} Markdown files.")
