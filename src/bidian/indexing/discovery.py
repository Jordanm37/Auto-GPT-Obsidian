"""Functions for discovering and parsing Markdown files in an Obsidian vault."""

import logging
from pathlib import Path
from typing import List, Generator, Dict, Any, Optional, Tuple, Callable

import yaml
from pydantic import BaseModel, Field
import gitignore_parser

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


def find_markdown_files(
    vault_path: Path,
    respect_gitignore: bool = True
) -> Generator[Path, None, None]:
    """Recursively finds all Markdown (.md) files in the given directory.

    Args:
        vault_path: The absolute path to the Obsidian vault directory.
        respect_gitignore: If True, loads .gitignore from the vault path and ignores
                           matching files/directories.

    Yields:
        Path objects representing the found Markdown files.

    Raises:
        FileNotFoundError: If the vault_path does not exist or is not a directory.
    """
    if not vault_path.is_dir():
        msg = f"Vault path does not exist or is not a directory: {vault_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    ignore_matcher: Optional[Callable[[Path], bool]] = None
    if respect_gitignore:
        gitignore_path = vault_path / ".gitignore"
        if gitignore_path.is_file():
            try:
                with open(gitignore_path, 'r') as f:
                    ignore_matcher = gitignore_parser.parse(f)
                logger.info(f"Loaded .gitignore rules from {gitignore_path}")
            except OSError as e:
                logger.warning(
                    f"Could not read .gitignore file at {gitignore_path}: {e}")
        else:
            logger.debug("No .gitignore file found in vault path.")

    logger.info(f"Starting Markdown file discovery in: {vault_path}")
    count = 0
    # Use rglob to find all potential files efficiently
    for item_path in vault_path.rglob("*"):  # Find all items first
        # Check if ignored before checking type or suffix
        if ignore_matcher and ignore_matcher(item_path):
            # logger.debug(f"Ignoring path due to .gitignore: {item_path}")
            continue

        # Now check if it's a markdown file
        if item_path.is_file() and item_path.suffix.lower() == ".md":
            logger.debug(f"Found Markdown file: {item_path}")
            yield item_path
            count += 1
        # We don't need to log non-markdown files unless debugging specific ignore rules
        # else:
        #     if item_path.is_file():
        #         logger.debug(f"Ignoring non-markdown file: {item_path}")
        #     else:
        #          logger.debug(f"Ignoring directory: {item_path}")

    logger.info(
        f"Finished discovery. Found {count} Markdown files (respect_gitignore={respect_gitignore}).")


def find_and_parse_markdown_files(
    vault_path: Path,
    respect_gitignore: bool = True
) -> Generator[ParsedNote, None, None]:
    """Finds and parses all Markdown files in the vault.

    Combines finding files (respecting .gitignore) and parsing them.

    Args:
        vault_path: The absolute path to the Obsidian vault directory.
        respect_gitignore: Passed to find_markdown_files.

    Yields:
        ParsedNote objects for each successfully parsed Markdown file.
    """
    # Reuse the find_markdown_files logic for discovery
    for file_path in find_markdown_files(vault_path, respect_gitignore=respect_gitignore):
        # Note: Front-matter exclusion happens later in the Indexer
        parsed_note = parse_markdown_file(file_path)
        if parsed_note:
            yield parsed_note
