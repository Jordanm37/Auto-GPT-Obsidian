"""Module for inserting and managing backlinks in Markdown files."""

import logging
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from bidian.core.retriever import RelatedNoteResult

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Configuration ---
# HTML comments used to delimit the auto-generated backlink section
BACKLINK_START_COMMENT = "<!-- bidian-backlinks:start -->"
BACKLINK_END_COMMENT = "<!-- bidian-backlinks:end -->"

# Path for the JSONL log file (Requirement F-7)
DEFAULT_LOG_PATH = os.environ.get(
    "BIDIAN_BACKLINK_LOG_PATH",
    "./data/backlink_log.jsonl"
)
DEFAULT_BACKLINK_HEADING = "## Related Notes"


def _format_backlinks(related_notes: List[RelatedNoteResult],
                      vault_base_path: Path,
                      target_file_path: Path) -> str:
    """Formats the list of related notes into a Markdown list of links.

    Args:
        related_notes: List of related notes found by the retriever.
        vault_base_path: The base path of the vault for generating relative links.
        target_file_path: The path of the file where backlinks are being inserted 
                          (to avoid self-links).

    Returns:
        A Markdown string representing the list of backlinks, or an empty string.
    """
    link_list = []
    # Ensure vault_base_path is absolute for correct relative path calculation
    vault_base_path = vault_base_path.resolve()
    target_file_path = target_file_path.resolve()

    for note in related_notes:
        # Avoid linking a note to itself
        if note.file_path.resolve() == target_file_path:
            logger.debug(f"Skipping self-link for {target_file_path.name}")
            continue

        try:
            # Generate relative path for Obsidian link if possible
            # Obsidian usually prefers links without the .md extension
            relative_path = note.file_path.relative_to(vault_base_path)
            link_target = str(relative_path.with_suffix(''))
            # Use filename as link text by default
            link_text = note.file_path.stem
            link = f"- [[{link_target}|{link_text}]]"  # Use alias format for clarity
            link_list.append(link)
        except ValueError:
            # If files are not in the same hierarchy (e.g., outside vault?)
            # fall back to absolute path or just filename
            logger.warning(
                f"Could not create relative path for {note.file_path} from {vault_base_path}. Using filename.")
            link_text = note.file_path.stem
            link = f"- [[{link_text}]]"
            link_list.append(link)

    if not link_list:
        return ""  # Return empty string if no valid links

    return "\n".join(link_list)


def _insert_backlinks_idempotent(target_file_path: Path,
                                 backlinks_md: str,
                                 heading: str = DEFAULT_BACKLINK_HEADING) -> Optional[str]:
    """Reads the target file and inserts/updates the backlink block.

    Uses HTML comments as guards for idempotency.

    Args:
        target_file_path: Path to the Markdown file to modify.
        backlinks_md: The formatted Markdown string of backlinks to insert.
        heading: The heading to place above the backlinks block.

    Returns:
        The updated file content as a string, or None if the file couldn't be read 
        or if no changes were needed.
    """
    try:
        original_content = target_file_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.error(f"Error reading target file {target_file_path}: {e}")
        return None

    new_block = f"\n\n{heading}\n{BACKLINK_START_COMMENT}\n{backlinks_md}\n{BACKLINK_END_COMMENT}\n"
    if not backlinks_md.strip():
        # If there are no backlinks to add, ensure any existing block is removed.
        new_block = ""

    # Regex to find the existing block, including the heading if present
    # Optional heading, start comment, anything inside, end comment.
    # Uses DOTALL to match across newlines.
    pattern = re.compile(
        rf"(\n*\n?{re.escape(heading)}\n)?{re.escape(BACKLINK_START_COMMENT)}.*?{re.escape(BACKLINK_END_COMMENT)}\n?",
        re.DOTALL
    )

    match = pattern.search(original_content)

    if match:
        existing_block = match.group(0)
        # Check if the content (excluding comments and heading) is the same
        # Extract content between comments from existing block
        existing_backlinks_match = re.search(
            rf"{re.escape(BACKLINK_START_COMMENT)}\n?(.*?)\n?{re.escape(BACKLINK_END_COMMENT)}",
            existing_block,
            re.DOTALL
        )
        existing_backlinks_content = existing_backlinks_match.group(
            1).strip() if existing_backlinks_match else ""

        # Compare generated backlinks with existing content
        if backlinks_md.strip() == existing_backlinks_content:
            logger.info(f"Backlinks in {target_file_path} are already up-to-date.")
            return None  # No changes needed
        else:
            logger.info(f"Updating existing backlink block in {target_file_path}")
            # Replace the entire matched block (including optional heading)
            updated_content = pattern.sub(new_block.strip(), original_content)
            # Ensure proper spacing if the block was at the very end
            if not updated_content.endswith('\n'):
                updated_content += '\n'
            elif not updated_content.endswith('\n\n') and new_block:
                # Add a newline if replacing with non-empty block and only one newline exists
                updated_content += '\n'
            return updated_content
    elif backlinks_md.strip():
        # No existing block found, and we have backlinks to add.
        # Append the new block to the end of the file.
        logger.info(f"Adding new backlink block to {target_file_path}")
        # Ensure there are two newlines before the appended block for separation
        sep = '\n\n' if not original_content.endswith('\n\n') else '\n'
        if not original_content.strip():  # Handle empty file
            sep = ''
        # Handle file without trailing newline
        elif not original_content.endswith('\n'):
            sep = '\n\n'

        updated_content = original_content.rstrip() + sep + new_block.strip() + '\n'
        return updated_content
    else:
        # No existing block and no new backlinks to add.
        logger.info(
            f"No backlinks to add and no existing block found in {target_file_path}.")
        return None  # No changes needed


def _log_backlink_change(target_file: Path,
                         added_links: List[str],
                         # TODO: Implement detection of removed links
                         removed_links: List[str],
                         log_file_path: Path) -> None:
    """Appends a record of the backlink change to a JSONL file."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_file": str(target_file),
        "added_links": added_links,
        "removed_links": removed_links,  # Placeholder for now
    }
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f)
            f.write('\n')
        logger.info(f"Logged backlink change for {target_file} to {log_file_path}")
    except OSError as e:
        logger.error(f"Failed to write to backlink log file {log_file_path}: {e}")


def update_backlinks(
    target_file_path: Path,
    related_notes: List[RelatedNoteResult],
    vault_base_path: Path,
    log_file_path_str: str = DEFAULT_LOG_PATH,
    heading: str = DEFAULT_BACKLINK_HEADING
) -> bool:
    """Updates the backlinks section in the target file based on related notes.

    This function is idempotent due to the use of guard comments.
    It also logs the changes made.

    Args:
        target_file_path: Path to the Markdown file to update.
        related_notes: List of related notes to link to.
        vault_base_path: Base path of the Obsidian vault.
        log_file_path_str: Path to the JSONL log file.
        heading: The heading for the backlinks section.

    Returns:
        True if the file was modified, False otherwise.
    """
    logger.debug(f"Attempting to update backlinks in: {target_file_path}")
    log_file_path = Path(log_file_path_str)

    # 1. Format the new list of backlinks
    backlinks_md = _format_backlinks(related_notes, vault_base_path, target_file_path)
    formatted_links = backlinks_md.split('\n') if backlinks_md else []

    # 2. Read the file and insert/update the block
    updated_content = _insert_backlinks_idempotent(
        target_file_path, backlinks_md, heading)

    # 3. If content changed, write back to file and log
    if updated_content is not None:
        try:
            target_file_path.write_text(updated_content, encoding="utf-8")
            logger.info(f"Successfully updated backlinks in {target_file_path}")

            # TODO: Enhance logging to detect removed links by comparing old/new blocks
            _log_backlink_change(
                target_file=target_file_path,
                added_links=formatted_links,
                removed_links=[],  # Placeholder
                log_file_path=log_file_path
            )
            return True
        except OSError as e:
            logger.error(f"Error writing updated backlinks to {target_file_path}: {e}")
            return False
    else:
        # No changes were made
        return False

# Example Usage (requires retriever setup)
# if __name__ == '__main__':
#     from bidian.config.logging_config import setup_logging
#     import tempfile
#
#     setup_logging("INFO")
#
#     with tempfile.TemporaryDirectory() as tmpdir:
#         vault_path = Path(tmpdir) / "test_vault"
#         vault_path.mkdir()
#         log_path = Path(tmpdir) / "test_log.jsonl"
#
#         file1 = vault_path / "Note A.md"
#         file2 = vault_path / "Note B.md"
#         file3 = vault_path / "Note C.md"
#         file_with_existing = vault_path / "Note D.md"
#         empty_file = vault_path / "Note E.md"
#
#         file1.write_text("# Note A Content\n\nSome text.")
#         file2.write_text("# Note B Content")
#         file3.write_text("# Note C Content")
#         file_with_existing.write_text(f"# Note D\n\nExisting text.\n\n{DEFAULT_BACKLINK_HEADING}\n{BACKLINK_START_COMMENT}\n- [[Note X]]\n{BACKLINK_END_COMMENT}\n\nMore text.")
#         empty_file.touch()
#
#         # Dummy related notes for Note A
#         related_for_a = [
#             RelatedNoteResult(file_path=file2, score=0.1, best_chunk_index=0, best_chunk_content="..."),
#             RelatedNoteResult(file_path=file3, score=0.2, best_chunk_index=0, best_chunk_content="..."),
#             RelatedNoteResult(file_path=file1, score=0.01, best_chunk_index=0, best_chunk_content="..."), # Self-link
#         ]
#         # Dummy related notes for Note D (to test update)
#         related_for_d = [
#             RelatedNoteResult(file_path=file1, score=0.3, best_chunk_index=0, best_chunk_content="..."),
#             RelatedNoteResult(file_path=file2, score=0.4, best_chunk_index=0, best_chunk_content="..."),
#         ]
#
#         print("--- Updating backlinks for Note A (new) ---")
#         update_backlinks(file1, related_for_a, vault_path, str(log_path))
#         print(f"Content of {file1.name}:\n{file1.read_text()[:300]}...")
#
#         print("\n--- Updating backlinks for Note D (update) ---")
#         update_backlinks(file_with_existing, related_for_d, vault_path, str(log_path))
#         print(f"Content of {file_with_existing.name}:\n{file_with_existing.read_text()[:300]}...")
#
#         print("\n--- Updating backlinks for Note B (no related notes) ---")
#         update_backlinks(file2, [], vault_path, str(log_path))
#         print(f"Content of {file2.name}:\n{file2.read_text()[:300]}...")
#
#         print("\n--- Updating backlinks for Note D again (no change expected) ---")
#         changed = update_backlinks(file_with_existing, related_for_d, vault_path, str(log_path))
#         print(f"Changed: {changed}")
#
#         print("\n--- Updating backlinks for Note D (empty list, remove existing) ---")
#         update_backlinks(file_with_existing, [], vault_path, str(log_path))
#         print(f"Content of {file_with_existing.name}:\n{file_with_existing.read_text()[:300]}...")
#
#         print("\n--- Updating backlinks for Empty Note E ---")
#         update_backlinks(empty_file, related_for_a, vault_path, str(log_path))
#         print(f"Content of {empty_file.name}:\n{empty_file.read_text()[:300]}...")
#
#         print(f"\n--- Log file content ({log_path.name}) ---")
#         if log_path.exists():
#             print(log_path.read_text()) # noqa: T201
#         else:
#             print("(Log file not created)")
