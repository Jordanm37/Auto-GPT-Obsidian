"""Core logic for analyzing and refactoring Markdown notes."""

import logging
from typing import List, Dict, Any, Tuple, Optional
import re  # For slugify

import yaml  # To dump front-matter
from markdown_it import MarkdownIt
from markdown_it.token import Token
from unidecode import unidecode  # For slugify

from bidian.indexing.discovery import ParsedNote

logger = logging.getLogger(__name__)

# Initialize Markdown parser
# Using the 'commonmark' spec is a safe default
md_parser = MarkdownIt("commonmark")


class HeadingInfo(Dict):
    """Simple structure to hold heading information."""
    level: int
    content: str
    # Future: add line numbers, raw token info?


def analyze_markdown_structure(content: str) -> Dict[str, Any]:
    """Parses Markdown content and extracts structural information.

    Currently extracts headings.

    Args:
        content: The Markdown content string (excluding front-matter).

    Returns:
        A dictionary containing extracted structural elements, e.g.,
        {'headings': [{'level': 1, 'content': 'Title'}, ...]}.
    """
    logger.debug("Analyzing Markdown structure...")
    structure: Dict[str, Any] = {
        "headings": [],
        # Future: paragraphs, lists, code_blocks etc.
    }

    try:
        tokens: List[Token] = md_parser.parse(content)
    except Exception as e:
        logger.error(f"Markdown parsing failed: {e}", exc_info=True)
        # Return empty structure on parsing failure
        return structure

    # Iterate through tokens to find headings
    # Heading levels are stored in token.markup (e.g., '#', '##')
    # Heading content is within inline tokens nested inside heading_open/heading_close
    current_heading_level: Optional[int] = None
    current_heading_content: str = ""

    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            current_heading_level = len(token.markup)  # '#'=1, '##'=2 etc.
            current_heading_content = ""
            # Content is typically in the next inline token
            if i + 1 < len(tokens) and tokens[i+1].type == "inline":
                # Concatenate content from children of the inline token
                current_heading_content = "".join(
                    t.content for t in tokens[i+1].children if t.content)

        elif token.type == "heading_close":
            if current_heading_level is not None:
                heading_info = HeadingInfo(
                    level=current_heading_level, content=current_heading_content.strip())
                structure["headings"].append(heading_info)
                logger.debug(
                    f"Found heading: Level {heading_info['level']} - '{heading_info['content']}'")
            # Reset for next heading
            current_heading_level = None
            current_heading_content = ""
        # We could extract other elements here (paragraphs, lists, etc.)

    logger.debug(
        f"Structure analysis complete. Found {len(structure['headings'])} headings.")
    return structure


def _slugify(text: str) -> str:
    """Generates a Markdown-friendly anchor slug from heading text."""
    # Basic slugify: lowercase, remove non-alphanumeric, replace space with hyphen
    text = unidecode(text)  # Handle non-ASCII chars
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text).strip('-')
    return text


def generate_toc(structure: Dict[str, Any], max_level: int = 3) -> str:
    """Generates a Markdown Table of Contents from the heading structure.

    Args:
        structure: The analysis result containing a list of headings.
        max_level: The maximum heading level to include in the ToC.

    Returns:
        A Markdown string representing the ToC, or empty string if no headings.
    """
    toc_lines: List[str] = []
    headings = structure.get("headings", [])
    if not headings:
        return ""

    logger.debug(
        f"Generating ToC up to level {max_level} from {len(headings)} headings.")
    # Using a list to track generated slugs to handle duplicates
    generated_slugs: Dict[str, int] = {}

    for heading in headings:
        level = heading["level"]
        content = heading["content"]
        if level > max_level or not content:
            continue

        # Generate slug, handling potential duplicates
        base_slug = _slugify(content)
        slug = base_slug
        count = generated_slugs.get(base_slug, 0)
        if count > 0:
            slug = f"{base_slug}-{count}"
        generated_slugs[base_slug] = count + 1

        # Indentation based on level (adjust base level if needed, assuming H1 is level 1)
        indent = "  " * (level - 1)
        toc_lines.append(f"{indent}- [{content}](#{slug})")

    return "\n".join(toc_lines)

# --- Refactoring Functions (Placeholders for now) ---


def restructure_headings(original_content: str, structure: Dict[str, Any]) -> str:
    """Applies rules to restructure headings.

    Current Rule:
    - Ensures at most one H1 heading. Demotes subsequent H1s to H2s.
    """
    logger.debug("Restructuring headings...")
    # Find all H1 headings from the structure analysis
    h1_indices = [i for i, h in enumerate(
        structure.get("headings", [])) if h["level"] == 1]

    if len(h1_indices) <= 1:
        logger.debug("No heading restructuring needed (0 or 1 H1 found).")
        return original_content  # No change needed

    logger.info(f"Found {len(h1_indices)} H1 headings. Demoting subsequent H1s to H2.")

    # We need to modify the original markdown text.
    # Using tokens is complex for modification. Let's try regex substitution first,
    # acknowledging its potential fragility with complex markdown.

    # Find all lines starting with '# ' (H1)
    lines = original_content.split('\n')
    modified_lines = []
    h1_count = 0
    for line in lines:
        # Basic check for H1 marker at the start of the line
        if line.strip().startswith('# '):
            h1_count += 1
            if h1_count > 1:
                # Demote to H2 by adding another #
                modified_line = "#" + line
                modified_lines.append(modified_line)
                logger.debug(
                    f"Demoted H1 to H2: '{line.strip()}' -> '{modified_line.strip()}'")
            else:
                modified_lines.append(line)  # Keep first H1 as is
        else:
            modified_lines.append(line)

    restructured_content = "\n".join(modified_lines)

    if restructured_content != original_content:
        logger.info("Successfully restructured headings (demoted extra H1s).")
        return restructured_content
    else:
        # Should not happen if h1_indices > 1, but as a safeguard
        logger.debug("Heading restructuring resulted in no changes.")
        return original_content


def update_front_matter(original_fm: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Updates the front-matter dictionary with new/modified values.
    """
    logger.debug(f"Updating front matter. Original: {original_fm}, Updates: {updates}")
    updated_fm = original_fm.copy()
    updated_fm.update(updates)
    # TODO: Add logic for appending to lists (like tags) vs overwriting?
    # For now, simple dictionary update.
    logger.debug(f"Updated front matter: {updated_fm}")
    return updated_fm


def generate_toc(structure: Dict[str, Any], max_level: int = 3) -> str:
    """Generates a Markdown Table of Contents from the heading structure.
       (Placeholder - returns empty string for now)
    """
    logger.warning("generate_toc is not implemented yet.")
    # TODO: Implement ToC generation logic
    # Needs to create hierarchical list based on heading levels and content
    # Handle links (#heading-slug) - needs slugification
    return ""  # Placeholder

# --- Orchestration Function ---


def refactor_note(
    note: ParsedNote,
    update_fm: Optional[Dict[str, Any]] = None,
    add_toc: bool = False,
    toc_max_level: int = 3,
    run_heading_restructure: bool = False
) -> Optional[str]:
    """Applies selected refactoring operations to a parsed note.

    Args:
        note: The ParsedNote object to refactor.
        update_fm: Dictionary of front-matter keys/values to update/add.
        add_toc: If True, generate and insert/update a Table of Contents.
        toc_max_level: Maximum heading level for the ToC.
        run_heading_restructure: If True, run the heading restructuring logic.

    Returns:
        The refactored note content as a string (including front-matter),
        or None if no changes were made or an error occurred.
    """
    logger.info(f"Starting refactoring for: {note.file_path}")
    original_content = note.content
    original_fm = note.front_matter
    changes_made = False

    # 1. Analyze structure
    structure = analyze_markdown_structure(original_content)

    # 2. Update Front Matter
    refactored_fm = original_fm
    if update_fm:
        updated = update_front_matter(original_fm, update_fm)
        if updated != original_fm:
            refactored_fm = updated
            changes_made = True
            logger.info(f"Front matter updated for {note.file_path}")
        else:
            logger.debug(f"No front matter changes needed for {note.file_path}")

    # 3. Restructure Headings (Now implemented)
    refactored_content = original_content
    if run_heading_restructure:
        restructured = restructure_headings(original_content, structure)
        if restructured != original_content:
            refactored_content = restructured
            changes_made = True
            logger.info(f"Headings restructured for {note.file_path}")
            # Re-analyze structure if headings changed, needed for ToC
            structure = analyze_markdown_structure(refactored_content)
        else:
            logger.debug(
                f"No heading restructuring changes needed for {note.file_path}")

    # 4. Generate and Insert/Update ToC
    if add_toc:
        toc_markdown = generate_toc(structure, toc_max_level)
        # Basic ToC insertion: Place it after front-matter (if any), before first heading or content.
        # TODO: Make ToC insertion more robust (e.g., idempotent block like backlinks)
        if toc_markdown:
            # Simple insertion logic for now:
            if refactored_content.lstrip().startswith("#") or refactored_content.strip():  # Content exists
                # Insert after potential initial whitespace, before first real content
                lines = refactored_content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip():  # Find first non-empty line
                        insert_pos = i
                        break

                toc_section = f"## Table of Contents\n\n{toc_markdown}\n\n"
                # Avoid inserting ToC if it seems to exist already (very basic check)
                # Check near top
                if "## Table of Contents" not in "\n".join(lines[:insert_pos+2]):
                    lines.insert(insert_pos, toc_section)
                    new_content = "\n".join(lines)
                    if new_content != refactored_content:
                        refactored_content = new_content
                        changes_made = True
                        logger.info(f"Added Table of Contents to {note.file_path}")
                else:
                    logger.debug(
                        f"Table of Contents section likely already exists in {note.file_path}, skipping add.")

            else:  # Insert into empty content section
                refactored_content = f"## Table of Contents\n\n{toc_markdown}\n"
                changes_made = True
                logger.info(f"Added Table of Contents to {note.file_path}")
        else:
            logger.debug(
                f"No ToC generated (no valid headings found) for {note.file_path}")

    # 5. Reconstruct the note only if changes were made
    if not changes_made:
        logger.info(f"No refactoring changes applied to {note.file_path}")
        return None

    final_parts = []
    # Add front matter if it exists
    if refactored_fm:
        try:
            fm_yaml = yaml.safe_dump(
                refactored_fm, default_flow_style=False, sort_keys=False)
            final_parts.append("---")
            final_parts.append(fm_yaml.strip())
            final_parts.append("---")
        except YAMLError as e:
            logger.error(
                f"Error formatting final front matter for {note.file_path}: {e}")
            # Return original content maybe? Or raise? Returning None for now.
            return None

    # Add refactored content
    final_parts.append(refactored_content.strip())

    # Join parts, ensure separation between FM and content if both exist
    full_note = "\n".join(final_parts)
    # Ensure a single trailing newline
    final_output = full_note.strip() + '\n'
    logger.info(f"Refactoring complete for: {note.file_path}")
    return final_output

# --- Orchestration ---

# TODO: Implement a main refactor_note function or class method
# that takes a ParsedNote, calls analysis, applies refactoring steps,
# reconstructs the note content (including updated front-matter),
# and returns the result.

# Example Usage (basic analysis)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#     test_md = """
# # Main Title
#
# Some intro text.
#
# ## Section 1
#
# Content here.
#
# ### Subsection 1.1
#
# More details.
#
# ## Section 2
#
# Final content.
# """
#
#     print("--- Analyzing Markdown --- ")
#     structure = analyze_markdown_structure(test_md)
#     import json
#     print(json.dumps(structure, indent=2))
#
#     print("\n--- Testing Front Matter Update ---")
#     fm = {"tags": ["old"], "status": "draft"}
#     updates = {"status": "published", "tags": ["new"], "author": "Bidian"}
#     updated = update_front_matter(fm, updates)
#     print(f"Original: {fm}")
#     print(f"Updates: {updates}")
#     print(f"Result: {updated}")
