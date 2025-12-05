"""Provides the main API for interacting with the Bidian agent functionalities."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
from pydantic import BaseModel, Field
import diff_match_patch as dmp_module

# Assuming retriever handles finding notes
from bidian.core.retriever import Retriever, RelatedNoteResult
from bidian.core.backlinker import update_backlinks  # Assuming backlinker handles writing
from bidian.indexing.discovery import parse_markdown_file, ParsedNote  # To get content for query
from bidian.creation.templating import TemplateRenderer  # Import TemplateRenderer
from bidian.common.patching import (
    save_creation_patch,
    rollback_creation_patch,
    save_refactor_patch,
    rollback_refactor_patch,
    PatchError,
    PatchFormatError,
    PATCH_FILE_SUFFIX  # Needed for identification
)
# Import refactoring logic
from bidian.refactoring.core import refactor_note

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default location for new notes if not specified otherwise
DEFAULT_NEW_NOTE_FOLDER = "Bidian Hubs"

# --- Data Models ---


class RefactorProposal(BaseModel):
    """Data structure for a refactoring proposal."""
    target_file_path: Path
    original_content: str
    refactored_content: str
    # Diff format can be adapted, e.g., unified diff string or patch object list
    diff: List[Tuple[int, str]] = Field(
        ..., description="Diff represented as list of (diff_type, text) tuples from diff-match-patch.")
    # diff_type: -1 for delete, 0 for equal, 1 for insert

# --- API Class ---


class BidianAgentAPI:
    """Encapsulates the core agent operations like finding and adding backlinks."""

    def __init__(self,
                 retriever: Retriever,
                 template_renderer: TemplateRenderer,  # Added
                 vault_path: Path,
                 # Add other dependencies like indexer, config manager etc. as needed
                 ) -> None:
        """Initializes the Bidian Agent API.

        Args:
            retriever: An instance of the Retriever class.
            template_renderer: An instance of the TemplateRenderer class.
            vault_path: The absolute path to the Obsidian vault.
            # backlinker_log_path: Optional path for backlink log
            # backlinker_heading: Optional heading for backlink section
        """
        self.retriever = retriever
        self.renderer = template_renderer  # Store renderer
        self.vault_path = vault_path.resolve()  # Ensure absolute path
        # Store other config/dependencies if needed
        # self.log_path = backlinker_log_path
        # self.backlink_heading = backlinker_heading
        # Initialize diff object
        self.dmp = dmp_module.diff_match_patch()

    def curate_backlinks_for_file(self, target_file_path: Path) -> bool:
        """Finds related notes for a given file and updates its backlinks section.

        This acts as the implementation for the 'Curate Current Note' command (F-2).

        Args:
            target_file_path: The absolute path to the target Markdown file.

        Returns:
            True if backlinks were successfully updated (or no update was needed 
            but the process completed), False if an error occurred.
        """
        logger.info(f"Starting backlink curation for: {target_file_path}")

        target_file_path = target_file_path.resolve()
        if not target_file_path.is_file():
            logger.error(
                f"Target file does not exist or is not a file: {target_file_path}")
            return False

        # 1. Parse the target file to get its content for querying
        parsed_note = parse_markdown_file(target_file_path)
        if not parsed_note:
            logger.error(f"Failed to parse target file: {target_file_path}")
            return False

        # Use the main content (excluding front-matter) for finding related notes
        query_content = parsed_note.content
        related_notes: List[RelatedNoteResult] = []  # Initialize empty list
        if not query_content.strip():
            logger.warning(
                f"Target file has no content to query for related notes: {target_file_path}")
            # Proceed with empty related_notes to potentially remove existing backlinks
        else:
            # 2. Find related notes using the Retriever
            try:
                # Retrieve notes based on the content of the target file
                # Use default k and threshold from Retriever for now
                related_notes = self.retriever.find_related_notes(
                    query_text=query_content
                )
                logger.info(
                    f"Retriever found {len(related_notes)} potential related notes for {target_file_path.name}.")
            except Exception as e:
                logger.error(
                    f"Error finding related notes for {target_file_path}: {e}", exc_info=True)
                return False

        # 3. Update the backlinks in the target file
        try:
            # update_backlinks handles idempotency and logging internally
            # It returns True if the file was written, False otherwise (incl. no change needed)
            was_modified: bool = update_backlinks(
                target_file_path=target_file_path,
                related_notes=related_notes,
                vault_base_path=self.vault_path,
                # Pass log path / heading from config if stored in self
            )
            logger.info(
                f"Backlink update process completed for {target_file_path}. File modified: {was_modified}")
            # Return True indicating successful completion, even if no change occurred
            # because the process itself didn't fail.
            return True
        except Exception as e:
            logger.error(
                f"Error updating backlinks in {target_file_path}: {e}", exc_info=True)
            return False

    def propose_hub_note(
        self,
        input_note_paths: List[Path],
        target_folder: str = DEFAULT_NEW_NOTE_FOLDER,
        template_name: str = "hub_page"
    ) -> Tuple[Path, str]:
        """Generates proposed content and path for a new hub note.

        Based on a list of input notes, suggests title, outline, and backlinks,
        then renders the specified template.

        Args:
            input_note_paths: List of absolute paths to the notes forming the hub.
            target_folder: Relative path within the vault to save the new note.
            template_name: Name of the template to use (e.g., 'hub_page').

        Returns:
            A tuple containing: 
              - The proposed absolute Path for the new note.
              - The proposed note content (rendered template string).

        Raises:
            ValueError: If input_note_paths is empty.
            FileNotFoundError: If any input note path does not exist.
            TemplateNotFoundError: If the specified template is not found.
            TemplateRenderingError: If template rendering fails.
        """
        if not input_note_paths:
            raise ValueError("Cannot propose a hub note with no input notes.")

        logger.info(f"Proposing hub note based on {len(input_note_paths)} input notes.")
        input_note_titles: List[str] = []
        relative_links: List[str] = []  # Links relative to vault root

        # Validate paths and extract info
        for note_path in input_note_paths:
            abs_path = note_path.resolve()
            if not abs_path.is_file():
                raise FileNotFoundError(f"Input note not found: {abs_path}")
            # Use filename stem as title for now
            input_note_titles.append(abs_path.stem)
            try:
                relative_path = abs_path.relative_to(self.vault_path)
                link_target = str(relative_path.with_suffix(''))
                relative_links.append(link_target)
            except ValueError:
                logger.warning(
                    f"Could not create relative link for {abs_path}, using stem.")
                relative_links.append(abs_path.stem)

        # --- Generate Title, Outline, Backlinks (Simple Heuristics for now) ---
        # TODO: Replace with LLM call for better title/outline generation
        # Simple title based on first few input notes
        max_title_notes = 3
        title_prefix = "Hub for"
        title_notes_str = ", ".join(input_note_titles[:max_title_notes])
        if len(input_note_titles) > max_title_notes:
            title_notes_str += ", ..."
        proposed_title = f"{title_prefix} {title_notes_str}"

        # Use input note titles as the outline
        proposed_outline = input_note_titles

        # Use relative links generated above as backlinks
        proposed_backlinks = relative_links
        # --- End Heuristics ---

        # Prepare context for the template
        context = {
            "title": proposed_title,
            "outline": proposed_outline,
            "backlinks": proposed_backlinks,
            # Add other potential context vars here
            # 'front_matter': {} # Optionally override template FM here
        }

        # Render the template
        logger.debug(f"Rendering template '{template_name}' with context.")
        rendered_content = self.renderer.render_template(template_name, context)

        # Determine target path
        # Sanitize title for filename
        safe_filename = "".join(c if c.isalnum() or c in (
            ' ', '-', '_') else '' for c in proposed_title).strip()
        if not safe_filename:
            safe_filename = "Untitled Hub Note"
        target_filename = f"{safe_filename}.md"

        target_dir_path = self.vault_path / target_folder
        proposed_path = target_dir_path / target_filename
        logger.info(f"Proposed path for new hub note: {proposed_path}")

        return proposed_path, rendered_content

    def commit_note(
        self,
        file_path: Path,
        content: str,
        overwrite: bool = False
    ) -> bool:
        """Writes the given content to the specified file path.

        Also saves a patch file upon successful creation (not overwrite).

        Args:
            file_path: The absolute path where the note should be saved.
            content: The string content to write to the note.
            overwrite: If True, overwrite the file if it already exists.
                       If False, fail if the file exists.

        Returns:
            True if the file was written successfully, False otherwise.
        """
        abs_path = file_path.resolve()
        logger.info(f"Attempting to commit note to: {abs_path}")
        is_creation = not abs_path.exists()

        if not is_creation and not overwrite:
            logger.error(
                f"Cannot commit note. File already exists and overwrite is False: {abs_path}")
            return False

        try:
            # Ensure parent directory exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            logger.info(f"Successfully committed note: {abs_path}")

            # Save patch file only on initial creation
            if is_creation:
                patch_path = save_creation_patch(abs_path)
                if not patch_path:
                    logger.warning(
                        f"Note committed, but failed to save creation patch for {abs_path}")
                    # Decide on behavior: maybe return False or raise specific error?
                    # For now, log warning and return True as note was written.

            return True
        except OSError as e:
            logger.error(f"Error writing note to file {abs_path}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error committing note {abs_path}: {e}", exc_info=True)
            return False

    def propose_refactor(
        self,
        target_file_path: Path,
        update_fm: Optional[Dict[str, Any]] = None,
        add_toc: bool = False,
        toc_max_level: int = 3,
        run_heading_restructure: bool = False
    ) -> Optional[RefactorProposal]:
        """Analyzes a note, applies refactoring rules, and proposes changes with a diff.

        Args:
            target_file_path: Absolute path to the note to refactor.
            update_fm: Dictionary of front-matter key/values to update/add.
            add_toc: If True, generate and add a Table of Contents.
            toc_max_level: Max heading level for ToC.
            run_heading_restructure: If True, run heading restructuring (placeholder).

        Returns:
            A RefactorProposal object containing original content, refactored content,
            and a diff, or None if the file doesn't exist, cannot be parsed,
            or no changes were generated.
        """
        logger.info(f"Proposing refactor for: {target_file_path}")
        abs_path = target_file_path.resolve()

        if not abs_path.is_file():
            logger.error(f"Target file for refactor not found: {abs_path}")
            return None

        # 1. Parse the note
        parsed_note = parse_markdown_file(abs_path)
        if not parsed_note:
            logger.error(f"Failed to parse target file for refactor: {abs_path}")
            return None

        # Need original full content (including FM) for diff later
        try:
            original_full_content = abs_path.read_text(encoding='utf-8')
        except OSError as e:
            logger.error(
                f"Failed to read target file content for refactor: {abs_path}: {e}")
            return None

        # 2. Apply refactoring logic
        try:
            refactored_content = refactor_note(
                note=parsed_note,
                update_fm=update_fm,
                add_toc=add_toc,
                toc_max_level=toc_max_level,
                run_heading_restructure=run_heading_restructure
            )
        except Exception as e:
            logger.error(
                f"Error during refactoring logic for {abs_path}: {e}", exc_info=True)
            return None

        # 3. Check if changes were made
        if refactored_content is None:
            logger.info(f"No refactoring changes proposed for {abs_path}.")
            return None  # No changes

        # 4. Generate diff
        # Use diff-match-patch library
        diff_list = self.dmp.diff_main(original_full_content, refactored_content)
        # Optional: clean up diff for efficiency
        self.dmp.diff_cleanupSemantic(diff_list)
        # Optional: convert to a more readable format if needed (e.g., unified diff)
        # diff_text = self.dmp.diff_prettyHtml(diff_list) # Example HTML output
        # For now, return the raw diff list
        logger.info(f"Generated diff for proposed refactor of {abs_path}.")

        # 5. Create and return proposal
        proposal = RefactorProposal(
            target_file_path=abs_path,
            original_content=original_full_content,
            refactored_content=refactored_content,
            diff=diff_list
        )
        return proposal

    def commit_refactor(
        self,
        proposal: RefactorProposal
    ) -> bool:
        """Commits the refactored note content to the file and saves a patch.

        Args:
            proposal: The RefactorProposal object containing the changes.

        Returns:
            True if commit was successful, False otherwise.
        """
        abs_path = proposal.target_file_path.resolve()
        logger.info(f"Attempting to commit refactor to: {abs_path}")

        try:
            # Write the refactored content (effectively overwriting)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(proposal.refactored_content, encoding="utf-8")
            logger.info(f"Successfully committed refactored note: {abs_path}")

            # Save the patch file containing the diff AND original content
            patch_path = save_refactor_patch(
                target_file_path=abs_path,
                original_content=proposal.original_content,  # Pass original content
                diff=proposal.diff
            )
            if not patch_path:
                logger.warning(
                    f"Refactor committed, but failed to save patch file for {abs_path}")
                # Decide behavior - for now return True as note was written

            return True
        except OSError as e:
            logger.error(
                f"Error writing refactored note to file {abs_path}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error committing refactor {abs_path}: {e}", exc_info=True)
            return False

    def rollback_patch(self, patch_file_path: Path) -> bool:
        """Rolls back the change described in the given patch file.

        Args:
            patch_file_path: The absolute path to the .bidian-patch file.

        Returns:
            True if the rollback was successful, False otherwise.
        """
        abs_patch_path = patch_file_path.resolve()
        logger.info(f"Attempting to roll back patch: {abs_patch_path}")

        if not abs_patch_path.is_file() or not abs_patch_path.name.endswith(PATCH_FILE_SUFFIX):
            logger.error(f"Invalid patch file path: {abs_patch_path}")
            return False

        try:
            # Peek at the action type without fully parsing complex diffs yet
            action = None
            try:
                with open(abs_patch_path, 'r', encoding='utf-8') as f:
                    patch_data = json.load(f)
                    action = patch_data.get("action")
            except json.JSONDecodeError as json_err:
                logger.warning(
                    f"Could not parse patch file as JSON: {abs_patch_path}. Error: {json_err}")
                # Might be a plain diff file later? For now, assume JSON.
                raise PatchFormatError("Patch file is not valid JSON.") from json_err
            except OSError as os_err:
                raise PatchError(
                    f"Could not read patch file {abs_patch_path}: {os_err}") from os_err

            if action == "create":
                rollback_creation_patch(abs_patch_path)
                logger.info(
                    f"Successfully rolled back 'create' action from patch: {abs_patch_path}")
                return True
            elif action == "refactor":
                # Call the rollback function for refactor patches
                rollback_refactor_patch(abs_patch_path)
                logger.info(
                    f"Successfully rolled back 'refactor' action from patch: {abs_patch_path}")
                return True
            else:
                logger.error(
                    f"Unsupported or unknown action type '{action}' in patch file: {abs_patch_path}")
                raise PatchFormatError(f"Unknown action type: {action}")

        except PatchError as e:  # Catch specific patch errors from rollback functions
            logger.error(
                f"Patch rollback failed for {abs_patch_path}: {e}", exc_info=True)
            return False
        except FileNotFoundError:
            # This might happen if the patch file or target file was deleted manually
            logger.error(
                f"File not found during rollback: {abs_patch_path}", exc_info=True)
            return False
        except NotImplementedError as ni_err:
            logger.error(
                f"Rollback for action type failed (Not Implemented): {ni_err}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during patch rollback for {abs_patch_path}: {e}", exc_info=True)
            return False

# Example Usage (requires setting up retriever, vault etc.)
# if __name__ == '__main__':
#     from bidian.config.logging_config import setup_logging
#     from bidian.indexing.embedding import EmbeddingGenerator
#     from bidian.indexing.vector_store import ChromaVectorStore
#     from bidian.indexing.indexer import Indexer
#     import tempfile
#     import time
#
#     setup_logging("INFO")
#
#     # --- Setup similar to backlinker/retriever examples ---
#     with tempfile.TemporaryDirectory() as tmpdir:
#         vault_dir = Path(tmpdir) / "test_vault"
#         vault_dir.mkdir()
#         data_dir = Path(tmpdir) / "test_data"
#         data_dir.mkdir()
#         models_dir = Path(tmpdir) / "test_models"
#         models_dir.mkdir()
#
#         dummy_model_path = models_dir / "dummy_model.gguf"
#         dummy_model_path.touch()
#
#         note_a_path = vault_dir / "Note A.md"
#         note_b_path = vault_dir / "Note B.md"
#         note_c_path = vault_dir / "Note C.md"
#
#         note_a_path.write_text("# Note A\n\nContent about apples.")
#         note_b_path.write_text("# Note B\n\nContent about oranges and apples.")
#         note_c_path.write_text("# Note C\n\nContent about bananas.")
#
#         # --- Mocking ---
#         class MockEmbeddingGenerator:
#             def __init__(self): self.dim = 4
#             def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#                 # Simple mock: find keyword, return fixed vec
#                 if "apples" in texts[0].lower(): return [[0.1, 0.2, 0.3, 0.4]]
#                 if "oranges" in texts[0].lower(): return [[0.5, 0.4, 0.3, 0.2]]
#                 return [[0.9] * self.dim] # Default
#             def get_embedding_dim(self) -> Optional[int]: return self.dim
#
#         mock_embedder = MockEmbeddingGenerator()
#         chroma_path = str(data_dir / "chroma_db")
#         state_file = str(data_dir / "index_state.json")
#         vector_store = ChromaVectorStore(embedding_dimension=mock_embedder.get_embedding_dim(), persist_path=chroma_path)
#         indexer = Indexer(vault_dir, mock_embedder, vector_store, state_file)
#         print("\n--- Running indexer --- ")
#         indexer.run_incremental_update()
#
#         retriever = Retriever(mock_embedder, vector_store)
#
#         # --- Initialize API ---
#         agent_api = BidianAgentAPI(retriever=retriever, vault_path=vault_dir)
#
#         # --- Test curation ---
#         target_file = note_a_path
#         print(f"\n--- Curating backlinks for: {target_file.name} ---")
#         success = agent_api.curate_backlinks_for_file(target_file)
#         print(f"Curation successful: {success}")
#         print(f"Content of {target_file.name} after curation:\n{target_file.read_text()}")
#
#         target_file = note_b_path
#         print(f"\n--- Curating backlinks for: {target_file.name} ---")
#         success = agent_api.curate_backlinks_for_file(target_file)
#         print(f"Curation successful: {success}")
#         print(f"Content of {target_file.name} after curation:\n{target_file.read_text()}")
#
#         # Check log file
#         log_file = Path(DEFAULT_LOG_PATH).resolve()
#         if not log_file.parent.is_relative_to(Path(tmpdir)):
#             log_file = data_dir / log_file.name # Adjust if default path is outside tmpdir
#         print(f"\n--- Log file content ({log_file.name}) ---")
#         if log_file.exists():
#             print(log_file.read_text()) # noqa: T201
#         else:
#             print("(Log file not created)")
