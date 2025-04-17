"""Orchestrates the indexing process for the Obsidian vault."""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Set, Optional

from bidian.indexing.discovery import find_markdown_files, parse_markdown_file
from bidian.indexing.chunking import chunk_by_paragraph, TextChunk
from bidian.indexing.embedding import EmbeddingGenerator
from bidian.indexing.vector_store import ChromaVectorStore

# Configure logger for this module
logger = logging.getLogger(__name__)

DEFAULT_INDEX_STATE_PATH = "./data/index_state.json"


class Indexer:
    """Manages the discovery, chunking, embedding, and storage of notes.

    Handles incremental updates based on file modification times.
    """

    def __init__(self,
                 vault_path: Path,
                 embedding_generator: EmbeddingGenerator,
                 vector_store: ChromaVectorStore,
                 state_path: str = DEFAULT_INDEX_STATE_PATH):
        """Initializes the Indexer.

        Args:
            vault_path: The absolute path to the Obsidian vault.
            embedding_generator: An instance of EmbeddingGenerator.
            vector_store: An instance of ChromaVectorStore.
            state_path: Path to the file storing the index state (mtimes).
        """
        self.vault_path = vault_path
        self.embedder = embedding_generator
        self.vector_store = vector_store
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._indexed_mtimes: Dict[str, float] = self._load_index_state()

    def _load_index_state(self) -> Dict[str, float]:
        """Loads the last known modification times from the state file."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    logger.info(
                        f"Loaded index state for {len(state)} files from {self.state_path}")
                    return state
            except (json.JSONDecodeError, OSError) as e:
                logger.error(
                    f"Error loading index state from {self.state_path}: {e}. Starting fresh.", exc_info=True)
        else:
            logger.info(
                f"Index state file not found at {self.state_path}. Starting fresh.")
        return {}

    def _save_index_state(self) -> None:
        """Saves the current modification times to the state file."""
        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self._indexed_mtimes, f, indent=2)
            logger.info(
                f"Saved index state for {len(self._indexed_mtimes)} files to {self.state_path}")
        except OSError as e:
            logger.error(
                f"Error saving index state to {self.state_path}: {e}", exc_info=True)

    def _process_file(self, file_path: Path) -> int:
        """Parses, chunks, embeds, and adds a single file to the vector store.

        Args:
            file_path: The path to the Markdown file.

        Returns:
            The number of chunks added for this file, or 0 on failure.
        """
        logger.info(f"Processing file: {file_path}")
        parsed_note = parse_markdown_file(file_path)
        if not parsed_note:
            logger.warning(f"Skipping file due to parsing error: {file_path}")
            return 0

        chunks: List[TextChunk] = list(chunk_by_paragraph(parsed_note))
        if not chunks:
            logger.info(f"No valid chunks found in file: {file_path}")
            # Update mtime even if no chunks, to avoid reprocessing empty files
            return 0

        try:
            texts_to_embed = [chunk.content for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(texts_to_embed)

            if len(embeddings) == len(chunks):
                self.vector_store.add_chunks(chunks, embeddings)
                logger.info(
                    f"Successfully processed and added {len(chunks)} chunks for: {file_path}")
                return len(chunks)
            else:
                logger.error(
                    f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)}) for: {file_path}. Skipping add."
                )
                return 0
        except Exception as e:
            logger.error(
                f"Failed to generate embeddings or add chunks for {file_path}: {e}", exc_info=True)
            return 0

    def run_incremental_update(self) -> Dict[str, int]:
        """Performs an incremental update of the vector store.

        Detects new, modified, and deleted files based on modification times.

        Returns:
            A dictionary summarizing the update results (added, updated, deleted count).
        """
        start_time = time.monotonic()
        logger.info("Starting incremental index update...")

        stats = {"added": 0, "updated": 0, "deleted": 0, "processed_chunks": 0}
        current_files: Dict[str, float] = {}
        found_paths: Set[Path] = set()

        # 1. Scan vault for current files and their mtimes
        try:
            for file_path in find_markdown_files(self.vault_path):
                try:
                    mtime = file_path.stat().st_mtime
                    current_files[str(file_path)] = mtime
                    found_paths.add(file_path)
                except OSError as e:
                    logger.warning(
                        f"Could not get mtime for {file_path}: {e}. Skipping.")
        except FileNotFoundError:
            logger.error(f"Vault path {self.vault_path} not found during scan.")
            return stats

        logger.info(
            f"Scan complete. Found {len(current_files)} markdown files on disk.")

        indexed_paths_set = set(self._indexed_mtimes.keys())
        current_paths_set = set(current_files.keys())

        # 2. Identify changes
        new_paths = current_paths_set - indexed_paths_set
        deleted_paths = indexed_paths_set - current_paths_set
        # Check existing indexed paths for modification
        modified_paths = {
            path_str for path_str in indexed_paths_set.intersection(current_paths_set)
            if current_files[path_str] > self._indexed_mtimes[path_str]
        }
        # Files that haven't changed (mtime matches and exists)
        # unchanged_paths = indexed_paths_set.intersection(current_paths_set) - modified_paths

        logger.info(
            f"Identified: {len(new_paths)} new, {len(deleted_paths)} deleted, {len(modified_paths)} modified files.")

        # 3. Process deleted files
        for path_str in deleted_paths:
            logger.info(f"Deleting chunks for deleted file: {path_str}")
            try:
                self.vector_store.delete_chunks_by_source(Path(path_str))
                del self._indexed_mtimes[path_str]  # Remove from our state
                stats["deleted"] += 1
            except Exception as e:
                logger.error(
                    f"Failed to delete chunks for {path_str}: {e}", exc_info=True)

        # 4. Process new and modified files
        files_to_process = {Path(p) for p in new_paths.union(modified_paths)}
        logger.info(f"Processing {len(files_to_process)} new/modified files...")

        for file_path in files_to_process:
            path_str = str(file_path)
            is_update = path_str in modified_paths
            if is_update:
                logger.info(f"Deleting old chunks for modified file: {path_str}")
                try:
                    # Delete old chunks before adding new ones
                    self.vector_store.delete_chunks_by_source(file_path)
                except Exception as e:
                    logger.error(
                        f"Failed to delete old chunks for modified file {path_str}: {e}", exc_info=True)
                    continue  # Skip processing this file if deletion failed

            processed_count = self._process_file(file_path)
            if processed_count > 0:
                self._indexed_mtimes[path_str] = current_files[path_str]
                stats["processed_chunks"] += processed_count
                if is_update:
                    stats["updated"] += 1
                else:
                    stats["added"] += 1
            elif path_str in current_files:  # Record mtime even if file processing failed or yielded 0 chunks
                self._indexed_mtimes[path_str] = current_files[path_str]
                # Optionally count as added/updated even with 0 chunks?
                # if is_update:
                #     stats["updated"] += 1
                # else:
                #     stats["added"] += 1

        # 5. Save updated state
        self._save_index_state()

        end_time = time.monotonic()
        duration = end_time - start_time
        logger.info(f"Incremental index update finished in {duration:.2f} seconds.")
        logger.info(
            f"Stats: Added={stats['added']}, Updated={stats['updated']}, Deleted={stats['deleted']}, Total Chunks Processed={stats['processed_chunks']}")
        return stats

    def run_full_reindex(self) -> Dict[str, int]:
        """Performs a full re-index of the entire vault.

        Clears the existing collection and state before processing all files.

        Returns:
            A dictionary summarizing the reindex results.
        """
        start_time = time.monotonic()
        logger.warning(
            "Starting full re-index. Clearing existing collection and state...")

        stats = {"processed_files": 0, "processed_chunks": 0}
        self._indexed_mtimes = {}  # Clear in-memory state
        self._save_index_state()  # Clear persisted state

        if self.vector_store.collection:
            try:
                # Delete and recreate the collection to ensure it's empty
                collection_name = self.vector_store.collection_name
                logger.info(f"Deleting existing collection: {collection_name}")
                self.vector_store.client.delete_collection(collection_name)
                logger.info(f"Recreating collection: {collection_name}")
                self.vector_store.collection = self.vector_store.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": self.vector_store.distance_function}
                )
            except Exception as e:
                logger.error(
                    f"Failed to clear ChromaDB collection '{self.vector_store.collection_name}': {e}. Aborting re-index.", exc_info=True)
                return stats
        else:
            logger.warning("Vector store collection was not initialized. Cannot clear.")
            # Attempt to proceed anyway, assuming it might be created on first add

        # Process all files
        try:
            for file_path in find_markdown_files(self.vault_path):
                try:
                    mtime = file_path.stat().st_mtime
                    processed_count = self._process_file(file_path)
                    if processed_count > 0:
                        self._indexed_mtimes[str(file_path)] = mtime
                        stats["processed_files"] += 1
                        stats["processed_chunks"] += processed_count
                    else:
                        # Record mtime even if processing failed/yielded 0 chunks, so it's not picked up as 'new' later
                        self._indexed_mtimes[str(file_path)] = mtime
                        # Count as processed even if 0 chunks
                        stats["processed_files"] += 1

                except OSError as e:
                    logger.warning(
                        f"Could not get mtime or process {file_path}: {e}. Skipping.")
        except FileNotFoundError:
            logger.error(f"Vault path {self.vault_path} not found during scan.")
            return stats

        self._save_index_state()
        end_time = time.monotonic()
        duration = end_time - start_time
        logger.info(f"Full re-index finished in {duration:.2f} seconds.")
        logger.info(
            f"Stats: Files Processed={stats['processed_files']}, Total Chunks Added={stats['processed_chunks']}")
        return stats


# Example Usage (requires setting up embedder and store correctly)
# if __name__ == '__main__':
#     from bidian.config.logging_config import setup_logging
#     import tempfile
#     import time
#
#     setup_logging("INFO")
#
#     # --- Create dummy vault and files ---
#     with tempfile.TemporaryDirectory() as tmpdir:
#         vault_dir = Path(tmpdir) / "test_vault"
#         vault_dir.mkdir()
#         data_dir = Path(tmpdir) / "test_data"
#         data_dir.mkdir()
#         models_dir = Path(tmpdir) / "test_models"
#         models_dir.mkdir()
#
#         # Create dummy model file path (won't actually load)
#         dummy_model_path = models_dir / "dummy_model.gguf"
#         dummy_model_path.touch()
#
#         # Create dummy files
#         (vault_dir / "note1.md").write_text("Content for note 1.")
#         time.sleep(0.01) # Ensure mtime difference
#         (vault_dir / "note2.md").write_text("Content for note 2.\n\nAnother paragraph.")
#         (vault_dir / "subdir").mkdir()
#         (vault_dir / "subdir" / "note3.md").write_text("Subdirectory note.")
#
#         # --- Mocking dependencies ---
#         # In a real scenario, initialize these properly
#         class MockEmbeddingGenerator:
#             def __init__(self):
#                 self.dim = 4 # Dummy dimension
#                 print("Initialized MockEmbeddingGenerator")
#             def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#                 print(f"Mock generating embeddings for {len(texts)} texts...")
#                 return [[0.1 * i] * self.dim for i in range(len(texts))]
#             def get_embedding_dim(self) -> Optional[int]:
#                 return self.dim
#
#         # Use the actual ChromaVectorStore but point to temp data dir
#         mock_embedder = MockEmbeddingGenerator()
#         chroma_path = str(data_dir / "chroma_db")
#         state_file = str(data_dir / "index_state.json")
#         vector_store = ChromaVectorStore(embedding_dimension=mock_embedder.get_embedding_dim(), persist_path=chroma_path)
#
#         # --- Initialize Indexer ---
#         indexer = Indexer(
#             vault_path=vault_dir,
#             embedding_generator=mock_embedder,
#             vector_store=vector_store,
#             state_path=state_file
#         )
#
#         # --- Run first update (all files are new) ---
#         print("\n--- Running first incremental update ---")
#         stats1 = indexer.run_incremental_update()
#         print(f"Update 1 Stats: {stats1}")
#         print(f"Vector Store Count: {vector_store.count()}")
#
#         # --- Simulate modification and new file ---
#         print("\n--- Simulating changes ---")
#         time.sleep(0.1) # Ensure mtime difference
#         (vault_dir / "note1.md").write_text("Updated content for note 1.")
#         (vault_dir / "note4.md").write_text("A new note.")
#
#         # --- Run second update ---
#         print("\n--- Running second incremental update ---")
#         stats2 = indexer.run_incremental_update()
#         print(f"Update 2 Stats: {stats2}")
#         print(f"Vector Store Count: {vector_store.count()} (Should reflect changes)")
#
#         # --- Simulate deletion ---
#         print("\n--- Simulating deletion ---")
#         (vault_dir / "note2.md").unlink()
#
#         # --- Run third update ---
#         print("\n--- Running third incremental update ---")
#         stats3 = indexer.run_incremental_update()
#         print(f"Update 3 Stats: {stats3}")
#         print(f"Vector Store Count: {vector_store.count()} (Should reflect deletion)")
#
#         # --- Test full reindex ---
#         # print("\n--- Running full re-index ---")
#         # reindex_stats = indexer.run_full_reindex()
#         # print(f"Re-index Stats: {reindex_stats}")
#         # print(f"Vector Store Count after re-index: {vector_store.count()}")
