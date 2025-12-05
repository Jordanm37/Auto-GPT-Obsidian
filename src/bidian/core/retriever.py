"""Module for retrieving related notes based on semantic similarity."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from pydantic import BaseModel, Field

from bidian.indexing.embedding import EmbeddingGenerator
from bidian.indexing.vector_store import ChromaVectorStore
from bidian.indexing.chunking import TextChunk  # For result type hint

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default number of results to fetch initially before filtering
DEFAULT_K_NEIGHBORS = 10
# Default distance threshold (lower is more similar, adjust based on embedding model)
DEFAULT_DISTANCE_THRESHOLD = 0.7


class RelatedNoteResult(BaseModel):
    """Represents a single related note found by the retriever."""
    file_path: Path = Field(..., description="Path to the related note file.")
    # Aggregate score/distance (lower is better for distance metrics like cosine)
    score: float = Field(...,
                         description="Relevance score (e.g., minimum distance found).")
    # Optionally include the best matching chunk info
    best_chunk_index: Optional[int] = Field(
        None, description="Index of the most relevant chunk within the note.")
    best_chunk_content: Optional[str] = Field(
        None, description="Content of the most relevant chunk.")


class Retriever:
    """Handles querying the vector store to find related notes."""

    def __init__(self,
                 embedding_generator: EmbeddingGenerator,
                 vector_store: ChromaVectorStore) -> None:
        """Initializes the Retriever.

        Args:
            embedding_generator: An instance of EmbeddingGenerator.
            vector_store: An instance of ChromaVectorStore.
        """
        self.embedder = embedding_generator
        self.vector_store = vector_store

    def find_related_notes(
        self,
        query_text: str,
        k: int = DEFAULT_K_NEIGHBORS,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    ) -> List[RelatedNoteResult]:
        """Finds notes related to the query text using semantic search.

        Args:
            query_text: The text to find related notes for.
            k: The maximum number of related notes to return.
            distance_threshold: The maximum distance for a chunk to be considered
                                related (lower values mean higher similarity).

        Returns:
            A list of RelatedNoteResult objects, sorted by relevance (lowest distance first).
        """
        logger.info(f"Finding related notes for query: '{query_text[:50]}...'")

        if not query_text:
            logger.warning("Received empty query text. Returning no results.")
            return []

        try:
            query_embedding = self.embedder.generate_embeddings([query_text])[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}", exc_info=True)
            return []

        try:
            # Fetch more results initially than k, as filtering might remove some
            # We fetch k * 2 or at least k + 5 to have a buffer
            initial_k = max(k * 2, k + 5)
            chunk_results: List[Tuple[TextChunk, float]] = self.vector_store.query(
                query_embedding=query_embedding,
                k=initial_k
            )
        except Exception as e:
            logger.error(f"Failed to query vector store: {e}", exc_info=True)
            return []

        # Group results by source path and find the minimum distance per note
        notes_min_distance: Dict[Path, Tuple[float, TextChunk]] = {}
        for chunk, distance in chunk_results:
            if distance <= distance_threshold:
                current_min_dist, _ = notes_min_distance.get(
                    chunk.source_path, (float('inf'), None))
                if distance < current_min_dist:
                    notes_min_distance[chunk.source_path] = (distance, chunk)

        if not notes_min_distance:
            logger.info("No notes found within the distance threshold.")
            return []

        # Create results and sort by distance (score)
        results: List[RelatedNoteResult] = []
        for path, (min_dist, best_chunk) in notes_min_distance.items():
            results.append(RelatedNoteResult(
                file_path=path,
                score=min_dist,
                best_chunk_index=best_chunk.chunk_index,
                best_chunk_content=best_chunk.content
            ))

        # Sort by score (ascending distance)
        results.sort(key=lambda r: r.score)

        logger.info(
            f"Found {len(results)} related notes within threshold {distance_threshold}. Returning top {k}.")
        # Return only the top k results after filtering and sorting
        return results[:k]

# Example Usage (requires setup like indexer example)
# if __name__ == '__main__':
#     from bidian.config.logging_config import setup_logging
#     from bidian.indexing.indexer import Indexer # Need this to populate the store
#     import tempfile
#     import time
#
#     setup_logging("INFO")
#
#     # --- Use the same mocking setup as in indexer.py ---
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
#         (vault_dir / "note_apple.md").write_text("This note is about apples.\n\nThey are red or green.")
#         (vault_dir / "note_orange.md").write_text("Oranges are orange and round.")
#         (vault_dir / "note_banana.md").write_text("Bananas are yellow and long.")
#         (vault_dir / "note_code.md").write_text("Let's write some python code.\n\ndef hello():\n  print('hi')")
#
#         class MockEmbeddingGenerator:
#             def __init__(self):
#                 self.dim = 4
#                 # Assign fixed-ish vectors for predictability
#                 self.vectors = {
#                     "apples": [0.1, 0.2, 0.3, 0.4],
#                     "orange": [0.5, 0.4, 0.3, 0.2],
#                     "banana": [0.6, 0.7, 0.8, 0.9],
#                     "python": [0.9, 0.8, 0.7, 0.6],
#                     "fruit":  [0.3, 0.3, 0.3, 0.3] # Query vector
#                 }
#                 self.next_vec = 0.0
#             def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#                 results = []
#                 for text in texts:
#                     text_lower = text.lower()
#                     found = False
#                     for key, vec in self.vectors.items():
#                         if key in text_lower:
#                             results.append(vec)
#                             found = True
#                             break
#                     if not found:
#                          # Default pseudo-random vector
#                         self.next_vec += 0.01
#                         results.append([self.next_vec] * self.dim)
#                 return results
#             def get_embedding_dim(self) -> Optional[int]:
#                 return self.dim
#
#         mock_embedder = MockEmbeddingGenerator()
#         chroma_path = str(data_dir / "chroma_db")
#         state_file = str(data_dir / "index_state.json")
#         vector_store = ChromaVectorStore(embedding_dimension=mock_embedder.get_embedding_dim(), persist_path=chroma_path)
#
#         indexer = Indexer(vault_dir, mock_embedder, vector_store, state_file)
#         print("\n--- Running indexer to populate vector store ---")
#         indexer.run_incremental_update()
#         print(f"Vector Store Count: {vector_store.count()}")
#
#         # --- Initialize Retriever ---
#         retriever = Retriever(mock_embedder, vector_store)
#
#         # --- Run Query ---
#         query = "Tell me about fruit"
#         k_results = 2
#         threshold = 0.8 # Adjust based on mock vectors/distance
#         print(f"\n--- Finding related notes (k={k_results}, threshold={threshold}) for: '{query}' ---")
#
#         # Note: Distances will be calculated based on cosine similarity (default)
#         # Since vectors are mocked, actual distances might not be intuitive.
#         # We mainly test the flow and filtering.
#         related_notes = retriever.find_related_notes(query, k=k_results, distance_threshold=threshold)
#
#         print("\nFound related notes:")
#         if not related_notes:
#             print("  None found within threshold.")
#         else:
#             for note in related_notes:
#                 print(f"  Path: {note.file_path.name}") # noqa: T201
#                 print(f"  Score (Distance): {note.score:.4f}") # noqa: T201
#                 print(f"  Best Chunk Index: {note.best_chunk_index}") # noqa: T201
#                 print(f"  Best Chunk Content: {note.best_chunk_content[:80]}...") # noqa: T201
#                 print("---") # noqa: T201
