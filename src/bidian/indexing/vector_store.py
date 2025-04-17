"""Module for interacting with a ChromaDB vector store."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from bidian.indexing.chunking import TextChunk

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_CHROMA_PATH = os.environ.get(
    "BIDIAN_CHROMA_PATH",
    "./data/chroma_db"
)
DEFAULT_COLLECTION_NAME = os.environ.get(
    "BIDIAN_COLLECTION_NAME",
    "bidian_notes_v1"
)
# Default distance function for embeddings
DEFAULT_DISTANCE_FUNCTION = "cosine"


class ChromaVectorStore:
    """Manages interactions with a persistent ChromaDB vector collection."""

    def __init__(self,
                 embedding_dimension: int,
                 persist_path: str = DEFAULT_CHROMA_PATH,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 distance_function: str = DEFAULT_DISTANCE_FUNCTION) -> None:
        """Initializes the ChromaDB client and gets/creates the collection.

        Args:
            embedding_dimension: The dimensionality of the vectors to be stored.
            persist_path: Path to the directory for persistent storage.
            collection_name: Name of the ChromaDB collection to use.
            distance_function: Distance function for similarity ('cosine', 'l2', etc.).

        Raises:
            ValueError: If the existing collection has a different embedding dimension.
            Exception: Any exception raised by ChromaDB during initialization.
        """
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_function = distance_function
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None

        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Initializing ChromaDB client with persistence path: {self.persist_path}")
            # Note: Settings can be customized further if needed
            self.client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=Settings(
                    anonymized_telemetry=False  # Recommended for local/privacy focus
                )
            )

            logger.info(f"Getting or creating collection: {self.collection_name}")
            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_function}  # Configure distance
                # embedding_function can be omitted if we provide embeddings directly
            )

            # Validate embedding dimension if collection already existed
            # Note: ChromaDB doesn't directly expose dimension easily after creation.
            # We rely on the initial setup being correct or catching errors during insertion.
            # A potential check: try adding a dummy vector of the expected dimension.
            logger.info(
                f"Vector store initialized. Collection '{self.collection_name}' ready.")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaVectorStore: {e}", exc_info=True)
            self.client = None
            self.collection = None
            raise

    def _generate_chunk_id(self, chunk: TextChunk) -> str:
        """Creates a unique ID for a TextChunk."""
        # Using '::' as a separator, ensure file paths don't conflict if possible
        return f"{str(chunk.source_path)}::{chunk.chunk_index}"

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        """Adds or updates text chunks and their embeddings in the collection.

        Args:
            chunks: A list of TextChunk objects.
            embeddings: A list of corresponding embedding vectors.

        Raises:
            ValueError: If the number of chunks and embeddings doesn't match.
            RuntimeError: If the ChromaDB collection is not available.
            Exception: Any exception raised by ChromaDB during upsert.
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection is not initialized.")

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must be the same.")

        if not chunks:
            logger.info("No chunks provided to add. Skipping.")
            return

        ids: List[str] = [self._generate_chunk_id(chunk) for chunk in chunks]
        docs: List[str] = [chunk.content for chunk in chunks]
        metadatas: List[Dict[str, Any]] = [
            {"source_path": str(chunk.source_path), "chunk_index": chunk.chunk_index}
            for chunk in chunks
        ]

        try:
            logger.info(
                f"Adding/updating {len(ids)} chunks in collection '{self.collection_name}'.")
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metadatas
            )
            logger.info(f"Successfully added/updated {len(ids)} chunks.")
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}", exc_info=True)
            raise

    def query(self, query_embedding: List[float], k: int = 5,
              where_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[TextChunk, float]]:
        """Queries the collection for the most similar chunks.

        Args:
            query_embedding: The embedding vector to query against.
            k: The number of nearest neighbors to return.
            where_filter: Optional ChromaDB `where` filter dictionary.

        Returns:
            A list of tuples, each containing a TextChunk and its distance
            to the query embedding, sorted by distance.
            Returns an empty list if no results or on error.

        Raises:
            RuntimeError: If the ChromaDB collection is not available.
            Exception: Any exception raised by ChromaDB during query.
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection is not initialized.")

        results = []
        try:
            logger.info(
                f"Querying collection '{self.collection_name}' for {k} nearest neighbors.")
            query_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=['metadatas', 'documents', 'distances']
            )
            logger.debug(f"Raw query results: {query_results}")

            # Process results (Chroma returns lists for each requested field)
            if query_results and query_results['ids'] and query_results['ids'][0]:
                ids = query_results['ids'][0]
                distances = query_results['distances'][0]
                metadatas = query_results['metadatas'][0]
                documents = query_results['documents'][0]

                for i, doc_id in enumerate(ids):
                    metadata = metadatas[i]
                    distance = distances[i]
                    content = documents[i]

                    # Reconstruct TextChunk (or as close as possible)
                    try:
                        chunk = TextChunk(
                            source_path=Path(metadata['source_path']),
                            chunk_index=metadata['chunk_index'],
                            content=content
                        )
                        results.append((chunk, distance))
                    except (KeyError, TypeError) as e:
                        logger.warning(
                            f"Could not reconstruct TextChunk for id {doc_id}. Metadata: {metadata}. Error: {e}")

            logger.info(f"Query returned {len(results)} results.")
            # Results should already be sorted by distance by ChromaDB
            return results

        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}", exc_info=True)
            raise  # Re-raise the exception
            # return [] # Optionally return empty list instead of raising

    def delete_chunks_by_source(self, source_path: Path) -> None:
        """Deletes all chunks associated with a specific source file path.

        Args:
            source_path: The path to the source file whose chunks should be deleted.

        Raises:
            RuntimeError: If the ChromaDB collection is not available.
            Exception: Any exception raised by ChromaDB during deletion.
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection is not initialized.")

        source_path_str = str(source_path)
        logger.info(
            f"Deleting chunks from collection '{self.collection_name}' with source_path: {source_path_str}")
        try:
            self.collection.delete(where={"source_path": source_path_str})
            logger.info(f"Successfully deleted chunks for source: {source_path_str}")
        except Exception as e:
            logger.error(
                f"Failed to delete chunks from ChromaDB for source {source_path_str}: {e}", exc_info=True)
            raise

    def count(self) -> int:
        """Returns the total number of items in the collection."""
        if not self.collection:
            logger.error("ChromaDB collection is not initialized. Cannot get count.")
            return 0
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get count from ChromaDB: {e}", exc_info=True)
            return 0

# Example Usage (depends on previous steps: model download, embedding gen)
# if __name__ == '__main__':
#     from bidian.config.logging_config import setup_logging
#     from bidian.indexing.embedding import EmbeddingGenerator
#
#     setup_logging("INFO")
#
#     # --- Assumes EmbeddingGenerator works and model is available ---
#     try:
#         print("Initializing Embedding Generator...")
#         embedder = EmbeddingGenerator()
#         emb_dim = embedder.get_embedding_dim()
#         if not emb_dim:
#             raise RuntimeError("Could not determine embedding dimension.")
#         print(f"Embedding dimension: {emb_dim}")
#
#         print("\nInitializing Vector Store...")
#         vector_store = ChromaVectorStore(embedding_dimension=emb_dim)
#         print(f"Initial collection count: {vector_store.count()}")
#
#         # --- Create dummy data ---
#         dummy_path1 = Path("/vault/notes/note1.md")
#         dummy_path2 = Path("/vault/notes/subdir/note2.md")
#         chunks_to_add = [
#             TextChunk(source_path=dummy_path1, chunk_index=0, content="This is the first chunk from note 1."),
#             TextChunk(source_path=dummy_path1, chunk_index=1, content="This is the second chunk from note 1, about apples."),
#             TextChunk(source_path=dummy_path2, chunk_index=0, content="Note 2 only has one chunk, about oranges."),
#         ]
#         texts_to_embed = [c.content for c in chunks_to_add]
#
#         print(f"\nGenerating embeddings for {len(texts_to_embed)} chunks...")
#         embeddings_to_add = embedder.generate_embeddings(texts_to_embed)
#
#         print("\nAdding chunks to vector store...")
#         vector_store.add_chunks(chunks_to_add, embeddings_to_add)
#         print(f"Collection count after add: {vector_store.count()}")
#
#         # --- Test Query ---
#         query_text = "Tell me about fruit"
#         print(f"\nQuerying for: '{query_text}'")
#         query_embedding = embedder.generate_embeddings([query_text])[0]
#
#         query_results = vector_store.query(query_embedding, k=2)
#         print("Query results:")
#         for chunk, distance in query_results:
#             print(f"  Distance: {distance:.4f}") # noqa: T201
#             print(f"  Source: {chunk.source_path}") # noqa: T201
#             print(f"  Index: {chunk.chunk_index}") # noqa: T201
#             print(f"  Content: {chunk.content}") # noqa: T201
#             print("---") # noqa: T201
#
#         # --- Test Deletion ---
#         print(f"\nDeleting chunks from source: {dummy_path1}")
#         vector_store.delete_chunks_by_source(dummy_path1)
#         print(f"Collection count after delete: {vector_store.count()}")
#
#         # --- Test Query Again ---
#         print(f"\nQuerying again for: '{query_text}'")
#         query_results_after_delete = vector_store.query(query_embedding, k=2)
#         print("Query results after delete:")
#         if not query_results_after_delete:
#             print("  (No results as expected)")
#         else:
#             for chunk, distance in query_results_after_delete:
#                 print(f"  Distance: {distance:.4f}, Source: {chunk.source_path}, Index: {chunk.chunk_index}") # noqa: T201
#
#     except FileNotFoundError:
#         print("\n*** Embedding model file not found. Skipping example usage. ***") # noqa: T201
#     except Exception as e:
#         print(f"\nAn error occurred during VectorStore example usage: {e}") # noqa: T201
