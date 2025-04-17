"""Tests for the vector_store module."""

import pytest
from pathlib import Path
import time

from bidian.indexing.vector_store import ChromaVectorStore
from bidian.indexing.chunking import TextChunk

# --- Constants ---

TEST_EMBEDDING_DIM = 3
TEST_COLLECTION_NAME = "test_collection"

# --- Fixtures ---


@pytest.fixture
def temp_chroma_path(tmp_path: Path) -> str:
    """Provides a temporary path for ChromaDB persistence."""
    return str(tmp_path / "test_chroma_db")


@pytest.fixture
def vector_store(temp_chroma_path: str) -> ChromaVectorStore:
    """Creates a ChromaVectorStore instance using a temporary path."""
    vs = ChromaVectorStore(
        embedding_dimension=TEST_EMBEDDING_DIM,
        persist_path=temp_chroma_path,
        collection_name=TEST_COLLECTION_NAME
    )
    # Ensure collection is clean before test
    if vs.collection:
        vs.client.delete_collection(vs.collection_name)
        vs.collection = vs.client.get_or_create_collection(
            name=vs.collection_name,
            metadata={"hnsw:space": vs.distance_function}
        )
    yield vs
    # Teardown: client might hold resources, though persistent client is simpler.
    # Resetting/deleting the collection is handled by chromadb itself on path deletion.
    # Explicitly resetting can be good practice if needed:
    # try:
    #     if vs.client:
    #        vs.client.reset() # Resets the entire database state!
    # except Exception:
    #     pass


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    """Provides a list of sample TextChunk objects."""
    path1 = Path("/vault/note1.md")
    path2 = Path("/vault/notes/note2.md")
    return [
        TextChunk(source_path=path1, chunk_index=0, content="Apple is red."),
        TextChunk(source_path=path1, chunk_index=1, content="Orange is orange."),
        TextChunk(source_path=path2, chunk_index=0, content="Banana is yellow."),
        TextChunk(source_path=path2, chunk_index=1, content="Apple is also green."),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Provides sample embeddings corresponding to sample_chunks."""
    # Simple mock embeddings (ensure dimension matches TEST_EMBEDDING_DIM)
    return [
        [0.1, 0.2, 0.3],  # apple red
        [0.5, 0.4, 0.3],  # orange orange
        [0.8, 0.9, 0.1],  # banana yellow
        [0.1, 0.3, 0.2],  # apple green
    ]


# --- Tests for ChromaVectorStore ---

def test_vector_store_init(vector_store: ChromaVectorStore, temp_chroma_path: str):
    """Test successful initialization and collection creation."""
    assert vector_store.client is not None
    assert vector_store.collection is not None
    assert vector_store.collection.name == TEST_COLLECTION_NAME
    assert Path(temp_chroma_path).is_dir()  # Check persistence path created
    assert vector_store.count() == 0


def test_add_chunks(vector_store: ChromaVectorStore, sample_chunks: list[TextChunk], sample_embeddings: list[list[float]]):
    """Test adding chunks to the collection."""
    vector_store.add_chunks(sample_chunks, sample_embeddings)
    assert vector_store.count() == len(sample_chunks)

    # Verify one added item (optional, requires knowing internal ID format)
    test_id = vector_store._generate_chunk_id(sample_chunks[0])
    retrieved = vector_store.collection.get(ids=[test_id], include=['documents'])
    assert retrieved['ids'] == [test_id]
    assert retrieved['documents'] == [sample_chunks[0].content]


def test_add_chunks_mismatched_length(vector_store: ChromaVectorStore, sample_chunks: list[TextChunk]):
    """Test error handling when chunks and embeddings lengths differ."""
    mismatched_embeddings = [[0.1] * TEST_EMBEDDING_DIM]  # Only one embedding
    with pytest.raises(ValueError, match="Number of chunks and embeddings must be the same"):
        vector_store.add_chunks(sample_chunks, mismatched_embeddings)


def test_add_chunks_empty(vector_store: ChromaVectorStore):
    """Test adding an empty list of chunks."""
    vector_store.add_chunks([], [])
    assert vector_store.count() == 0


def test_query_simple(vector_store: ChromaVectorStore, sample_chunks: list[TextChunk], sample_embeddings: list[list[float]]):
    """Test querying the vector store."""
    vector_store.add_chunks(sample_chunks, sample_embeddings)

    # Query for something close to "Apple is red."
    query_vec = [0.11, 0.19, 0.31]
    results = vector_store.query(query_vec, k=2)

    assert len(results) == 2
    # Expect chunk 0 (Apple is red) to be first, then chunk 3 (Apple is also green)
    assert results[0][0].source_path == sample_chunks[0].source_path
    assert results[0][0].chunk_index == sample_chunks[0].chunk_index
    assert results[0][0].content == sample_chunks[0].content
    # Check second result
    assert results[1][0].chunk_index == sample_chunks[3].chunk_index
    assert results[0][1] < results[1][1]  # Check distance ordering


def test_query_with_filter(vector_store: ChromaVectorStore, sample_chunks: list[TextChunk], sample_embeddings: list[list[float]]):
    """Test querying with a metadata filter."""
    vector_store.add_chunks(sample_chunks, sample_embeddings)

    # Query for apple, but only from note2.md
    query_vec = [0.1, 0.25, 0.25]  # Close to apple embeddings
    note2_path_str = str(Path("/vault/notes/note2.md"))
    results = vector_store.query(query_vec, k=3, where_filter={
                                 "source_path": note2_path_str})

    assert len(results) == 1  # Only chunk 3 ("Apple is also green.") is from note2
    assert results[0][0].source_path == Path(note2_path_str)
    assert results[0][0].chunk_index == 1  # Index 1 within note2.md
    assert results[0][0].content == sample_chunks[3].content


def test_query_no_results(vector_store: ChromaVectorStore, sample_chunks: list[TextChunk], sample_embeddings: list[list[float]]):
    """Test query returning no results within threshold (implicitly tested by k)."""
    # Chroma query doesn't have a distance threshold directly, k limits it.
    vector_store.add_chunks(sample_chunks, sample_embeddings)
    query_vec = [0.99, 0.99, 0.99]  # Very different vector
    results = vector_store.query(query_vec, k=1)
    # Should still return the closest one, even if far away
    assert len(results) == 1


def test_delete_chunks_by_source(vector_store: ChromaVectorStore, sample_chunks: list[TextChunk], sample_embeddings: list[list[float]]):
    """Test deleting chunks associated with a specific source file."""
    vector_store.add_chunks(sample_chunks, sample_embeddings)
    assert vector_store.count() == 4

    path_to_delete = Path("/vault/note1.md")
    vector_store.delete_chunks_by_source(path_to_delete)

    assert vector_store.count() == 2  # Only chunks from note2 should remain

    # Verify remaining chunks are from note2
    remaining = vector_store.collection.get(include=['metadatas'])
    assert len(remaining['ids']) == 2
    assert all(meta['source_path'] == str(Path("/vault/notes/note2.md"))
               for meta in remaining['metadatas'])


def test_delete_chunks_nonexistent_source(vector_store: ChromaVectorStore, sample_chunks: list[TextChunk], sample_embeddings: list[list[float]]):
    """Test deleting chunks for a source path that has no entries."""
    vector_store.add_chunks(sample_chunks, sample_embeddings)
    assert vector_store.count() == 4

    non_existent_path = Path("/vault/note_other.md")
    # Should execute without error
    vector_store.delete_chunks_by_source(non_existent_path)

    assert vector_store.count() == 4  # Count should be unchanged
