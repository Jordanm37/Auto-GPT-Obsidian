"""Tests for the retriever module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from bidian.core.retriever import Retriever, RelatedNoteResult, DEFAULT_DISTANCE_THRESHOLD
from bidian.indexing.embedding import EmbeddingGenerator
from bidian.indexing.vector_store import ChromaVectorStore
from bidian.indexing.chunking import TextChunk

# --- Mocks and Fixtures ---


@pytest.fixture
def mock_embedder(mocker):  # type: ignore
    mock = MagicMock(spec=EmbeddingGenerator)
    # Mock embedding generation: returns a fixed vector for simplicity
    mock.generate_embeddings.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts]
    return mock


@pytest.fixture
def mock_vector_store(mocker):  # type: ignore
    mock = MagicMock(spec=ChromaVectorStore)
    # Setup default return value for query
    mock.query.return_value = []
    return mock


@pytest.fixture
def retriever(mock_embedder, mock_vector_store):  # type: ignore
    """Creates a Retriever instance with mocked dependencies."""
    return Retriever(embedding_generator=mock_embedder, vector_store=mock_vector_store)


# Sample chunk data for vector store mock return
PATH_A = Path("/vault/Note A.md")
PATH_B = Path("/vault/Note B.md")
PATH_C = Path("/vault/Note C.md")

MOCK_QUERY_RESULTS = [
    (TextChunk(source_path=PATH_B, chunk_index=0, content="Content B1"), 0.1),
    (TextChunk(source_path=PATH_A, chunk_index=1, content="Content A2"), 0.2),
    (TextChunk(source_path=PATH_B, chunk_index=1, content="Content B2"), 0.3),
    (TextChunk(source_path=PATH_C, chunk_index=0, content="Content C1"), 0.4),
    (TextChunk(source_path=PATH_A, chunk_index=0, content="Content A1"), 0.5),
    # Above default threshold
    (TextChunk(source_path=PATH_C, chunk_index=1, content="Content C2"), 0.8),
]

# --- Tests for Retriever ---


# type: ignore
def test_find_related_notes_success(retriever, mock_vector_store, mock_embedder):
    """Test finding related notes successfully."""
    mock_vector_store.query.return_value = MOCK_QUERY_RESULTS
    query_text = "test query"
    results = retriever.find_related_notes(query_text, k=3)

    # Verify embedder and store were called
    mock_embedder.generate_embeddings.assert_called_once_with([query_text])
    mock_vector_store.query.assert_called_once()
    # Check query args (embedding vector is fixed by mock)
    call_args, call_kwargs = mock_vector_store.query.call_args
    assert call_kwargs.get('query_embedding') == [0.1, 0.2, 0.3]
    # Initial k should be larger than requested k (default logic max(k*2, k+5))
    assert call_kwargs.get('k') == max(3*2, 3+5)

    # Verify results (filtered by default threshold 0.7, grouped by path, sorted, limited by k)
    # Expected order based on MOCK_QUERY_RESULTS and threshold 0.7:
    # 1. Note B (min distance 0.1 from chunk B1)
    # 2. Note A (min distance 0.2 from chunk A2)
    # 3. Note C (min distance 0.4 from chunk C1)
    # Note C chunk 1 (dist 0.8) is excluded by threshold.
    assert len(results) == 3
    assert results[0].file_path == PATH_B
    assert results[0].score == 0.1
    assert results[0].best_chunk_index == 0  # From B1

    assert results[1].file_path == PATH_A
    assert results[1].score == 0.2
    assert results[1].best_chunk_index == 1  # From A2

    assert results[2].file_path == PATH_C
    assert results[2].score == 0.4
    assert results[2].best_chunk_index == 0  # From C1


# type: ignore
def test_find_related_notes_custom_k_threshold(retriever, mock_vector_store, mock_embedder):
    """Test finding related notes with custom k and threshold."""
    mock_vector_store.query.return_value = MOCK_QUERY_RESULTS
    query_text = "custom query"
    k = 1
    threshold = 0.15  # Should only include Note B

    results = retriever.find_related_notes(
        query_text, k=k, distance_threshold=threshold)

    mock_vector_store.query.assert_called_once()
    call_args, call_kwargs = mock_vector_store.query.call_args
    # Initial k should be larger
    assert call_kwargs.get('k') == max(k*2, k+5)

    assert len(results) == 1
    assert results[0].file_path == PATH_B
    assert results[0].score == 0.1


# type: ignore
def test_find_related_notes_no_results_above_threshold(retriever, mock_vector_store):
    """Test when no results meet the distance threshold."""
    # Return results, but all with high distance
    high_dist_results = [
        (TextChunk(source_path=PATH_A, chunk_index=0, content="A1"), 0.9),
        (TextChunk(source_path=PATH_B, chunk_index=0, content="B1"), 0.95),
    ]
    mock_vector_store.query.return_value = high_dist_results
    results = retriever.find_related_notes("query", distance_threshold=0.8)
    assert len(results) == 0


# type: ignore
def test_find_related_notes_no_results_from_store(retriever, mock_vector_store):
    """Test when the vector store query itself returns nothing."""
    mock_vector_store.query.return_value = []  # Store returns empty
    results = retriever.find_related_notes("query")
    assert len(results) == 0


# type: ignore
def test_find_related_notes_empty_query(retriever, mock_embedder, mock_vector_store):
    """Test providing an empty query string."""
    results = retriever.find_related_notes("")
    assert results == []
    mock_embedder.generate_embeddings.assert_not_called()
    mock_vector_store.query.assert_not_called()


def test_find_related_notes_embedding_error(retriever, mock_embedder):  # type: ignore
    """Test handling of errors during query embedding generation."""
    mock_embedder.generate_embeddings.side_effect = Exception("Embedding failed")
    results = retriever.find_related_notes("query")
    assert results == []


# type: ignore
def test_find_related_notes_vector_store_error(retriever, mock_vector_store):
    """Test handling of errors during vector store query."""
    mock_vector_store.query.side_effect = Exception("DB query failed")
    results = retriever.find_related_notes("query")
    assert results == []
