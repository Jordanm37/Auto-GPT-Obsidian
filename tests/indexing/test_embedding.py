"""Tests for the embedding module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from bidian.indexing.embedding import EmbeddingGenerator, DEFAULT_EMBEDDING_MODEL_PATH


# --- Fixtures ---

@pytest.fixture
def mock_llama(mocker):  # type: ignore
    """Mocks the llama_cpp.Llama class."""
    mock_llama_instance = MagicMock()
    # Configure the mock 'embed' method

    def mock_embed(texts: list[str]) -> list[list[float]]:
        # Simple mock embedding: vector dimension 3, value based on text length
        return [[len(text) / 10.0] * 3 for text in texts]
    mock_llama_instance.embed = mock_embed

    # Mock the class constructor to return our instance
    mock_llama_class = mocker.patch(
        "bidian.indexing.embedding.Llama", return_value=mock_llama_instance)
    return mock_llama_class, mock_llama_instance


@pytest.fixture
def embedding_generator(tmp_path: Path, mock_llama):  # type: ignore
    """Creates an EmbeddingGenerator instance with a mocked Llama and dummy model path."""
    # Create a dummy model file so the FileNotFoundError check passes
    dummy_model_path = tmp_path / "dummy_model.gguf"
    dummy_model_path.touch()

    generator = EmbeddingGenerator(model_path=str(dummy_model_path))
    # The mock_llama fixture already patched the Llama class, so the real
    # Llama() won't be called, but the path check still happens.
    return generator


@pytest.fixture
def embedding_generator_load_fails(tmp_path: Path, mocker):  # type: ignore
    """Simulates a failure during Llama model loading."""
    dummy_model_path = tmp_path / "dummy_model_fail.gguf"
    dummy_model_path.touch()

    # Patch Llama to raise an exception during __init__
    mocker.patch("bidian.indexing.embedding.Llama",
                 side_effect=Exception("Mock Llama load failed"))

    with pytest.raises(Exception, match="Mock Llama load failed"):
        EmbeddingGenerator(model_path=str(dummy_model_path))
    # This fixture doesn't return anything, it just sets up the expectation


# --- Tests for EmbeddingGenerator ---

# type: ignore
def test_embedding_generator_init_success(embedding_generator, mock_llama):
    """Test successful initialization with mocked Llama."""
    mock_llama_class, _ = mock_llama
    assert embedding_generator.model is not None
    # Check if Llama was called with expected embedding=True flag
    mock_llama_class.assert_called_once()
    call_args, call_kwargs = mock_llama_class.call_args
    assert call_kwargs.get('embedding') is True


def test_embedding_generator_init_file_not_found():
    """Test initialization failure when the model file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        EmbeddingGenerator(model_path="/path/to/nonexistent/model.gguf")


# type: ignore
def test_embedding_generator_init_load_failure(embedding_generator_load_fails):
    """Test handling of exceptions during Llama model loading."""
    # The fixture itself contains the assertion via pytest.raises
    pass


def test_generate_embeddings_success(embedding_generator, mock_llama):  # type: ignore
    """Test generating embeddings successfully using the mocked model."""
    _, mock_llama_instance = mock_llama
    test_texts = ["hello", "world", "test document"]
    embeddings = embedding_generator.generate_embeddings(test_texts)

    assert len(embeddings) == len(test_texts)
    assert len(embeddings[0]) == 3  # Dimension from mock_embed
    # Check mock values
    assert embeddings[0] == [0.5] * 3  # len("hello") = 5
    assert embeddings[1] == [0.5] * 3  # len("world") = 5
    assert embeddings[2] == [1.3] * 3  # len("test document") = 13

    # Check that the mock model's embed method was called
    mock_llama_instance.embed.assert_called_once_with(test_texts)


def test_generate_embeddings_empty_list(embedding_generator):  # type: ignore
    """Test generating embeddings with an empty input list."""
    embeddings = embedding_generator.generate_embeddings([])
    assert embeddings == []


def test_generate_embeddings_model_not_loaded(tmp_path: Path):
    """Test calling generate_embeddings when the model failed to load."""
    # Don't use the fixture that patches Llama successfully
    # Create instance without calling __init__
    generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
    generator.model = None  # Explicitly set model to None

    with pytest.raises(RuntimeError, match="Embedding model not loaded"):
        generator.generate_embeddings(["test"])


def test_get_embedding_dim_success(embedding_generator, mock_llama):  # type: ignore
    """Test getting embedding dimension from the mocked model."""
    _, mock_llama_instance = mock_llama
    dim = embedding_generator.get_embedding_dim()
    assert dim == 3  # Dimension from mock_embed
    # Check that embed was called by get_embedding_dim
    mock_llama_instance.embed.assert_called_with(
        ["test"])  # Called by get_embedding_dim


def test_get_embedding_dim_model_not_loaded(tmp_path: Path):
    """Test getting dimension when model isn't loaded."""
    generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
    generator.model = None
    assert generator.get_embedding_dim() is None
