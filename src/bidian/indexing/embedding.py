"""Functions for generating text embeddings using GGUF models."""

import logging
import os
from pathlib import Path
from typing import List, Optional

from llama_cpp import Llama

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Configuration ---
# TODO: Move model path to a configuration file or environment variable
# Example GGUF embedding model: bge-small-en-v1.5-f16.gguf
# Download from Hugging Face Hub: https://huggingface.co/BAAI/bge-small-en-v1.5-gguf
DEFAULT_EMBEDDING_MODEL_PATH = os.environ.get(
    "BIDIAN_EMBEDDING_MODEL_PATH",
    "./models/bge-small-en-v1.5-f16.gguf"  # Default path, ensure model exists here
)
# It's recommended to use more GPU layers if available, adjust n_gpu_layers accordingly.
# -1 means try to offload all layers.
DEFAULT_N_GPU_LAYERS = int(os.environ.get("BIDIAN_N_GPU_LAYERS", "0"))
DEFAULT_N_CTX = int(os.environ.get("BIDIAN_EMBEDDING_N_CTX", "512")
                    )  # Max sequence length


class EmbeddingGenerator:
    """Handles loading a GGUF model and generating embeddings for text.

    Attributes:
        model_path: The path to the GGUF embedding model file.
        model: The loaded Llama model instance, or None if loading failed.
    """

    def __init__(self,
                 model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
                 n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
                 n_ctx: int = DEFAULT_N_CTX,
                 verbose: bool = False) -> None:
        """Initializes the EmbeddingGenerator and loads the model.

        Args:
            model_path: Path to the GGUF model file.
            n_gpu_layers: Number of layers to offload to GPU. 0 for CPU only, -1 for all.
            n_ctx: The context size (max sequence length) for the model.
            verbose: Whether to enable verbose logging from llama_cpp.

        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
        """
        self.model_path = Path(model_path)
        self.model: Optional[Llama] = None

        if not self.model_path.is_file():
            msg = (
                f"Embedding model not found at: {self.model_path}. "
                f"Please ensure the model exists or set the 'BIDIAN_EMBEDDING_MODEL_PATH' env var."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            logger.info(f"Loading embedding model from: {self.model_path}")
            logger.info(f"Using n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
            self.model = Llama(
                model_path=str(self.model_path),
                embedding=True,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=verbose  # Set to True for detailed llama_cpp logs
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load embedding model '{self.model_path}': {e}", exc_info=True)
            # Keep self.model as None
            raise  # Re-raise the exception to signal failure

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text strings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors (each a list of floats).
            Returns an empty list if the model failed to load or if input is empty.

        Raises:
            RuntimeError: If the model was not loaded successfully.
            Exception: Any exception raised by the underlying llama_cpp library during embedding.
        """
        if not self.model:
            logger.error("Embedding model is not loaded. Cannot generate embeddings.")
            raise RuntimeError("Embedding model not loaded.")

        if not texts:
            logger.warning(
                "Received empty list of texts for embedding. Returning empty list.")
            return []

        try:
            logger.info(f"Generating embeddings for {len(texts)} text chunk(s)...")
            embeddings = self.model.embed(texts)
            logger.info(
                f"Successfully generated {len(embeddings)} embedding vector(s).")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise  # Re-raise the exception

    def get_embedding_dim(self) -> Optional[int]:
        """Returns the dimensionality of the embeddings produced by the model."""
        if not self.model:
            logger.error(
                "Embedding model is not loaded. Cannot determine embedding dimension.")
            return None
        try:
            # llama-cpp-python exposes embedding size via embedding_length
            # It might require model context to be initialized, let's check if available.
            # A simple way is to embed an empty string if needed, but let's check the attribute first.
            # Note: The attribute might not be present in all versions or directly accessible easily.
            # Let's try embedding a dummy string to get the dimension.
            dummy_embedding = self.model.embed(["test"])
            return len(dummy_embedding[0])
        except Exception as e:
            logger.error(f"Could not determine embedding dimension: {e}", exc_info=True)
            return None

# Example Usage (requires a GGUF model at the default path or env var)
# if __name__ == '__main__':
#     from bidian.config.logging_config import setup_logging
#
#     setup_logging("INFO")
#
#     # --- IMPORTANT ---
#     # Ensure you have a GGUF embedding model downloaded and placed at the correct path.
#     # For example, download bge-small-en-v1.5-f16.gguf from Hugging Face
#     # and place it in a 'models' subdirectory relative to this script, or set the env var.
#     # mkdir models
#     # wget -P models https://huggingface.co/BAAI/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf
#     # ---
#
#     try:
#         generator = EmbeddingGenerator(verbose=True)
#         print(f"Model loaded. Embedding dimension: {generator.get_embedding_dim()}")
#
#         test_texts = [
#             "This is the first document.",
#             "This document is the second document.",
#             "And this is the third one.",
#             "Is this the first document?",
#         ]
#
#         embeddings = generator.generate_embeddings(test_texts)
#
#         print(f"\nGenerated {len(embeddings)} embeddings.")
#         for i, emb in enumerate(embeddings):
#             print(f"Embedding {i} (first 5 dims): {emb[:5]}...") # noqa: T201
#
#     except FileNotFoundError:
#         print("\n*** Embedding model file not found. Skipping example usage. ***") # noqa: T201
#         print("Please download a GGUF model and place it in ./models/ or set BIDIAN_EMBEDDING_MODEL_PATH.") # noqa: T201
#     except Exception as e:
#         print(f"\nAn error occurred during example usage: {e}") # noqa: T201
