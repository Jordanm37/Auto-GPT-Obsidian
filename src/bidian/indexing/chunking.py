"""Functions for splitting text content into manageable chunks for embedding."""

import logging
from pathlib import Path
from typing import List, Generator, Dict, Any

from pydantic import BaseModel, Field

from bidian.indexing.discovery import ParsedNote  # Assuming ParsedNote is in discovery

# Configure logger for this module
logger = logging.getLogger(__name__)


class TextChunk(BaseModel):
    """Represents a chunk of text derived from a source note."""
    source_path: Path = Field(...,
                              description="Path to the original source Markdown file.")
    chunk_index: int = Field(...,
                             description="The sequential index of this chunk within the source note.")
    content: str = Field(..., description="The text content of the chunk.")
    # Potential future additions: metadata like start/end position, associated heading


def chunk_by_paragraph(note: ParsedNote, min_chunk_length: int = 20) -> Generator[TextChunk, None, None]:
    """Splits the note content into chunks based on paragraph breaks (double newlines).

    Filters out chunks shorter than `min_chunk_length` after stripping whitespace.

    Args:
        note: The ParsedNote object containing the content to chunk.
        min_chunk_length: The minimum character length for a chunk to be yielded.

    Yields:
        TextChunk objects for each valid paragraph chunk.
    """
    logger.debug(f"Chunking content from: {note.file_path}")
    paragraphs = note.content.split('\n\n')
    chunk_idx = 0
    for paragraph in paragraphs:
        cleaned_paragraph = paragraph.strip()
        if len(cleaned_paragraph) >= min_chunk_length:
            chunk = TextChunk(
                source_path=note.file_path,
                chunk_index=chunk_idx,
                content=cleaned_paragraph
            )
            logger.debug(
                f"Yielding chunk {chunk_idx} from {note.file_path} (length {len(cleaned_paragraph)})")
            yield chunk
            chunk_idx += 1
        else:
            logger.debug(
                f"Skipping short paragraph (length {len(cleaned_paragraph)}) from {note.file_path}")

    logger.debug(f"Finished chunking {note.file_path}, yielded {chunk_idx} chunks.")


# Example Usage (for testing)
# if __name__ == '__main__':
#     from bidian.config.logging_config import setup_logging
#
#     setup_logging("DEBUG")
#
#     test_content = """
# First paragraph.
# This should be one chunk.
#
# Second paragraph, a bit longer.
# It has multiple lines.
#
#
# Third paragraph, separated by multiple newlines.
#
# Short.
#
# Final paragraph that meets the minimum length requirement.
# """
#
#     # Create a dummy ParsedNote
#     dummy_path = Path("/path/to/dummy/note.md")
#     dummy_note = ParsedNote(file_path=dummy_path, content=test_content)
#
#     print(f"Testing paragraph chunking for: {dummy_path}")
#     chunks = list(chunk_by_paragraph(dummy_note, min_chunk_length=20))
#     print(f"\nFound {len(chunks)} chunks:")
#     for i, chunk in enumerate(chunks):
#         print(f"--- Chunk {i} (Index {chunk.chunk_index}) ---") # noqa: T201
#         print(chunk.content) # noqa: T201
#         print("---") # noqa: T201
