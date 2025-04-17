# Bidian Note Curator LLM Agent (Auto-GPT-Obsidian Plugin)

A plugin for Auto-GPT that allows interaction with Obsidian, enhanced with LLM-powered note curation, creation, and refactoring capabilities.

This project provides an autonomous agent designed to work with your Obsidian vault to:

- Surface hidden relationships between notes using semantic similarity and add backlinks.
- Generate new notes (summaries, hub pages) based on identified knowledge gaps or user requests.
- Refactor existing notes to improve structure, metadata, and readability.

## Features (Based on PRD v1.0)

- **Backlink Curation:** Indexes your vault, finds related notes, and adds backlinks idempotently.
- **Note Creation:** Generates new notes (e.g., hub pages for clusters of related notes) using configurable templates.
- **Note Refactoring:** Restructures headings, updates front-matter, and adds tables of contents.
- **Local-First:** Designed to run offline with support for local GGUF models (as well as OpenAI/Anthropic).
- **Safety:** Includes features like dry-run mode, review queues, and undo patches.
- **Logging:** Tracks backlink changes.
- **Customization:** Respects `.gitignore` and `no-curate` front-matter; uses YAML templates for note creation.

## Installation & Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/conneroisu/Auto-GPT-Obsidian.git # Replace with your repo URL if different
    cd Auto-GPT-Obsidian
    ```
2.  **Install dependencies:**
    ```bash
    poetry install
    ```
3.  **Configuration:** _(Details TBD - Likely involves setting vault path, model choice, etc. in a `.env` or config file)_

## Usage

_(Details TBD - Outline commands, API usage, etc.)_

- **Curate Backlinks (Current Note):** _(Obsidian command TBD)_
- **Curate Backlinks (Vault):** _(Command/Setting TBD)_
- **Create Hub Note:** `Ctrl + Option + N` (or command)
- **Refactor Current Note:** `Ctrl + Option + R` (or command)

## Configuration

_(Details TBD - How to configure models, templates, thresholds, scheduling, etc.)_

## Testing

This project uses `pytest`.

1.  **Run tests:**
    ```bash
    poetry run pytest
    ```

_(More details on test coverage and types TBD)_
