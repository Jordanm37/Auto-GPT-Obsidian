# Bidian Note Curator LLM Agent - Task Breakdown

This document outlines the tasks required to develop the Bidian Note Curator agent based on PRD v1.0.

## Phase 1: Core Backlink Curation (Foundation)

- **[ ] Setup Project Structure & Tooling:**
  - [ ] Initialize Poetry project (`poetry init`).
  - [ ] Configure `pyproject.toml` with dependencies (e.g., `fastapi`, embedding library, vector store library, `pydantic`, `pytest`, `black`, `isort`, `flake8`).
  - [ ] Set up basic ML project structure (`src/`, `tests/`, `scripts/`, `config/`, `data/`).
  - [ ] Implement basic logging setup (`logging` module).
  - [ ] Set up Git repository and initial commit (`git init`, add `.gitignore`). Ref: `git-version-control.mdc`.
  - [ ] Create initial `README.md`.
- **[ ] F-1: Indexing:**
  - [ ] Implement Markdown file discovery and parsing within the Obsidian vault.
  - [ ] Implement text chunking strategy.
  - [ ] Implement embedding generation (support for local GGUF initially - F-8).
  - [ ] Implement vector store integration (e.g., FAISS, ChromaDB) for indexing and persistence.
  - [ ] Implement incremental update logic for the index.
- **[ ] F-4: Similarity Search:**
  - [ ] Implement function to compute top-k related notes based on vector similarity.
  - [ ] Add adjustable similarity threshold (τ).
- **[ ] F-5 & F-7: Backlink Insertion:**
  - [ ] Implement logic to find insertion points in Markdown files.
  - [ ] Implement idempotent backlink writing using HTML guard comments (`<!-- bidian-backlinks:start -->`).
  - [ ] Implement JSONL logging for all backlink changes (`backlink_log.jsonl`).
- **[ ] F-2: Command - Curate Current Note:**
  - [ ] Develop Obsidian command palette integration stub (requires knowledge of Obsidian plugin API).
  - [ ] Link command to trigger similarity search and backlink insertion for the active note.
- **[ ] F-9: Respect Exclusions:**
  - [ ] Implement logic to read and respect `.gitignore`.
  - [ ] Implement logic to parse and respect `no-curate` front-matter tags.
- **[ ] Testing & Documentation (Phase 1):**
  - [ ] Write unit tests for parsing, chunking, embedding, indexing, similarity search, and backlink writing.
  - [ ] Write integration tests for the core indexing and backlinking flow.
  - [ ] Add Google-style docstrings to all functions/classes.
  - [ ] Update `README.md` with setup and basic usage.

## Phase 2: Note Creation (Milestone T + 6 weeks)

- **[ ] F-11 & F-14: Note Creation Core:**
  - [ ] Design and implement logic to detect note clusters lacking a central page (requires clustering algorithm or heuristics on vector space).
  - [ ] Design YAML schema for note creation templates.
  - [ ] Implement template engine to render new notes from YAML templates.
  - [ ] Ship default templates (summary, glossary, hub page).
  - [ ] Allow loading custom templates from a dedicated folder.
- **[ ] Interaction - Create Hub Note (Ctrl ⌥ N):**
  - [ ] Implement Obsidian modal dialog for previewing proposed title, outline, and backlinks.
  - [ ] Connect command to trigger note creation logic and display preview.
  - [ ] Implement commit logic upon user approval.
- **[ ] F-13: Undo Patch System (Creation):**
  - [ ] Implement diff generation for new note creation (trivial diff: the whole file).
  - [ ] Implement saving of patch file (`*.bidian-patch`) for created notes.
  - [ ] Implement `rollback_patch` command logic for creation patches (delete file).
- **[ ] Testing & Documentation (Phase 2):**
  - [ ] Write unit tests for cluster detection, template rendering, patch generation.
  - [ ] Write integration tests for the note creation flow (command -> preview -> commit -> patch).
  - [ ] Document template format and creation command usage.

## Phase 3: Note Refactoring (Milestone T + 8 weeks)

- **[ ] F-12: Refactoring Logic:**
  - [ ] Implement logic to analyze Markdown structure (headings, paragraphs).
  - [ ] Implement heading restructuring algorithm (e.g., re-leveling, standardizing).
  - [ ] Implement front-matter update logic (add/update aliases, tags, status).
  - [ ] Implement Table of Contents generation/update logic.
- **[ ] Interaction - Refactor Current Note (Ctrl ⌥ R):**
  - [ ] Implement diff generation highlighting structural changes (added/removed/modified lines).
  - [ ] Implement preview mechanism showing the diff (potentially section-by-section).
  - [ ] Connect command to trigger refactoring logic and display preview.
  - [ ] Implement commit logic (full or per-section approval).
- **[ ] F-13: Undo Patch System (Refactoring):**
  - [ ] Implement saving of diff patch file (`*.bidian-patch`) for refactored notes.
  - [ ] Implement `rollback_patch` command logic for refactoring patches (apply reverse diff).
- **[ ] Testing & Documentation (Phase 3):**
  - [ ] Write unit tests for Markdown analysis, restructuring logic, ToC generation, front-matter updates, diff generation.
  - [ ] Write integration tests for the refactoring flow (command -> preview -> commit -> patch).
  - [ ] Document refactoring capabilities and command usage.

## Phase 4: Batch Operations & Polish (Milestone T + 10 weeks)

- **[ ] F-3: Batch Backlink Curation:**
  - [ ] Implement logic to iterate through the entire vault for backlink curation.
  - [ ] Integrate with logging (F-7) and exclusions (F-9).
- **[ ] Batch Refactor:**
  - [ ] Implement logic to iterate through the vault for refactoring.
  - [ ] Add settings UI integration for triggering batch refactor.
  - [ ] Implement cron-style scheduler for batch refactoring.
- **[ ] F-6: Review Queue Mode:**
  - [ ] Modify backlink curation (F-2, F-3) to optionally output proposed changes to a review file/UI instead of direct writes.
- **[ ] F-15: Dry-Run Mode:**
  - [ ] Implement dry-run flag for creation/refactoring commands to output diffs/previews without writing changes or patches.
- **[ ] Performance Optimization:**
  - [ ] Profile indexing and curation performance.
  - [ ] Optimize critical paths to meet ≤ 10 min / 10k notes goal.
- **[ ] F-8: Expanded LLM/Model Support:**
  - [ ] Add support for OpenAI API.
  - [ ] Add support for Anthropic API.
  - [ ] Refactor embedding/generation logic to be pluggable.
- **[ ] Testing & Documentation (Phase 4):**
  - [ ] Write tests for batch operations, scheduler, review queue, dry-run mode.
  - [ ] Test performance on sample large vaults.
  - [ ] Finalize `README.md` and add comprehensive documentation.
  - [ ] Add testing instructions to `README.md`.

## Phase 5: Optional Features & Future

- **[ ] F-10: API Endpoint:**
  - [ ] Design and implement a simple API endpoint (e.g., using FastAPI) for related-note queries.
- **[ ] Further LLM Integration:**
  - [ ] Explore using LLMs for more sophisticated refactoring or summary generation.
- **[ ] User Feedback & Iteration:**
  - [ ] Gather user feedback and plan v1.1.

---

_Assignees and deadlines to be added._
_Tasks reference Functional Requirement IDs (e.g., F-1) from PRD v1.0._
_Remember to commit regularly. Ref: `git-version-control.mdc`._
_Refer to `core.mdc` and `task.mdc` as needed._
