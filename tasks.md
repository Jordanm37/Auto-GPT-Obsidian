# Bidian Note Curator LLM Agent - Task Breakdown

This document outlines the tasks required to develop the Bidian Note Curator agent based on PRD v1.0.

**Legend:** `[x]` = Done, `[/]` = Partially Done / Basic Implementation, `[ ]` = Not Started

## Phase 1: Core Backlink Curation (Foundation)

- **[x] Setup Project Structure & Tooling:**
  - [x] Initialize Poetry project (`poetry init`).
  - [x] Configure `pyproject.toml` with dependencies.
  - [x] Set up basic ML project structure.
  - [x] Implement basic logging setup (`logging` module).
  - [x] Set up Git repository and initial commit.
  - [x] Create initial `README.md`.
- **[x] F-1: Indexing:**
  - [x] Implement Markdown file discovery and parsing.
  - [x] Implement text chunking strategy (paragraph-based).
  - [x] Implement embedding generation (support for local GGUF via llama-cpp-python).
  - [x] Implement vector store integration (ChromaDB).
  - [x] Implement incremental update logic (mtime-based).
- **[x] F-4: Similarity Search:**
  - [x] Implement function to compute top-k related notes (`Retriever` class).
  - [x] Add adjustable distance threshold (τ).
- **[/] F-5 & F-7: Backlink Insertion:**
  - [x] Implement logic to find insertion points (end of file / replace block).
  - [x] Implement idempotent backlink writing using HTML guard comments.
  - [/] Implement JSONL logging (`backlink_log.jsonl`) - _(Basic logging done, needs refinement for removed links)_.
- **[/] F-2: Command - Curate Current Note:**
  - [ ] Develop Obsidian command palette integration stub.
  - [x] Link command to trigger similarity search and backlink insertion (Backend API method `curate_backlinks_for_file` created).
- **[x] F-9: Respect Exclusions:**
  - [x] Implement logic to read and respect `.gitignore`.
  - [x] Implement logic to parse and respect `no-curate` front-matter tags.
- **[/] Testing & Documentation (Phase 1):**
  - [x] Write unit tests for parsing, chunking, embedding, indexing, similarity search, backlink writing, API, patching. _(Basic tests added)_.
  - [ ] Write Integration Tests (Indexing, Curation).
  - [ ] Achieve >80% Test Coverage.
  - [ ] Add/Complete Google-style Docstrings.
  - [ ] Update `README.md` (Usage, Configuration).

## Phase 2: Note Creation (Milestone T + 6 weeks)

- **[/] F-11 & F-14: Note Creation Core:**
  - [ ] Design and implement logic to detect note clusters lacking a central page.
  - [x] Design YAML schema for note creation templates.
  - [x] Implement template engine (`TemplateRenderer`).
  - [x] Ship default templates (summary, glossary, hub page).
  - [x] Allow loading custom templates.
  - [ ] Implement LLM call for Title/Outline generation in `propose_hub_note`.
- **[/] Interaction - Create Hub Note (Ctrl ⌥ N):**
  - [ ] Implement Obsidian modal dialog for previewing.
  - [x] Connect command to trigger note creation logic (Backend API method `propose_hub_note` created).
  - [x] Implement commit logic upon user approval (Backend API method `commit_note` created).
- **[x] F-13: Undo Patch System (Creation):**
  - [x] Implement diff generation (Metadata only for creation).
  - [x] Implement saving of patch file (`*.bidian-patch`) for created notes.
  - [x] Implement `rollback_patch` command logic for creation patches.
- **[/] Testing & Documentation (Phase 2):**
  - [/] Write unit tests (Templating, Patching done; Cluster detection pending).
  - [ ] Write integration tests for the note creation flow.
  - [ ] Document template format and creation command usage.

## Phase 3: Note Refactoring (Milestone T + 8 weeks)

- **[/] F-12: Refactoring Logic:**
  - [x] Implement logic to analyze Markdown structure (headings).
  - [/] Implement heading restructuring algorithm (Basic H1 demotion done).
  - [x] Implement front-matter update logic.
  - [/] Implement Table of Contents generation/update logic (Basic generation done, insertion needs improvement).
  - [ ] Implement robust/idempotent ToC insertion/update.
  - [ ] Implement advanced heading restructuring rules.
- **[/] Interaction - Refactor Current Note (Ctrl ⌥ R):**
  - [x] Implement diff generation highlighting structural changes (`propose_refactor` uses diff-match-patch).
  - [ ] Implement preview mechanism showing the diff (UI concern).
  - [x] Connect command to trigger refactoring logic (Backend `propose_refactor` created).
  - [x] Implement commit logic (Backend `commit_refactor` created).
- **[x] F-13: Undo Patch System (Refactoring):**
  - [x] Implement saving of patch file (storing original content) for refactored notes.
  - [x] Implement `rollback_patch` command logic for refactoring patches (restoring original content).
- **[/] Testing & Documentation (Phase 3):**
  - [/] Write unit tests (Analysis, FM update, basic ToC/restructure, patching done).
  - [ ] Write integration tests for the refactoring flow.
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
