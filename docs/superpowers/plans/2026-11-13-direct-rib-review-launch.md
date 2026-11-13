# Direct Rib Review Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit the rib reviewer to start directly without command-line parameters while preserving traceable manual judgments.

**Architecture:** `review_rib_dataset.py` resolves omitted arguments into a project-local database and obtains missing human text through `tkinter.simpledialog`. The review loop continues to use `DatasetReviewStore` unchanged.

**Tech Stack:** Python 3, `argparse`, standard-library `tkinter`, OpenCV, SQLite.

**Spec:** `docs/superpowers/specs/2026-11-13-direct-rib-review-launch-design.md`

## Global Constraints

- Explicit `--db` and `--reviewer` values remain backward compatible.
- A cancelled reviewer dialog must write no review.
- No dataset schema or inspection pipeline changes.

## Review Focus

- Omitted database argument resolves to the database in the project `Database` folder.
- Explicit database and reviewer arguments bypass defaults and prompts.
- An empty/cancelled reviewer value stops before an OpenCV review window opens.
- Optional notes may be empty and must not block a keyboard judgment.

---

### Task 1: Resolve direct-launch configuration and popup input

**Files:**
- Modify: `scripts/review_rib_dataset.py`
- Create: `tests/test_review_rib_dataset_launch.py`

**Interfaces:**
- Produces: `default_database_path() -> Path`
- Produces: `resolve_review_config(db_argument, reviewer_argument, prompt_reviewer) -> tuple[Path, str]`

- [ ] **Step 1: Write failing tests**

Assert an omitted database resolves to `PROJECT_ROOT / "Database" / "AI_Slat_Rib_Conveyor.db"`, an explicit database path is retained, and an omitted reviewer uses the supplied prompt function.

- [ ] **Step 2: Run the launch tests to verify failure**

Run: `python -m unittest tests.test_review_rib_dataset_launch -v`

Expected: FAIL because the configuration helpers do not exist.

- [ ] **Step 3: Implement direct-launch configuration and dialogs**

Make both CLI arguments optional. Use `tkinter.simpledialog` only when a value is absent; exit with status 0 when the reviewer name is empty or cancelled. Use the same dialog helper for optional notes.

- [ ] **Step 4: Run tests to verify success**

Run: `python -m unittest tests.test_review_rib_dataset_launch -v`

Expected: PASS.

- [ ] **Step 5: Run full regression suite and commit**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.
