# Rib Dataset PC Review Implementation Plan

**Spec:** `docs/superpowers/specs/2026-11-13-rib-dataset-review-design.md`

## Global constraints

- Do not modify detector verdicts, calibration, models, or thresholds.
- Use the canonical database `Database/AI_Slat_Rib_Conveyor.db`.
- Tests use temporary databases and capture directories only.

## Tasks

1. Create the SQLite dataset store and canonical database resolution; test schema, manual-label validation, and environment precedence.
2. Create a one-rib best-evidence recorder and attach it only to the root runner; test short/WAIT transits, best-frame selection, and duplicate finalization.
3. Create the local OpenCV reviewer and operator guide; test that only rows with both evidence files are reviewable.
4. Create a read-only CSV evaluator; test false-NG, false-OK, and INVALID handling.

## Review focus

- Default writes must target `Database/AI_Slat_Rib_Conveyor.db`.
- A short or WAIT transit must not create a dataset sample.
- Repeated finalization must produce one row and one image pair.
- Missing evidence must never receive a manual label.
- Manual INVALID is excluded from the OK/NG confusion matrix but included in invalid rate.
