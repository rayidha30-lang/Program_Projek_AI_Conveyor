# Rib Dataset Evidence and PC Review Design

## Intent

Turn the existing `Rib_Dataset` telemetry into a reviewable dataset without changing the live rib decision. Each completed rib keeps its model metrics, one best raw frame, one AI-overlay frame, and a later manual QA judgment.

## Requirements

- The root project is the sole live implementation target and uses `Database/AI_Slat_Rib_Conveyor.db`.
- Existing `Rib_Dataset.Label` remains the model label (`OK` or `NG`).
- `Rib_Dataset_Evidence` stores one evidence pair per dataset row; `Rib_Dataset_Review` stores at most one manual label (`OK`, `NG`, or `INVALID`).
- Capture only the highest `(coverage, confidence)` frame for a finalized rib with at least 15 active frames.
- The PC reviewer must skip missing evidence. The evaluator reports quality without changing thresholds.
- No new dependencies; database access uses SQLite WAL with a 30-second busy timeout.
