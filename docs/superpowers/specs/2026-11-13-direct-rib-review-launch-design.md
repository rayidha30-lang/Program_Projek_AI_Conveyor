# Direct Rib Review Launch Design

## Intent

Allow the local QA reviewer to run `scripts/review_rib_dataset.py` from an IDE Run button or by opening the Python file, without first typing command-line parameters.

## Behavior

- When `--db` is omitted, use `Database/AI_Slat_Rib_Conveyor.db` under the project root.
- When `--reviewer` is omitted, ask for the reviewer name in a small Windows popup. Cancelling the popup exits without changing data.
- After an OK, NG, or INVALID keypress, ask for the optional note in a popup rather than from terminal input.
- Existing explicit `--db` and `--reviewer` arguments remain supported for automated and terminal use.

## Constraints

- Retain the existing OpenCV evidence window and O/N/I/S/Q controls.
- Do not alter existing dataset, evidence, or review data.
- Use only Python standard-library UI for the new dialogs.
