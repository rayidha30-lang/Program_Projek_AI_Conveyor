# Rib Mask Contour Tester Design

## Intent

Add an experimental, per-frame rib-bending decision to `Program/test_yolo_v26_video.py` without changing the main inspection pipeline.

## Scope

- Input: each `rib` instance mask produced by `best_rib_segmentation.pt`.
- Output: an overlay with mask contour, fitted local reference line, normalized bending score, and `OK`, `NG`, or `WAIT` status.
- Interaction: one OpenCV trackbar controls the normalized bending threshold.
- Reference labels for tuning: frame 416 is NG; frames 707 and 1244 are OK.

## Method

For each binary instance mask, sample the top and bottom edge at valid horizontal positions and derive their midpoint. Fit a straight line to the midpoint samples. The bending score is the peak-to-peak residual of the midpoint from that line, divided by the mask width. A mask with too little horizontal coverage or too few samples is `WAIT`; otherwise it is `NG` only when its normalized score exceeds the current trackbar threshold.

## Constraints

- Modify only `Program/test_yolo_v26_video.py`.
- Do not modify the production inspection pipeline.
- Keep a single segmentation class: `rib`.
- This stage makes per-frame experimental decisions only; it does not add object tracking, multi-frame voting, or camera-pose correction.
