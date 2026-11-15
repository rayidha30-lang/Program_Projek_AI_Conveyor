# Rib Mask Contour Tester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-frame bending status derived from the contour of each YOLO segmentation mask in the isolated video tester.

**Architecture:** `MaskContourAnalyzer` will transform an individual binary mask into a local centreline, evaluate its normalized peak-to-peak deviation against a fitted line, and return an inspectable result. `VideoYoloTester` will call it per detected mask, draw its result, and expose its threshold through an OpenCV trackbar.

**Tech Stack:** Python, OpenCV, NumPy, Ultralytics YOLO segmentation.

**Spec:** `docs/superpowers/specs/2026-10-30-rib-mask-contour-tester-design.md`

## Global Constraints

- Modify only `Program/test_yolo_v26_video.py`.
- Keep `rib` as the only model class.
- Do not alter the production inspection pipeline.
- `WAIT` must be returned for masks that have insufficient valid contour samples.

## Review Focus

- Empty or absent `result.masks` must not crash video playback.
- A very narrow or fragmented mask must produce `WAIT`, not a false NG.
- Multiple rib masks in a frame must each receive an independent result.
- Contour calculations must use mask-local coordinates rather than fixed image positions.
- The existing detector-only HUD, video controls, FPS calculation, and snapshot operation must still work.

---

### Task 1: Add mask-contour analysis unit

**Files:**
- Modify: `Program/test_yolo_v26_video.py`
- Test: inline assert-based smoke checks callable with a new `--self-check` option

**Interfaces:**
- Produces: `MaskContourAnalyzer.analyze(mask: np.ndarray) -> dict` with `status`, `score`, `baseline`, and `centreline`.
- Consumes: a two-dimensional binary instance mask.

- [ ] **Step 1: Add failing self-checks for straight, curved, and undersampled synthetic masks**

Expected statuses: straight mask is `OK`, curved mask is `NG` at the chosen threshold, and a mask with insufficient columns is `WAIT`.

- [ ] **Step 2: Run the self-check to verify it fails before the analyzer exists**

Run: `python Program/test_yolo_v26_video.py --self-check`

Expected: failure because the analyzer is unavailable.

- [ ] **Step 3: Implement `MaskContourAnalyzer.analyze(mask: np.ndarray) -> dict`**

For each valid x-coordinate, locate the upper and lower occupied pixel then store their midpoint. Fit a degree-one line with NumPy, calculate peak-to-peak absolute residual normalized by the horizontal mask span, and assign `OK`, `NG`, or `WAIT` according to the spec.

- [ ] **Step 4: Run the self-check to verify all three expected outcomes**

Run: `python Program/test_yolo_v26_video.py --self-check`

Expected: exit code 0 and a concise success message.

### Task 2: Integrate and visualize results in the tester

**Files:**
- Modify: `Program/test_yolo_v26_video.py`

**Interfaces:**
- Consumes: `results[0].masks.data` from Ultralytics and `MaskContourAnalyzer.analyze`.
- Produces: a per-rib overlay containing the mask outline, local baseline, centreline, score, and status.

- [ ] **Step 1: Add a contour-threshold trackbar and pass its value to the analyzer**

The UI control uses a percentage representation of the normalized score and initializes from a named constant.

- [ ] **Step 2: Associate each mask with the corresponding YOLO box and draw the analysis overlay**

Render only for a segmentation result. Keep existing bounding-box display operational when no masks exist.

- [ ] **Step 3: Add `OK`, `NG`, and `WAIT` colors and show status plus score in the label**

Use green for `OK`, red for `NG`, and orange for `WAIT`.

- [ ] **Step 4: Run syntax validation and model smoke inference on frames 416, 707, and 1244**

Run: `python -m py_compile Program/test_yolo_v26_video.py`, then use the existing video-3 frame extraction procedure.

Expected: no exceptions; every detected mask receives a contour-analysis status.
