# Rib Perspective Rectification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mask-based CTR/TPL bending judgment with rectified YOLO-Pose keypoint metrology.

**Architecture:** Use the existing 13-keypoint `best_rib_pose.pt` model on the original or currently stabilized frame. Transform confident keypoints into the calibrated conveyor plane, then score their robust fitted profile and apply temporal voting in the selected single-rib zone.

**Tech Stack:** Python, OpenCV, NumPy, Ultralytics YOLO segmentation.

**Spec:** `docs/superpowers/specs/2026-10-31-rib-perspective-rectification-design.md`

## Global Constraints
- YOLO-Pose continues to infer on the original frame.
- Do not use segmentation masks or the old CTR/TPL judgments for the bending verdict.
- Without valid perspective calibration or enough keypoints, withhold the verdict as WAIT.
- Homography only measures bending visible in the conveyor plane.
- NG threshold remains empirical and requires labeled OK/NG samples for calibration.

## Review Focus
- Calibration is missing or incomplete: use the current unrectified path.
- Selected corners are malformed or degenerate: reject calibration and withhold rectified judgment.
- Rib mask touches or lies outside the selected plane: clip safely and avoid invalid contour measurements.
- Inspection zone and template coordinates remain consistent with the chosen measurement plane.
- Rectified metrics and overlays stay aligned to the displayed frame.

### Task 1: Replace mask bending judgment with pose metrology

**Files:**
- Modify: `Program/test_yolo_v26_video.py`

**Interfaces:**
- Reuse `PerspectiveStabilizer`'s four-click interaction only if it can represent the conveyor-plane quadrilateral without conflating frame stabilization; otherwise add a distinct four-point rectification state.
- Produce a source-to-canonical homography and its inverse for mask analysis and display overlays.

- [ ] Load the existing `best_rib_pose.pt` checkpoint by default and consume its 13 keypoints instead of segmentation masks.
- [ ] Transform confident keypoints with the four-corner homography and return WAIT when calibration/point coverage is invalid.
- [ ] Calculate a Welsch-fit baseline, normalized P95 and local peak residual, and temporal vote inside the one-rib zone.
- [ ] Remove CTR/TPL from the judgment controls and display; show only the new pose-based result and its component scores.
- [ ] Draw the pose profile and fit baseline back on the source frame; retain video and FPS controls.
- [ ] Run a Python syntax compilation check and inspect the final diff; do not run project tests unless requested.
