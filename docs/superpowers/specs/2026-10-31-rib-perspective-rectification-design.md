# Rib Perspective Rectification Design

## Intent
Replace the segmentation-mask CTR/TPL judgment path in `Program/test_yolo_v26_video.py` with the project's existing 13-keypoint YOLO-Pose model, then measure bending from its rectified keypoint profile.

## Goals
- Use `models/best_rib_pose.pt` to infer the existing 13 rib keypoints.
- Let the operator rectify the conveyor plane, then transform the keypoints before measuring bending.
- Use a robust Welsch-fit baseline and normalized P95/local-peak residual as the bending score.
- Use temporal voting inside a single-rib inspection zone.
- Remove segmentation-mask CTR and normal-template TPL from the tester's judgment path.

## Non-goals and limits
- Do not retrain or change the YOLO model in this change.
- This homography measures bending visible in the conveyor plane; it cannot measure rib height or out-of-plane deformation.
- Report geometric scores normalized by rib span; conversion to millimetres needs a known physical scale that is not currently configured.
- The transform is required for a bending judgment; without four-corner calibration, show WAIT.

## Approaches
- Segmentation mask plus contour/template scores: replaced because it overlaps the current method and produced persistent NG/WAIT behavior.
- YOLO-Pose keypoints plus robust profile metrology: selected because a trained 13-keypoint checkpoint and dataset already exist in the project.
- New learned OK/NG classifier: deferred because it needs a separate defect-labeled training dataset.

## Constraints and calls
- Reuse the existing four-click calibration interaction.
- Use the existing pose checkpoint; keep the original frame as model input and rectify its keypoints for measurement.
- Require enough confident keypoints and a sufficient rib span; otherwise return WAIT.
- Keep the bending threshold empirical until checked against labeled OK/NG samples.

## Builder discretion
Keep the tester's video controls and object/FPS reporting, while replacing only the bending detection and its relevant controls/labels.
