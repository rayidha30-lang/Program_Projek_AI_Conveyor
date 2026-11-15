import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


SRC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "Vision_Engine_Core",
        "02_Top_Inspection_Slat_Rib",
        "src",
    )
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import combined_top_inspector  # noqa: E402
import hybrid_rib_inspector  # noqa: E402
from calibrate_dual_rails import DualRailCalibrator  # noqa: E402
from conveyor_motion_calibrator import ConveyorMotionCalibrator  # noqa: E402
from run_multi_camera_benchmark import record_rib_dataset_step  # noqa: E402


def _make_synthetic_conveyor_frame(
    lt_x=310,
    lb_x=140,
    rt_x=1600,
    rb_x=1780,
    w=1920,
    h=1080,
    add_belt_distractors=True,
):
    """Build a synthetic 1080p top conveyor frame with inner belt gradient distractors."""
    frame = np.full((h, w, 3), 35, dtype=np.uint8)
    y_top = int(0.20 * h)
    y_bot = int(0.80 * h)

    # Outer bright guide rail walls tapering outward toward bottom (perspective trapezoid)
    def interp_x(x_t, x_b, y):
        return int(round(x_t + (y - y_top) * (x_b - x_t) / float(y_bot - y_top)))

    pts_poly = np.array(
        [
            [interp_x(lt_x, lb_x, 0), 0],
            [interp_x(rt_x, rb_x, 0), 0],
            [interp_x(rt_x, rb_x, h - 1), h - 1],
            [interp_x(lt_x, lb_x, h - 1), h - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts_poly], (110, 110, 110))
    cv2.line(frame, (interp_x(lt_x, lb_x, 0), 0), (interp_x(lt_x, lb_x, h - 1), h - 1), (235, 235, 235), 4)
    cv2.line(frame, (interp_x(rt_x, rb_x, 0), 0), (interp_x(rt_x, rb_x, h - 1), h - 1), (235, 235, 235), 4)

    if add_belt_distractors:
        # Strong vertical edges inside the lower half of the belt (bolts/ribs) that fooled argmax before
        cv2.rectangle(frame, (420, int(0.65 * h)), (455, int(0.85 * h)), (250, 250, 250), -1)
        cv2.rectangle(frame, (1330, int(0.65 * h)), (1365, int(0.85 * h)), (250, 250, 250), -1)
    return frame


class TopCalibrationTests(unittest.TestCase):
    def test_auto_snap_rails_preserves_trapezoid_perspective_despite_belt_distractors(self):
        frame = _make_synthetic_conveyor_frame(
            lt_x=310, lb_x=140, rt_x=1600, rb_x=1780, add_belt_distractors=True
        )
        calibrator = DualRailCalibrator()
        calibrator.auto_snap_rails(frame)

        lt_x, lb_x = calibrator.pts[0][0], calibrator.pts[1][0]
        rt_x, rb_x = calibrator.pts[2][0], calibrator.pts[3][0]

        self.assertLess(lb_x, lt_x)
        self.assertGreater(rb_x, rt_x)
        self.assertAlmostEqual(lt_x, 310.0, delta=30.0)
        self.assertAlmostEqual(lb_x, 140.0, delta=20.0)
        self.assertAlmostEqual(rt_x, 1600.0, delta=30.0)
        self.assertAlmostEqual(rb_x, 1780.0, delta=20.0)

    def test_compute_rectification_golden_trapezoid_is_identity_for_perspective_and_affine(self):
        calibrator = DualRailCalibrator()
        calibrator.pts = [
            np.array([278.0, 216.0]),
            np.array([104.0, 864.0]),
            np.array([1630.0, 216.0]),
            np.array([1808.0, 864.0]),
        ]

        calibrator.mode = "PERSPECTIVE"
        M_p, mtype_p = calibrator.compute_rectification(1920, 1080)
        self.assertEqual(mtype_p, "PERSPECTIVE")
        self.assertEqual(M_p.shape, (3, 3))
        self.assertTrue(np.allclose(M_p, np.eye(3), atol=1e-4))
        self.assertAlmostEqual(calibrator.metrology["scale_x"], 1.0, places=3)
        self.assertAlmostEqual(calibrator.metrology["lateral_offset_px"], 0.0, places=2)

        calibrator.mode = "AFFINE"
        M_a, mtype_a = calibrator.compute_rectification(1920, 1080)
        self.assertEqual(mtype_a, "AFFINE")
        self.assertEqual(M_a.shape, (2, 3))
        self.assertTrue(np.allclose(M_a, np.eye(2, 3), atol=1e-4))

    def test_combined_top_inspector_loads_calibrated_profile_when_default_init(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = os.path.join(directory, "top_inspection_config.json")
            default_cfg_path = os.path.join(directory, "top_inspection_config.default.json")
            with open(default_cfg_path, "w", encoding="utf-8") as f_def:
                json.dump({"profile_name": "DEFAULT_STABIL", "thresholds": {}}, f_def)
            expected_matrix = [
                [1.02, 0.0, -18.5],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            cfg = {
                "profile_name": "HASIL_KALIBRASI",
                "thresholds": {
                    "rib_pv_threshold": 16.5,
                    "rib_dev_threshold": 9.5,
                    "rib_tip_threshold": 9.5,
                    "rib_sagitta_threshold": 11.5,
                    "rib_prime_min_y": 290,
                    "rib_prime_max_y": 520,
                },
                "rectification_matrix": expected_matrix,
                "calibration_profile": {
                    "golden_reference": {
                        "target_tilt_deg": 0.9,
                        "target_cx": 960.0,
                        "left_rail_px": 188,
                        "right_rail_px": 1759,
                        "conveyor_width_px": 1571,
                        "top_y_px": 216,
                        "bot_y_px": 864,
                        "top_left_rail_px": 278,
                        "top_right_rail_px": 1630,
                        "bot_left_rail_px": 104,
                        "bot_right_rail_px": 1808,
                    }
                },
            }
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with patch.object(hybrid_rib_inspector, "HAS_YOLO", False):
                inspector = combined_top_inspector.CombinedTopInspector(
                    config_path=cfg_path,
                    rib_mode="DUAL_POLARITY_ONLY",
                    device="cpu",
                    enable_db=False,
                )

            self.assertEqual(inspector.active_profile_name, "HASIL_KALIBRASI")
            self.assertIsNotNone(inspector.rect_matrix)
            self.assertTrue(
                np.allclose(inspector.rect_matrix, np.array(expected_matrix, dtype=np.float32), atol=1e-5)
            )

    def test_motion_calibrator_uses_temporal_median_trapezoid_rectification(self):
        frame_base = _make_synthetic_conveyor_frame(
            lt_x=320, lb_x=150, rt_x=1610, rb_x=1790, add_belt_distractors=False
        )
        frames = []
        for idx in range(15):
            f = frame_base.copy()
            # Moving rib bar sliding upward so median suppression removes it completely
            y_bar = 250 + idx * 35
            if y_bar < 900:
                cv2.rectangle(f, (400, y_bar), (1500, y_bar + 25), (255, 255, 255), -1)
            frames.append(f)

        cmc = ConveyorMotionCalibrator()
        res = cmc.calibrate(frames)
        self.assertTrue(res["success"])
        M = np.array(res["rectification_matrix"], dtype=np.float32)
        self.assertEqual(M.shape, (3, 3))
        self.assertGreater(res["conveyor_width_px"], 1400)

    def test_record_rib_dataset_step_syncs_active_profile_name(self):
        class StubRecorder:
            def __init__(self):
                self.camera_profile = "DEFAULT_STABIL"
                self.observed = 0

            def should_observe(self, *_args):
                return True

            def observe(self, *_args):
                self.observed += 1

            def finalize(self, *_args):
                pass

            def start_new_session(self):
                pass

        class StubRibInspector:
            pv_thresh = 16.5
            sagitta_thresh = 11.5
            dev_thresh = 9.5
            tip_thresh = 9.5
            active_rib_frames = 18

        class StubSlatInspector:
            slat_count = 1

        class StubInspector:
            active_profile_name = "HASIL_KALIBRASI"
            rib_inspector = StubRibInspector()
            slat_inspector = StubSlatInspector()

            def is_db_enabled(self):
                return True

            def render_combined_dashboard(self, **_kwargs):
                return np.zeros((10, 10, 3), dtype=np.uint8)

        rec = StubRecorder()
        insp = StubInspector()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        inspect_res = {
            "rib_res": {
                "status": "OK",
                "rib_count": 1,
                "is_ng": False,
                "pv_flatness": 3.0,
                "sagitta": 1.0,
                "inner_dev": 2.0,
                "tip_dev": 2.0,
                "coverage": 0.9,
                "confidence": 0.9,
                "ys_contour": [10.0, 20.0],
            }
        }
        record_rib_dataset_step(rec, insp, frame, inspect_res, 30.0, 1, 100, 0, None)
        self.assertEqual(rec.camera_profile, "HASIL_KALIBRASI")
        self.assertEqual(rec.observed, 1)


if __name__ == "__main__":
    unittest.main()
