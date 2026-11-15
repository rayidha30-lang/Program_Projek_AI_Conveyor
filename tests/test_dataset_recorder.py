import os
import sqlite3
import tempfile
import unittest

import cv2
import numpy as np

from dataset_recorder import RibDatasetRecorder
from dataset_review_store import DatasetReviewStore


def rib_result(status="OK", coverage=0.9, confidence=0.8, active_frames=18):
    return {
        "status": status,
        "is_ng": status == "NG",
        "pv_flatness": 4.0,
        "sagitta": 2.0,
        "inner_dev": 3.0,
        "tip_dev": 5.0,
        "coverage": coverage,
        "confidence": confidence,
        "active_rib_frames": active_frames,
        "tracker_engine": "LEGACY",
        "ys_contour": [1.0, 2.0, 3.0],
    }


class RibDatasetRecorderTests(unittest.TestCase):
    def make_recorder(self, directory):
        store = DatasetReviewStore(os.path.join(directory, "review.db"))
        recorder = RibDatasetRecorder(store, os.path.join(directory, "captures"), session_id="test-session")
        return store, recorder

    def test_recorder_uses_highest_coverage_then_confidence_frame(self):
        """Choosing an earlier, lower-quality frame must fail this evidence test."""
        with tempfile.TemporaryDirectory() as directory:
            store, recorder = self.make_recorder(directory)
            first = np.full((12, 12, 3), (10, 20, 30), dtype=np.uint8)
            best = np.full((12, 12, 3), (40, 50, 60), dtype=np.uint8)
            recorder.observe(rib_result(coverage=0.80, confidence=0.99), first, first, 2, 4)
            recorder.observe(rib_result(coverage=0.95, confidence=0.10), best, best, 2, 4)

            dataset_id = recorder.finalize(rib_result(coverage=0.95, confidence=0.10), 2, 4)
            self.assertIsNotNone(dataset_id)
            evidence = store.evidence_for(int(dataset_id))
            written = cv2.imread(evidence["raw_image_path"])
            self.assertEqual(written[0, 0].tolist(), [40, 50, 60])

    def test_recorder_initialization_creates_review_schema_before_first_rib(self):
        """A new run must expose its tables even before a rib reaches finalization."""
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "review.db")
            store = DatasetReviewStore(db_path)
            RibDatasetRecorder(store, os.path.join(directory, "captures"))

            conn = sqlite3.connect(db_path)
            try:
                tables = {
                    row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                conn.close()
            self.assertIn("Rib_Dataset", tables)
            self.assertIn("Rib_Dataset_Evidence", tables)

    def test_recorder_skips_wait_and_short_transits(self):
        """Recording WAIT or fewer than 15 active frames would pollute the dataset."""
        with tempfile.TemporaryDirectory() as directory:
            store, recorder = self.make_recorder(directory)
            frame = np.zeros((12, 12, 3), dtype=np.uint8)
            recorder.observe(rib_result(status="WAIT"), frame, frame, 1, 1)
            self.assertIsNone(recorder.finalize(rib_result(status="WAIT"), 1, 1))
            recorder.observe(rib_result(active_frames=14), frame, frame, 2, 1)
            self.assertIsNone(recorder.finalize(rib_result(active_frames=14), 2, 1))
            self.assertEqual(store.sample_count(), 0)

    def test_recorder_persists_one_sample_for_repeated_finalize_calls(self):
        """Removing finalized-rib deduplication would create duplicate telemetry and image pairs."""
        with tempfile.TemporaryDirectory() as directory:
            store, recorder = self.make_recorder(directory)
            frame = np.full((12, 12, 3), 80, dtype=np.uint8)
            result = rib_result(status="NG")
            recorder.observe(result, frame, frame, 9, 2)
            first_id = recorder.finalize(result, 9, 2)
            second_id = recorder.finalize(result, 9, 2)

            self.assertIsNotNone(first_id)
            self.assertEqual(second_id, first_id)
            self.assertEqual(store.sample_count(), 1)
            self.assertEqual(store.evidence_count(), 1)

    def test_new_session_allows_reused_counter_without_reusing_old_sample(self):
        """A counter reset for a new video must not suppress its first physical rib."""
        with tempfile.TemporaryDirectory() as directory:
            store, recorder = self.make_recorder(directory)
            frame = np.full((12, 12, 3), 90, dtype=np.uint8)
            result = rib_result()
            recorder.observe(result, frame, frame, 1, 1)
            first_id = recorder.finalize(result, 1, 1)

            recorder.start_new_session("next-video")
            recorder.observe(result, frame, frame, 1, 1)
            second_id = recorder.finalize(result, 1, 1)

            self.assertNotEqual(first_id, second_id)
            self.assertEqual(store.sample_count(), 2)

    def test_should_observe_only_accepts_strictly_better_eligible_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            _, recorder = self.make_recorder(directory)
            frame = np.full((12, 12, 3), 50, dtype=np.uint8)
            self.assertFalse(recorder.should_observe(rib_result(status="WAIT"), 1))
            self.assertFalse(recorder.should_observe(rib_result(active_frames=14), 1))
            self.assertTrue(recorder.should_observe(rib_result(coverage=0.80, confidence=0.50), 1))
            recorder.observe(rib_result(coverage=0.80, confidence=0.50), frame, frame, 1, 1)
            self.assertFalse(recorder.should_observe(rib_result(coverage=0.80, confidence=0.50), 1))
            self.assertTrue(recorder.should_observe(rib_result(coverage=0.90, confidence=0.10), 1))

    def test_recorder_survives_slat_number_change_mid_transit(self):
        with tempfile.TemporaryDirectory() as directory:
            store, recorder = self.make_recorder(directory)
            best = np.full((12, 12, 3), (40, 50, 60), dtype=np.uint8)
            trailing = np.full((12, 12, 3), (10, 10, 10), dtype=np.uint8)
            recorder.observe(rib_result(coverage=0.95, confidence=0.90), best, best, 5, 1)
            recorder.observe(rib_result(coverage=0.70, confidence=0.50), trailing, trailing, 5, 2)

            dataset_id = recorder.finalize(rib_result(coverage=0.70, confidence=0.50), 5, 2)
            self.assertIsNotNone(dataset_id)
            evidence = store.evidence_for(int(dataset_id))
            written = cv2.imread(evidence["raw_image_path"])
            self.assertEqual(written[0, 0].tolist(), [40, 50, 60])
            self.assertEqual(store.fetch_sample(int(dataset_id))["slat_number"], 1)

    def test_finalize_preserves_best_frame_geometry_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            store, recorder = self.make_recorder(directory)
            frame = np.full((12, 12, 3), 70, dtype=np.uint8)
            best_res = rib_result(status="OK", coverage=0.95, confidence=0.90, active_frames=18)
            best_res["pv_flatness"] = 4.25
            best_res["sagitta"] = 1.75
            recorder.observe(best_res, frame, frame, 3, 1)

            end_res = rib_result(status="NG", coverage=0.60, confidence=0.40, active_frames=26)
            end_res["pv_flatness"] = 18.5
            end_res["sagitta"] = 9.9
            dataset_id = recorder.finalize(end_res, 3, 1)
            self.assertIsNotNone(dataset_id)

            sample = store.fetch_sample(int(dataset_id))
            self.assertEqual(sample["model_label"], "NG")
            self.assertEqual(sample["active_frames"], 26)
            self.assertAlmostEqual(sample["pv_flatness"], 4.25)
            self.assertAlmostEqual(sample["sagitta"], 1.75)

    def test_async_write_persists_evidence_after_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            recorder = RibDatasetRecorder(
                store,
                os.path.join(directory, "captures"),
                session_id="async-session",
                async_write=True,
            )
            frame = np.full((12, 12, 3), 120, dtype=np.uint8)
            res = rib_result(status="OK")
            recorder.observe(res, frame, frame, 4, 2)
            recorder.finalize(res, 4, 2)
            recorder.flush()
            recorder.close()
            self.assertEqual(store.sample_count(), 1)
            self.assertEqual(store.evidence_count(), 1)

    def test_recorder_ignores_transit_hysteresis_frames_without_box_or_contour(self):
        """Empty hysteresis buffer frames with coverage=1.0 must never overwrite real rib frames."""
        with tempfile.TemporaryDirectory() as directory:
            store, recorder = self.make_recorder(directory)
            real_frame = np.full((12, 12, 3), (40, 50, 60), dtype=np.uint8)
            hysteresis_frame = np.full((12, 12, 3), (10, 10, 10), dtype=np.uint8)

            real_res = rib_result(status="OK", coverage=0.92, confidence=0.85)
            real_res["active_box"] = (10, 10, 100, 40)
            recorder.observe(real_res, real_frame, real_frame, 6, 2)

            hyst_res = rib_result(status="OK", coverage=1.0, confidence=0.99)
            hyst_res["active_box"] = None
            hyst_res["detection_reason"] = "Transit Hysteresis"
            hyst_res.pop("ys_contour", None)
            self.assertFalse(recorder.should_observe(hyst_res, 6))
            recorder.observe(hyst_res, hysteresis_frame, hysteresis_frame, 6, 2)

            dataset_id = recorder.finalize(hyst_res, 6, 2)
            self.assertIsNotNone(dataset_id)
            evidence = store.evidence_for(int(dataset_id))
            written = cv2.imread(evidence["raw_image_path"])
            self.assertEqual(written[0, 0].tolist(), [40, 50, 60])


if __name__ == "__main__":
    unittest.main()
