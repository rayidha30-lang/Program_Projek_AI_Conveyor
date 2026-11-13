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


if __name__ == "__main__":
    unittest.main()
