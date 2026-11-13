import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from dataset_review_store import DatasetReviewStore
from conveyor_db_path import resolve_db_path


def sample_values():
    return {
        "rib_number": 7,
        "slat_number": 3,
        "label": "NG",
        "tracker_engine": "LEGACY",
        "pv_flatness": 12.5,
        "sagitta": 4.0,
        "inner_dev": 6.0,
        "tip_dev": 15.0,
        "coverage": 0.98,
        "active_frames": 20,
        "confidence": 0.88,
        "profile_json": "{\"profile\": \"test\"}",
        "camera_profile": "TOP_TEST",
    }


class DatasetReviewStoreTests(unittest.TestCase):
    def test_store_creates_evidence_and_review_tables_in_temp_database(self):
        """Removing either schema table must make this test fail."""
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "review.db")
            store = DatasetReviewStore(db_path)
            store.ensure_schema()

            conn = sqlite3.connect(db_path)
            try:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                conn.close()
            self.assertIn("Rib_Dataset", tables)
            self.assertIn("Rib_Dataset_Evidence", tables)
            self.assertIn("Rib_Dataset_Review", tables)

    def test_review_is_unique_per_dataset_id_and_accepts_only_three_manual_labels(self):
        """Allowing a second or unsupported judgment would corrupt ground truth."""
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            dataset_id = store.insert_sample(sample_values())

            self.assertTrue(store.save_review(dataset_id, "OK", "qa-01", "clear"))
            self.assertFalse(store.save_review(dataset_id, "NG", "qa-01", "duplicate"))
            with self.assertRaises(ValueError):
                store.save_review(dataset_id, "MAYBE", "qa-01", "unsupported")

    def test_db_bridge_prefers_conveyor_db_path_then_sqlite_fallback(self):
        """Changing environment precedence must fail this canonical-path contract."""
        with patch.dict(os.environ, {"CONVEYOR_DB_PATH": "C:/tmp/canonical.db", "SQLITE_DB_PATH": "C:/tmp/legacy.db"}, clear=False):
            self.assertEqual(resolve_db_path(), os.path.abspath("C:/tmp/canonical.db"))
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "C:/tmp/legacy.db"}, clear=True):
            self.assertEqual(resolve_db_path(), os.path.abspath("C:/tmp/legacy.db"))


if __name__ == "__main__":
    unittest.main()
