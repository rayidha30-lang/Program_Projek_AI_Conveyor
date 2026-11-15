import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from dataset_review_store import DatasetReviewStore
from conveyor_db_path import resolve_db_path
from scripts.evaluate_rib_dataset import summarize_rows, write_report


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


def attach_reviewable_evidence(store, directory, dataset_id):
    raw_path = os.path.join(directory, f"{dataset_id}_raw.jpg")
    annotated_path = os.path.join(directory, f"{dataset_id}_annotated.jpg")
    open(raw_path, "wb").close()
    open(annotated_path, "wb").close()
    store.attach_evidence(dataset_id, {
        "session_id": "session",
        "raw_image_path": raw_path,
        "annotated_image_path": annotated_path,
    })


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
            attach_reviewable_evidence(store, directory, dataset_id)

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

    def test_next_unreviewed_returns_oldest_sample_with_both_evidence_paths(self):
        """Returning the newest sample or a reviewed row would break operator ordering."""
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            first_id = store.insert_sample(sample_values())
            second_id = store.insert_sample({**sample_values(), "rib_number": 8})
            for dataset_id in (first_id, second_id):
                raw_path = os.path.join(directory, f"{dataset_id}_raw.jpg")
                annotated_path = os.path.join(directory, f"{dataset_id}_annotated.jpg")
                open(raw_path, "wb").close()
                open(annotated_path, "wb").close()
                store.attach_evidence(dataset_id, {
                    "session_id": "session", "raw_image_path": raw_path,
                    "annotated_image_path": annotated_path,
                })
            self.assertTrue(store.save_review(first_id, "OK", "qa"))

            pending = store.next_unreviewed()
            self.assertEqual(pending["dataset_id"], second_id)

    def test_missing_evidence_sample_is_not_reviewable(self):
        """A missing image must never be offered for an unsupported manual label."""
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            dataset_id = store.insert_sample(sample_values())
            store.attach_evidence(dataset_id, {
                "session_id": "session",
                "raw_image_path": os.path.join(directory, "missing_raw.jpg"),
                "annotated_image_path": os.path.join(directory, "missing_annotated.jpg"),
            })
            self.assertIsNone(store.next_unreviewed())
            self.assertFalse(store.save_review(dataset_id, "INVALID", "qa"))

    def test_evaluation_counts_model_ng_manual_ok_as_false_ng(self):
        """Swapping model/manual labels must not hide a false NG from the report."""
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            dataset_id = store.insert_sample(sample_values())  # Model NG.
            attach_reviewable_evidence(store, directory, dataset_id)
            store.save_review(dataset_id, "OK", "qa")

            summary = summarize_rows(store.evaluation_rows())
            self.assertEqual(summary["false_ng"], 1)
            self.assertEqual(summary["false_ok"], 0)
            self.assertEqual(summary["comparable_total"], 1)

    def test_evaluation_excludes_manual_invalid_from_confusion_matrix(self):
        """Counting INVALID as OK or NG would falsify both error rates."""
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            dataset_id = store.insert_sample({**sample_values(), "label": "OK"})
            attach_reviewable_evidence(store, directory, dataset_id)
            store.save_review(dataset_id, "INVALID", "qa")

            summary = summarize_rows(store.evaluation_rows())
            self.assertEqual(summary["invalid"], 1)
            self.assertEqual(summary["comparable_total"], 0)
            self.assertEqual(summary["invalid_rate"], 1.0)

    def test_evaluation_never_updates_configuration_files(self):
        """Generating a report must only create CSV output, never alter a detector config."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "top_inspection_config.json")
            with open(config_path, "w", encoding="utf-8") as config:
                config.write('{"threshold": 12}')
            output_path = write_report(summarize_rows([]), os.path.join(directory, "reports"))

            self.assertTrue(os.path.isfile(output_path))
            with open(config_path, encoding="utf-8") as config:
                self.assertEqual(config.read(), '{"threshold": 12}')

    def test_next_unreviewed_skips_excluded_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            first_id = store.insert_sample(sample_values())
            second_id = store.insert_sample({**sample_values(), "rib_number": 8})
            attach_reviewable_evidence(store, directory, first_id)
            attach_reviewable_evidence(store, directory, second_id)

            pending = store.next_unreviewed(exclude_ids={first_id})
            self.assertIsNotNone(pending)
            self.assertEqual(pending["dataset_id"], second_id)

    def test_save_review_allow_update_updates_existing_review(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            dataset_id = store.insert_sample(sample_values())
            attach_reviewable_evidence(store, directory, dataset_id)

            self.assertTrue(store.save_review(dataset_id, "OK", "qa-01", "awal"))
            self.assertTrue(store.save_review(dataset_id, "NG", "qa-01", "koreksi", allow_update=True))
            fetched = store.fetch_sample(dataset_id)
            self.assertEqual(fetched["manual_label"], "NG")
            self.assertEqual(fetched["note"], "koreksi")

    def test_evaluation_computes_distributions_and_threshold_recommendations(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            ok_id = store.insert_sample({
                **sample_values(),
                "label": "NG",
                "pv_flatness": 6.0,
                "sagitta": 3.0,
                "inner_dev": 4.0,
                "tip_dev": 5.0,
            })
            ng_id = store.insert_sample({
                **sample_values(),
                "rib_number": 8,
                "label": "NG",
                "pv_flatness": 14.0,
                "sagitta": 9.0,
                "inner_dev": 10.0,
                "tip_dev": 13.0,
            })
            attach_reviewable_evidence(store, directory, ok_id)
            attach_reviewable_evidence(store, directory, ng_id)
            store.save_review(ok_id, "OK", "qa")
            store.save_review(ng_id, "NG", "qa")

            summary = summarize_rows(store.evaluation_rows())
            recs = summary["threshold_recommendations"]
            self.assertEqual(recs["rib_pv_threshold"]["status"], "clean_separation")
            self.assertAlmostEqual(recs["rib_pv_threshold"]["value"], 10.0)
            self.assertAlmostEqual(recs["rib_sagitta_threshold"]["value"], 6.0)
            report_path = write_report(summary, os.path.join(directory, "reports"))
            with open(report_path, encoding="utf-8") as report:
                content = report.read()
            self.assertIn("recommended:rib_pv_threshold,10.0", content)


if __name__ == "__main__":
    unittest.main()
