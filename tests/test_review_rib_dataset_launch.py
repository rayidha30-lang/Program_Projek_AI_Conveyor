import os
import tempfile
import unittest
from pathlib import Path

from dataset_review_store import DatasetReviewStore
from scripts.review_rib_dataset import ReviewSession, default_database_path, resolve_review_config


class DirectReviewLaunchTests(unittest.TestCase):
    def test_omitted_database_uses_project_database_and_prompts_for_reviewer(self):
        """A Run-button launch must not require command-line arguments."""
        prompts = []

        database_path, reviewer = resolve_review_config(
            None,
            None,
            lambda: prompts.append(True) or "  Rayidha  ",
        )

        self.assertEqual(database_path, default_database_path())
        self.assertEqual(reviewer, "Rayidha")
        self.assertEqual(prompts, [True])

    def test_explicit_values_bypass_the_reviewer_popup(self):
        """Existing terminal automation must retain its explicit values."""
        with tempfile.TemporaryDirectory() as directory:
            expected_database = Path(directory) / "review.db"

            database_path, reviewer = resolve_review_config(
                str(expected_database),
                "QA-02",
                lambda: self.fail("popup must not be requested"),
            )

        self.assertEqual(database_path, expected_database.resolve())
        self.assertEqual(reviewer, "QA-02")

    def test_cancelled_reviewer_returns_an_empty_reviewer(self):
        """A cancelled popup must prevent an untraceable judgment session."""
        _, reviewer = resolve_review_config(None, None, lambda: None)

        self.assertEqual(reviewer, "")

    def test_review_session_skip_undo_and_pending_note(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetReviewStore(os.path.join(directory, "review.db"))
            first_id = store.insert_sample({"rib_number": 1, "slat_number": 1, "label": "OK"})
            second_id = store.insert_sample({"rib_number": 2, "slat_number": 1, "label": "NG"})
            for dataset_id in (first_id, second_id):
                raw_path = os.path.join(directory, f"{dataset_id}_raw.jpg")
                ann_path = os.path.join(directory, f"{dataset_id}_ann.jpg")
                open(raw_path, "wb").close()
                open(ann_path, "wb").close()
                store.attach_evidence(dataset_id, {
                    "session_id": "s1",
                    "raw_image_path": raw_path,
                    "annotated_image_path": ann_path,
                })

            session = ReviewSession(store, "QA-01")
            self.assertEqual(session.current_sample()["dataset_id"], first_id)

            session.skip(first_id)
            self.assertEqual(session.current_sample()["dataset_id"], second_id)

            session.pending_note = "ujung bengkok"
            self.assertTrue(session.record_label(second_id, "NG"))
            self.assertEqual(session.pending_note, "")

            self.assertTrue(session.undo())
            reopened = session.current_sample()
            self.assertEqual(reopened["dataset_id"], second_id)
            self.assertEqual(reopened["manual_label"], "NG")
            self.assertTrue(session.record_label(second_id, "OK"))
            self.assertEqual(store.fetch_sample(second_id)["manual_label"], "OK")
            self.assertEqual(store.fetch_sample(second_id)["note"], "ujung bengkok")


if __name__ == "__main__":
    unittest.main()
