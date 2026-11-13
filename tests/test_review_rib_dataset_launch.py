import tempfile
import unittest
from pathlib import Path

from scripts.review_rib_dataset import default_database_path, resolve_review_config


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


if __name__ == "__main__":
    unittest.main()
