import os
import tempfile
import unittest

from run_multi_camera_benchmark import resolve_source_path


class SourceResolutionTests(unittest.TestCase):
    def test_skips_missing_candidate_and_uses_existing_media_file(self):
        """Returning the first missing candidate would open a blank benchmark window."""
        with tempfile.TemporaryDirectory() as directory:
            video_path = os.path.join(directory, "available.mp4")
            open(video_path, "wb").close()

            resolved = resolve_source_path("", [os.path.join(directory, "missing.mp4"), video_path])

            self.assertEqual(resolved, os.path.abspath(video_path))

    def test_returns_empty_when_no_candidate_source_is_usable(self):
        """Treating a missing path as usable would suppress the source-picker fallback."""
        with tempfile.TemporaryDirectory() as directory:
            resolved = resolve_source_path("", [os.path.join(directory, "missing.mp4")])

            self.assertEqual(resolved, "")


if __name__ == "__main__":
    unittest.main()
