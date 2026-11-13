import unittest

from run_multi_camera_benchmark import rib_result_for_dataset


class FakeRibInspector:
    def __init__(self, active_rib_frames):
        self.active_rib_frames = active_rib_frames


class RibResultForDatasetTests(unittest.TestCase):
    def test_adds_live_active_frame_count_missing_from_pipeline_result(self):
        """Dropping this runtime-only counter would make the recorder discard valid ribs."""
        original = {"status": "OK", "rib_count": 5}
        result = rib_result_for_dataset(original, FakeRibInspector(24))

        self.assertEqual(result["active_rib_frames"], 24)
        self.assertNotIn("active_rib_frames", original)

    def test_preserves_explicit_active_frame_count_from_pipeline(self):
        """A future pipeline-provided final count must take priority over the live counter."""
        result = rib_result_for_dataset(
            {"status": "NG", "active_rib_frames": 31}, FakeRibInspector(24)
        )
        self.assertEqual(result["active_rib_frames"], 31)


if __name__ == "__main__":
    unittest.main()
