import unittest

from run_multi_camera_benchmark import (
    finalize_pending_rib_dataset,
    record_rib_dataset_step,
    rib_result_for_dataset,
)


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

    def test_finalizes_last_pending_rib_when_run_stops(self):
        """Dropping the final transit at shutdown would permanently lose the last rib."""
        calls = []

        class Recorder:
            def finalize(self, result, rib_number, slat_number):
                calls.append((result, rib_number, slat_number))
                return 12

        result = finalize_pending_rib_dataset(
            Recorder(), ({"status": "NG", "active_rib_frames": 18}, 29, 118)
        )

        self.assertEqual(result, 12)
        self.assertEqual(calls, [({"status": "NG", "active_rib_frames": 18}, 29, 118)])

    def test_record_rib_dataset_step_renders_dashboard_only_when_should_observe_is_true(self):
        class FakeRecorder:
            def __init__(self):
                self.observed = 0

            def should_observe(self, result, rib_no):
                return float(result.get("coverage", 0.0)) >= 0.9

            def observe(self, result, frame, annotated, rib_no, slat_no):
                self.observed += 1

            def finalize(self, result, rib_no, slat_no):
                return None

            def start_new_session(self):
                pass

        class FakeInspector:
            def __init__(self):
                self.rib_inspector = FakeRibInspector(20)
                self.slat_inspector = type("Slat", (), {"slat_count": 3})()
                self.rendered = 0

            def is_db_enabled(self):
                return True

            def render_combined_dashboard(self, **kwargs):
                self.rendered += 1
                return "ann"

        recorder = FakeRecorder()
        inspector = FakeInspector()
        last_no, prev = record_rib_dataset_step(
            recorder, inspector, "raw", {"rib_res": {"status": "OK", "rib_count": 1, "coverage": 0.5}},
            30.0, 1, 100, 0, None,
        )
        self.assertEqual(inspector.rendered, 0)
        self.assertEqual(recorder.observed, 0)

        last_no, prev = record_rib_dataset_step(
            recorder, inspector, "raw", {"rib_res": {"status": "OK", "rib_count": 1, "coverage": 0.95}},
            30.0, 2, 100, last_no, prev,
        )
        self.assertEqual(inspector.rendered, 1)
        self.assertEqual(recorder.observed, 1)


if __name__ == "__main__":
    unittest.main()
