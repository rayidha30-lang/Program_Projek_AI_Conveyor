import unittest

from run_multi_camera_benchmark import record_top_ng_if_enabled


class FakeTopInspector:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.rendered = 0
        self.recorded = []

    def is_db_enabled(self):
        return self.enabled

    def render_combined_dashboard(self, **kwargs):
        self.rendered += 1
        return "ng-annotation"

    def record_if_ng(self, **kwargs):
        self.recorded.append(kwargs)


class TopNgRecordingTests(unittest.TestCase):
    def test_ok_or_wait_never_renders_or_records_ng_annotation(self):
        """Moving record_if_ng outside its NG guard would make this test fail."""
        inspector = FakeTopInspector()
        for status in ("OK", "WAIT"):
            self.assertFalse(record_top_ng_if_enabled(
                inspector, "raw", {"status": status}, 20.0, 1, 0, 3, 4,
            ))
        self.assertEqual(inspector.rendered, 0)
        self.assertEqual(inspector.recorded, [])

    def test_ng_renders_and_records_one_matching_annotation(self):
        """Skipping NG recording or passing a different image would lose the evidence."""
        inspector = FakeTopInspector()
        self.assertTrue(record_top_ng_if_enabled(
            inspector, "raw", {"status": "NG"}, 20.0, 1, 0, 3, 4,
        ))
        self.assertEqual(inspector.rendered, 1)
        self.assertEqual(inspector.recorded[0]["frame_ann"], "ng-annotation")


if __name__ == "__main__":
    unittest.main()
