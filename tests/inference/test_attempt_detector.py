"""Deterministic camera-free checks for hands-free attempt boundaries."""

from __future__ import annotations

import unittest

import numpy as np

from rehab_mtl.inference.attempt_detector import AutoAttemptDetector


def pose(wrist_lift: float = 0.0) -> np.ndarray:
    frame = np.zeros((25, 3), dtype=np.float32)
    frame[20] = [0.0, 0.5, 0.0]
    frame[4] = [-0.2, 0.5, 0.0]
    frame[5] = [-0.3, 0.3, 0.0]
    frame[6] = [-0.3, 0.1 + wrist_lift, 0.0]
    frame[8] = [0.2, 0.5, 0.0]
    frame[9] = [0.3, 0.3, 0.0]
    frame[10] = [0.3, 0.1 + wrist_lift, 0.0]
    return frame


class TestAutoAttemptDetector(unittest.TestCase):
    def test_starts_on_motion_and_ends_after_return_hold(self):
        detector = AutoAttemptDetector()
        events = []
        for frame_index in range(180):
            t = frame_index / 30
            frame = pose(0.4 if 1.4 <= t < 4.0 else 0.0)
            event = detector.update(t, frame)
            if event:
                events.append((event, t))
                if event == "start":
                    pre_roll = detector.consume_pre_roll()
                    self.assertLess(pre_roll[0][0], t)
                    self.assertEqual(pre_roll[-1][0], t)
        self.assertEqual([event for event, _ in events], ["start", "end"])
        self.assertGreaterEqual(events[1][1], 4.8)

    def test_short_jitter_does_not_start_attempt(self):
        detector = AutoAttemptDetector()
        events = []
        for frame_index in range(110):
            t = frame_index / 30
            frame = pose(0.4 if 1.5 <= t < 1.6 else 0.0)
            events.append(detector.update(t, frame))
        self.assertNotIn("start", events)

    def test_missing_tracking_clears_readiness(self):
        detector = AutoAttemptDetector()
        for frame_index in range(36):
            detector.update(frame_index / 30, pose())
        self.assertEqual(detector.state, "armed")
        detector.update(1.2, None)
        self.assertEqual(detector.state, "ready")

    def test_no_return_times_out_without_scoring_partial_attempt(self):
        detector = AutoAttemptDetector(max_attempt_seconds=2.0)
        events = []
        for frame_index in range(160):
            t = frame_index / 30
            event = detector.update(t, pose(0.4 if t >= 1.4 else 0.0))
            if event:
                events.append(event)
        self.assertEqual(events, ["start", "timeout"])


if __name__ == "__main__":
    unittest.main()
