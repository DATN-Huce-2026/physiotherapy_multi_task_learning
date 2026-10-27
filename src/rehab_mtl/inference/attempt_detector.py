"""Hands-free boundaries for one movement that returns to its starting pose.

This is a camera heuristic, separate from the trained exercise/error model. It
uses relative 3D joint positions so walking sideways does not start an attempt.
"""

from __future__ import annotations

from collections import deque

import numpy as np

MOTION_JOINTS = (4, 5, 6, 8, 9, 10, 20)  # shoulders, elbows, wrists, upper spine


def motion_signature(frame: np.ndarray) -> np.ndarray:
    """Upper-body joint positions relative to the pelvis and torso length."""
    xyz = np.asarray(frame, dtype=np.float32)
    if xyz.shape != (25, 3) or not np.isfinite(xyz).all():
        raise ValueError("Cần frame Kinect [25, 3] hữu hạn")
    torso_length = max(float(np.linalg.norm(xyz[20] - xyz[0])), 0.1)
    return (xyz[list(MOTION_JOINTS)] - xyz[0]) / torso_length


def pose_distance(current: np.ndarray, baseline: np.ndarray) -> float:
    """Mean displacement of the two most changed upper-body joints."""
    distances = np.linalg.norm(current - baseline, axis=1)
    return float(np.mean(np.sort(distances)[-2:]))


class AutoAttemptDetector:
    """Arm after a stable pose; capture motion until that pose is held again."""

    def __init__(
        self,
        *,
        ready_seconds: float = 1.0,
        start_threshold: float = 0.25,
        return_threshold: float = 0.12,
        start_hold_seconds: float = 0.15,
        return_hold_seconds: float = 0.8,
        min_attempt_seconds: float = 3.0,
        max_attempt_seconds: float = 60.0,
        pre_roll_seconds: float = 0.6,
        cooldown_seconds: float = 1.0,
    ) -> None:
        if not 0 < return_threshold < start_threshold:
            raise ValueError("Ngưỡng trở về phải nằm giữa 0 và ngưỡng bắt đầu")
        self.ready_seconds = ready_seconds
        self.start_threshold = start_threshold
        self.return_threshold = return_threshold
        self.start_hold_seconds = start_hold_seconds
        self.return_hold_seconds = return_hold_seconds
        self.min_attempt_seconds = min_attempt_seconds
        self.max_attempt_seconds = max_attempt_seconds
        self.pre_roll_seconds = pre_roll_seconds
        self.cooldown_seconds = cooldown_seconds
        self.state = "ready"
        self.baseline: np.ndarray | None = None
        self.ready_samples: deque[tuple[float, np.ndarray]] = deque()
        self.pre_roll: deque[tuple[float, np.ndarray]] = deque()
        self.motion_since: float | None = None
        self.return_since: float | None = None
        self.attempt_start: float | None = None
        self.cooldown_until = 0.0

    def consume_pre_roll(self) -> list[tuple[float, np.ndarray]]:
        frames = list(self.pre_roll)
        self.pre_roll.clear()
        return frames

    def _finish(self, timestamp: float, event: str) -> str:
        self.state = "cooldown"
        self.cooldown_until = timestamp + self.cooldown_seconds
        self.ready_samples.clear()
        self.pre_roll.clear()
        self.baseline = None
        self.motion_since = None
        self.return_since = None
        self.attempt_start = None
        return event

    def update(self, timestamp: float, frame: np.ndarray | None) -> str | None:
        """Return 'start', 'end', or 'timeout' when a boundary is detected."""
        if self.state == "cooldown":
            if timestamp < self.cooldown_until:
                return None
            self.state = "ready"

        if frame is None:
            if self.state != "recording":
                self.state = "ready"
                self.ready_samples.clear()
                self.pre_roll.clear()
                self.baseline = None
                self.motion_since = None
            else:
                self.return_since = None
                if timestamp - self.attempt_start >= self.max_attempt_seconds:
                    return self._finish(timestamp, "timeout")
            return None

        signature = motion_signature(frame)
        if self.state == "ready":
            self.ready_samples.append((timestamp, signature))
            while timestamp - self.ready_samples[0][0] > self.ready_seconds + 0.1:
                self.ready_samples.popleft()
            if timestamp - self.ready_samples[0][0] >= self.ready_seconds:
                baseline = np.median([item[1] for item in self.ready_samples], axis=0)
                if all(pose_distance(item[1], baseline) <= 0.1 for item in self.ready_samples):
                    self.baseline = baseline
                    self.state = "armed"
                    self.pre_roll.clear()
                    self.pre_roll.append((timestamp, frame.copy()))
            return None

        if self.state == "armed":
            self.pre_roll.append((timestamp, frame.copy()))
            while timestamp - self.pre_roll[0][0] > self.pre_roll_seconds:
                self.pre_roll.popleft()
            if pose_distance(signature, self.baseline) >= self.start_threshold:
                if self.motion_since is None:
                    self.motion_since = timestamp
                if timestamp - self.motion_since >= self.start_hold_seconds:
                    self.state = "recording"
                    self.attempt_start = self.pre_roll[0][0]
                    self.return_since = None
                    return "start"
            else:
                self.motion_since = None
            return None

        if timestamp - self.attempt_start >= self.max_attempt_seconds:
            return self._finish(timestamp, "timeout")
        if pose_distance(signature, self.baseline) <= self.return_threshold:
            if self.return_since is None:
                self.return_since = timestamp
            if (
                timestamp - self.return_since >= self.return_hold_seconds
                and timestamp - self.attempt_start >= self.min_attempt_seconds
            ):
                return self._finish(timestamp, "end")
        else:
            self.return_since = None
        return None
