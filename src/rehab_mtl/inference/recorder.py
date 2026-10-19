"""Ghi chuỗi khung xương của một lần tập."""

from __future__ import annotations

import numpy as np

from rehab_mtl.inference.mediapipe_to_kinect import (
    DEFAULT_CAMERA_PITCH_DEG,
    KERAAL_FPS,
    REQUIRED_LANDMARKS,
    mediapipe_world_to_kinect,
    resample_sequence,
)


def world_landmarks_array(results) -> tuple[np.ndarray, np.ndarray] | None:
    """Trả (tọa độ [33, 3], visibility [33]), hoặc None nếu không thấy người."""
    if results is None or not results.pose_world_landmarks:
        return None
    landmarks = results.pose_world_landmarks.landmark
    xyz = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
    return xyz, visibility


class SkeletonRecorder:
    """Frame thiếu khớp được lưu NaN, lấp bằng nội suy khi xuất chuỗi."""

    def __init__(
        self,
        min_visibility: float = 0.5,
        pitch_deg: float = DEFAULT_CAMERA_PITCH_DEG,
    ) -> None:
        self.min_visibility = float(min_visibility)
        self.pitch_deg = float(pitch_deg)
        self.timestamps: list[float] = []
        self.frames: list[np.ndarray] = []
        self.visible_frames = 0

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0

    def reset(self) -> None:
        self.timestamps.clear()
        self.frames.clear()
        self.visible_frames = 0

    def add(self, timestamp: float, results) -> bool:
        """Trả True nếu frame đủ khớp để dùng."""
        data = world_landmarks_array(results)
        frame = np.full((25, 3), np.nan, dtype=np.float32)
        usable = False
        if data is not None:
            xyz, visibility = data
            if (visibility[list(REQUIRED_LANDMARKS)] >= self.min_visibility).all():
                frame = mediapipe_world_to_kinect(xyz, self.pitch_deg)
                usable = True
        self.timestamps.append(float(timestamp))
        self.frames.append(frame)
        self.visible_frames += int(usable)
        return usable

    def sequence(self, fps: float = KERAAL_FPS) -> np.ndarray:
        if not self.frames:
            raise ValueError("Chưa ghi được frame nào")
        return resample_sequence(
            np.asarray(self.timestamps), np.stack(self.frames), fps=fps
        )
