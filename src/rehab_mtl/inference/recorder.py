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


def usable_kinect_frame(
    results, min_visibility: float, pitch_deg: float
) -> np.ndarray | None:
    """Return a complete Kinect-shaped frame, or None when tracking is unusable."""
    data = world_landmarks_array(results)
    if data is None:
        return None
    xyz, visibility = data
    if not np.isfinite(xyz).all() or not np.isfinite(visibility).all():
        return None
    if not (visibility[list(REQUIRED_LANDMARKS)] >= min_visibility).all():
        return None
    return mediapipe_world_to_kinect(xyz, pitch_deg)


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
        frame = usable_kinect_frame(results, self.min_visibility, self.pitch_deg)
        return self.add_frame(timestamp, frame)

    def add_frame(self, timestamp: float, frame: np.ndarray | None) -> bool:
        """Record an already converted frame; None represents lost tracking."""
        usable = frame is not None
        if usable:
            frame = np.asarray(frame, dtype=np.float32)
            if frame.shape != (25, 3) or not np.isfinite(frame).all():
                raise ValueError("Frame Kinect phải có shape [25, 3] và hữu hạn")
        else:
            frame = np.full((25, 3), np.nan, dtype=np.float32)
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

    def sequence_for_angles(self, *, max_gap_seconds: float = 0.5) -> np.ndarray:
        """Fill short tracking gaps while retaining the captured frame timestamps.

        Training extracts angles from each original Kinect frame before resampling
        the angle sequence. This method only repairs missing camera frames; it does
        not resample XYZ to an intermediate FPS.
        """
        if not self.frames:
            raise ValueError("Chưa ghi được frame nào")

        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        frames = np.stack(self.frames).astype(np.float32)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Mốc thời gian của frame phải tăng dần")

        valid = np.isfinite(frames).all(axis=(1, 2))
        if valid.sum() < 2:
            raise ValueError("Cần ít nhất 2 frame thấy đủ khớp để tạo chuỗi")
        if not valid[0] or not valid[-1]:
            raise ValueError("Cần thấy đủ khớp ở đầu và cuối lần tập")
        if np.diff(timestamps[valid]).max() > max_gap_seconds:
            raise ValueError("Mất khớp quá lâu trong lần tập")

        if valid.all():
            return frames

        known = frames[valid].reshape(int(valid.sum()), -1)
        repaired = np.stack(
            [
                np.interp(timestamps, timestamps[valid], known[:, column])
                for column in range(known.shape[1])
            ],
            axis=1,
        )
        return repaired.reshape(len(frames), 25, 3).astype(np.float32)
