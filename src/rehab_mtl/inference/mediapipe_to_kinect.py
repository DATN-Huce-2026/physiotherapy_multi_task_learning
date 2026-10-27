"""Chuyển 33 điểm MediaPipe sang khung xương 25 khớp Kinect v2 (dạng model đã học)."""

from __future__ import annotations

import numpy as np

KERAAL_FPS = 10.0
KIMORE_FPS = 30.0
DEFAULT_CAMERA_PITCH_DEG = 8.0  # bù góc lệch giữa webcam và Kinect

NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Khớp Kinect -> landmark MediaPipe tương ứng
DIRECT_JOINTS = {
    4: LEFT_SHOULDER,
    5: LEFT_ELBOW,
    6: LEFT_WRIST,
    8: RIGHT_SHOULDER,
    9: RIGHT_ELBOW,
    10: RIGHT_WRIST,
    12: LEFT_HIP,
    13: LEFT_KNEE,
    14: LEFT_ANKLE,
    15: LEFT_FOOT_INDEX,
    16: RIGHT_HIP,
    17: RIGHT_KNEE,
    18: RIGHT_ANKLE,
    19: RIGHT_FOOT_INDEX,
    21: LEFT_INDEX,  # HandTipLeft
    22: LEFT_THUMB,
    23: RIGHT_INDEX,  # HandTipRight
    24: RIGHT_THUMB,
}
# Tỉ lệ đo từ dữ liệu Kinect thật, dùng để nội suy khớp MediaPipe không có
SPINE_MID_RATIO = 0.57
NECK_RATIO = 0.31
SPINE_BASE_OFFSET = np.array([0.0, 0.0, 0.03], dtype=np.float32)
SPINE_SHOULDER_OFFSET = np.array([0.0, 0.015, 0.01], dtype=np.float32)
# Frame chỉ được dùng khi thấy đủ các landmark này
REQUIRED_LANDMARKS = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)


def rotation_x(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=np.float32
    )


def mediapipe_world_to_kinect(
    landmarks: np.ndarray,
    pitch_deg: float = DEFAULT_CAMERA_PITCH_DEG,
) -> np.ndarray:
    """[33, 3] MediaPipe world landmarks -> [25, 3] khớp Kinect."""
    mp = np.asarray(landmarks, dtype=np.float32)
    if mp.shape != (33, 3):
        raise ValueError("landmarks phải có shape [33, 3]")
    # MediaPipe: x sang trái, y xuống; Kinect: x sang phải, y lên
    mp = mp * np.array([-1.0, -1.0, 1.0], dtype=np.float32)

    kinect = np.zeros((25, 3), dtype=np.float32)
    for kinect_index, mp_index in DIRECT_JOINTS.items():
        kinect[kinect_index] = mp[mp_index]

    spine_base = (mp[LEFT_HIP] + mp[RIGHT_HIP]) / 2 + SPINE_BASE_OFFSET
    spine_shoulder = (
        mp[LEFT_SHOULDER] + mp[RIGHT_SHOULDER]
    ) / 2 + SPINE_SHOULDER_OFFSET
    head = (mp[LEFT_EAR] + mp[RIGHT_EAR]) / 2
    kinect[0] = spine_base
    kinect[1] = spine_base + SPINE_MID_RATIO * (spine_shoulder - spine_base)
    kinect[2] = spine_shoulder + NECK_RATIO * (head - spine_shoulder)
    kinect[3] = head
    kinect[20] = spine_shoulder
    kinect[7] = (mp[LEFT_WRIST] + mp[LEFT_PINKY] + mp[LEFT_INDEX]) / 3  # HandLeft
    kinect[11] = (mp[RIGHT_WRIST] + mp[RIGHT_PINKY] + mp[RIGHT_INDEX]) / 3
    if pitch_deg:
        kinect = kinect @ rotation_x(pitch_deg).T
    return kinect


def resample_sequence(
    timestamps: np.ndarray,
    frames: np.ndarray,
    fps: float = KERAAL_FPS,
) -> np.ndarray:
    """Lấy mẫu lại chuỗi [T, 25, 3] về ``fps``; frame NaN được lấp bằng nội suy."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float32)
    if len(timestamps) != len(frames):
        raise ValueError("timestamps và frames phải cùng độ dài")
    valid = np.isfinite(frames).all(axis=(1, 2))
    if valid.sum() < 2:
        raise ValueError("Cần ít nhất 2 frame nhìn thấy người để tạo chuỗi")
    times = timestamps[valid]
    values = frames[valid].reshape(int(valid.sum()), -1)
    order = np.argsort(times)
    times, values = times[order], values[order]

    target = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    resampled = np.stack(
        [np.interp(target, times, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )
    return resampled.reshape(len(target), 25, 3).astype(np.float32)
