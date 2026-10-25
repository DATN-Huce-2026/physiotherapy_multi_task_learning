"""Trích xuất 12 góc khớp sinh cơ học và chuẩn hóa chiều dài chuỗi (T=90 frames).

[BỔ SUNG MỚI THEO YÊU CẦU]:
1. Chuyển đổi chuỗi tọa độ skeleton (25 khớp x 3 = 75 tọa độ) sang 12 góc giải phẫu chính.
2. Đồng bộ thời gian: Sử dụng scipy.interpolate.interp1d để nội suy co/giãn chuỗi về cố định T=90 frames.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

# ==============================================================================
# Chỉ số 25 khớp chuẩn Kinect v2
# ==============================================================================
SPINE_BASE = 0
SPINE_MID = 1
NECK = 2
HEAD = 3
SHOULDER_LEFT = 4
ELBOW_LEFT = 5
WRIST_LEFT = 6
HAND_LEFT = 7
SHOULDER_RIGHT = 8
ELBOW_RIGHT = 9
WRIST_RIGHT = 10
HAND_RIGHT = 11
HIP_LEFT = 12
KNEE_LEFT = 13
ANKLE_LEFT = 14
FOOT_LEFT = 15
HIP_RIGHT = 16
KNEE_RIGHT = 17
ANKLE_RIGHT = 18
FOOT_RIGHT = 19
SPINE_SHOULDER = 20

# Danh sách tên 12 góc giải phẫu
ANGLE_NAMES = (
    "left_elbow",      # 1. Khuỷu tay trái
    "right_elbow",     # 2. Khuỷu tay phải
    "left_shoulder",   # 3. Khớp vai trái
    "right_shoulder",  # 4. Khớp vai phải
    "left_hip",        # 5. Khớp háng trái
    "right_hip",       # 6. Khớp háng phải
    "left_knee",       # 7. Khớp gối trái
    "right_knee",      # 8. Khớp gối phải
    "left_ankle",      # 9. Cổ chân trái
    "right_ankle",     # 10. Cổ chân phải
    "spine_flexion",   # 11. Gập/duỗi cột sống
    "trunk_tilt",      # 12. Nghiêng thân mình (Lateral trunk tilt)
)


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """[BỔ SUNG]: Tính góc 3D tại đỉnh b tạo bởi ba khớp (a - b - c).

    Công thức: theta = arccos((ba . bc) / (|ba| * |bc|))
    Giá trị trả về nằm trong khoảng [0, pi] (radian).
    """
    ba = a - b
    bc = c - b

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))

    # Xử lý trường hợp suy biến khi hai khớp trùng nhau
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    # Chặn trong [-1.0, 1.0] để tránh lỗi số học ngoài miền của arccos do sai số float
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.arccos(cosine))


def extract_angles_from_kinect_frame(frame: np.ndarray) -> np.ndarray:
    """[BỔ SUNG]: Chuyển 1 frame tọa độ Kinect sang vector 12 góc sinh cơ học.

    Tham số:
        frame: Mảng shape [25, 3] hoặc [75] chứa tọa độ XYZ của 25 khớp Kinect.
    Trả về:
        Mảng numpy shape [12] kiểu float32 chứa 12 góc lâm sàng.
    """
    joints = np.asarray(frame, dtype=np.float32).reshape(25, 3)

    angles = [
        # 1. Khuỷu tay trái: ShoulderLeft(4) - ElbowLeft(5) - WristLeft(6)
        calculate_angle_3d(joints[SHOULDER_LEFT], joints[ELBOW_LEFT], joints[WRIST_LEFT]),
        # 2. Khuỷu tay phải: ShoulderRight(8) - ElbowRight(9) - WristRight(10)
        calculate_angle_3d(joints[SHOULDER_RIGHT], joints[ELBOW_RIGHT], joints[WRIST_RIGHT]),
        # 3. Khớp vai trái: SpineShoulder(20) - ShoulderLeft(4) - ElbowLeft(5)
        calculate_angle_3d(joints[SPINE_SHOULDER], joints[SHOULDER_LEFT], joints[ELBOW_LEFT]),
        # 4. Khớp vai phải: SpineShoulder(20) - ShoulderRight(8) - ElbowRight(9)
        calculate_angle_3d(joints[SPINE_SHOULDER], joints[SHOULDER_RIGHT], joints[ELBOW_RIGHT]),
        # 5. Khớp háng trái: SpineBase(0) - HipLeft(12) - KneeLeft(13)
        calculate_angle_3d(joints[SPINE_BASE], joints[HIP_LEFT], joints[KNEE_LEFT]),
        # 6. Khớp háng phải: SpineBase(0) - HipRight(16) - KneeRight(17)
        calculate_angle_3d(joints[SPINE_BASE], joints[HIP_RIGHT], joints[KNEE_RIGHT]),
        # 7. Khớp gối trái: HipLeft(12) - KneeLeft(13) - AnkleLeft(14)
        calculate_angle_3d(joints[HIP_LEFT], joints[KNEE_LEFT], joints[ANKLE_LEFT]),
        # 8. Khớp gối phải: HipRight(16) - KneeRight(17) - AnkleRight(18)
        calculate_angle_3d(joints[HIP_RIGHT], joints[KNEE_RIGHT], joints[ANKLE_RIGHT]),
        # 9. Cổ chân trái: KneeLeft(13) - AnkleLeft(14) - FootLeft(15)
        calculate_angle_3d(joints[KNEE_LEFT], joints[ANKLE_LEFT], joints[FOOT_LEFT]),
        # 10. Cổ chân phải: KneeRight(17) - AnkleRight(18) - FootRight(19)
        calculate_angle_3d(joints[KNEE_RIGHT], joints[ANKLE_RIGHT], joints[FOOT_RIGHT]),
        # 11. Gập/duỗi cột sống: SpineBase(0) - SpineMid(1) - SpineShoulder(20)
        calculate_angle_3d(joints[SPINE_BASE], joints[SPINE_MID], joints[SPINE_SHOULDER]),
        # 12. Nghiêng thân mình: Góc giữa trục thân (SpineBase -> SpineShoulder) và trục hông (HipLeft -> HipRight)
        calculate_angle_3d(joints[SPINE_SHOULDER], joints[SPINE_BASE], joints[HIP_LEFT]),
    ]
    return np.array(angles, dtype=np.float32)


def resample_sequence_to_fixed_length(
    sequence: np.ndarray,
    target_length: int = 90,
) -> np.ndarray:
    """[BỔ SUNG]: Co/giãn chiều dài chuỗi về đúng target_length (mặc định 90 frames).

    Sử dụng nội suy tuyến tính (scipy.interpolate.interp1d).
    Tham số:
        sequence: Mảng shape [T, D] (ví dụ [T, 12] hoặc [T, 75]).
        target_length: Số lượng frames cố định mong muốn (mặc định 90).
    Trả về:
        Mảng numpy shape [target_length, D] kiểu float32.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    original_length = len(seq)

    if original_length == target_length:
        return seq

    if original_length < 2:
        # Nếu chuỗi chỉ có 1 frame, lặp lại cho đủ độ dài
        return np.repeat(seq, target_length, axis=0)[:target_length]

    # Trục thời gian chuẩn hóa [0, 1] cho chuỗi gốc và chuỗi mục tiêu
    x_orig = np.linspace(0.0, 1.0, num=original_length)
    x_target = np.linspace(0.0, 1.0, num=target_length)

    interpolator = interp1d(
        x_orig,
        seq,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    resampled = interpolator(x_target)
    return resampled.astype(np.float32)


def transform_kinect_sequence_to_angles(
    sequence: np.ndarray,
    target_length: int = 90,
) -> np.ndarray:
    """[BỔ SUNG]: Hàm biến đổi toàn diện từ chuỗi tọa độ sang chuỗi 12 góc 90 frames.

    Nhận chuỗi [T, 25, 3] hoặc [T, 75] -> Trích xuất 12 góc từng frame [T, 12] -> Nội suy về [90, 12].
    """
    seq = np.asarray(sequence, dtype=np.float32)
    t_frames = len(seq)

    # 1. Trích xuất 12 góc cho từng frame
    angles_list = [extract_angles_from_kinect_frame(seq[t]) for t in range(t_frames)]
    angles_seq = np.stack(angles_list, axis=0)  # Shape: [T, 12]

    # 2. Đồng bộ thời gian về target_length (90 frames)
    angles_resampled = resample_sequence_to_fixed_length(angles_seq, target_length=target_length)
    return angles_resampled
