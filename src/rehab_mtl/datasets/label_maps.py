from __future__ import annotations

from collections.abc import Mapping

# ==============================================================================
# 1. Exercise Classification (Phân loại bài tập phục hồi chức năng)
# ==============================================================================
EXERCISE_TO_ID = {
    "CTK": 0,  # Clear The Desk / Lateral trunk tilt (Nghiêng thân mình sang bên)
    "ELK": 1,  # Elevate the arms / Bilateral arm lifting (Vươn/nâng hai cánh tay)
    "RTK": 2,  # Rotate trunk / Trunk rotation (Xoay thân mình)
}

# ==============================================================================
# 2. Global Evaluation (Đánh giá tổng quát mức độ hoàn thành động tác)
# ==============================================================================
GLOBAL_EVALUATION_TO_ID = {
    "Correct": 0,     # Thực hiện đúng bài tập
    "Incorrect": 1,   # Thực hiện sai kỹ thuật (mắc lỗi động tác)
    "Incomplete": 2,  # Chưa hoàn thành trọn vẹn động tác (bỏ dở giữa chừng)
    "Motionless": 3,  # Bất động / không thực hiện động tác
}

# ==============================================================================
# 3. Error Type Detection (Phân loại loại lỗi cụ thể mắc phải)
# ==============================================================================
ERROR_TYPE_TO_ID = {
    "none": 0,    # Không có lỗi (động tác chuẩn)
    "Error1": 1,  # Lỗi loại 1
    "Error2": 2,  # Lỗi loại 2
    "Error3": 3,  # Lỗi loại 3
    "Other": 4,   # Lỗi khác (ngoài 3 lỗi chính được định nghĩa cho bài tập)
}

# ==============================================================================
# 4. Error Severity (Mức độ nghiêm trọng của lỗi)
# ==============================================================================
SEVERITY_TO_ID = {
    "SmallError": 0,  # Lỗi nhỏ / nhẹ
    "LargeError": 1,  # Lỗi lớn / nghiêm trọng
}

# ==============================================================================
# 5. Affected Body Part (Bộ phận cơ thể mắc lỗi)
# ==============================================================================
BODY_PART_TO_ID = {
    "Spine": 0,     # Cột sống / thân người / lưng
    "BothArms": 1,  # Cả hai cánh tay
    "RightArm": 2,  # Cánh tay phải
    "LeftArm": 3,   # Cánh tay trái
}

# ==============================================================================
# 6. KERAAL 4-Class Exercise Error (Phân loại 4 lớp chuẩn benchmark KERAAL)
# ==============================================================================
EXERCISE_ERROR_CLASS_TO_ID = {
    "Correct": 0,  # Động tác chuẩn / đúng hoàn toàn
    "Error1": 1,   # Lỗi loại 1
    "Error2": 2,   # Lỗi loại 2
    "Error3": 3,   # Lỗi loại 3
}

# The KERAAL four-class task aligns with the E1/E2/E3 filename prefixes.
# Keep B1/B2/B3 as raw filename variants; their semantics are not decoded here.
_GROUP3_CODE_TO_ERROR_TYPE = {
    "E1": "Error1",
    "E2": "Error2",
    "E3": "Error3",
}
GROUP3_LABEL_CODE_TO_EXERCISE_ERROR_CLASS = {
    "C": "Correct",
    **{
        f"{prefix}B{variant}": error_type
        for prefix, error_type in _GROUP3_CODE_TO_ERROR_TYPE.items()
        for variant in (1, 2, 3)
    },
}
GROUP3_LABEL_CODE_TO_GLOBAL_EVALUATION = {
    code: "Correct" if code == "C" else "Incorrect"
    for code in GROUP3_LABEL_CODE_TO_EXERCISE_ERROR_CLASS
}
GROUP3_LABEL_CODE_TO_ERROR_TYPE = {
    code: "none" if code == "C" else error_type
    for code, error_type in GROUP3_LABEL_CODE_TO_EXERCISE_ERROR_CLASS.items()
}

# ==============================================================================
# Registry quản lý ánh xạ cho tất cả các task trong Multi-Task Learning
# ==============================================================================
TASK_MAPPINGS = {
    "exercise": EXERCISE_TO_ID,                          # Phân loại bài tập (3 lớp)
    "exercise_error_class": EXERCISE_ERROR_CLASS_TO_ID,  # Phân loại 4 lớp benchmark KERAAL (Correct, E1, E2, E3)
    "global_evaluation": GLOBAL_EVALUATION_TO_ID,        # Đánh giá tổng quát (Correct, Incorrect, Incomplete, Motionless)
    "error_type": ERROR_TYPE_TO_ID,                      # Loại lỗi chi tiết (none, Error1-3, Other)
    "severity": SEVERITY_TO_ID,                          # Mức độ nghiêm trọng của lỗi (SmallError, LargeError)
    "body_part": BODY_PART_TO_ID,                        # Vị trí bộ phận cơ thể mắc lỗi (Spine, BothArms, RightArm, LeftArm)
}


def encode_target(
    value: str | None,
    mapping: Mapping[str, int],
    mask: bool,
) -> int | None:
    if not mask:
        if value is not None:
            raise ValueError("masked target must be None")
        return None

    if value is None:
        raise ValueError("unmasked target cannot be None")

    if value not in mapping:
        raise ValueError(f"unknown target label: {value!r}")

    return mapping[value]
