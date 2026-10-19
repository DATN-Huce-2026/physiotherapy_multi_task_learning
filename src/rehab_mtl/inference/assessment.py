"""Đổi đầu ra của model thành kết quả: đúng/sai, sai ở đâu, điểm 0-100.

- Đúng/sai: trung bình P(đúng) / (P(đúng) + P(lỗi mạnh nhất)) của 3 head có lớp đúng.
- Điểm: (1 - quality_weight) × độ đúng + quality_weight × quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rehab_mtl.datasets.label_maps import TASK_MAPPINGS

# Tên lớp theo thứ tự id lúc train
TASK_LABELS: dict[str, tuple[str, ...]] = {
    task: tuple(sorted(mapping, key=mapping.get)) for task, mapping in TASK_MAPPINGS.items()
}
CORRECT_HEADS = ("global_evaluation", "exercise_error_class", "error_type")
ERROR_CODES = ("Error1", "Error2", "Error3", "Other")
DEFAULT_PASS_SCORE = 50.0
DEFAULT_QUALITY_WEIGHT = 0.5

GLOBAL_EVALUATION_VI = {
    "Correct": "Đúng",
    "Incorrect": "Sai động tác",
    "Incomplete": "Chưa hoàn thành động tác",
    "Motionless": "Không có chuyển động",
}
SEVERITY_VI = {"SmallError": "nhẹ", "LargeError": "nặng"}
BODY_PART_VI = {
    "Spine": "cột sống / thân",
    "BothArms": "cả hai tay",
    "RightArm": "tay phải",
    "LeftArm": "tay trái",
}
GRADES = ((80.0, "Tốt"), (65.0, "Khá"), (50.0, "Đạt"), (35.0, "Chưa đạt"), (0.0, "Kém"))
DEFAULT_DESCRIPTIONS = Path(__file__).with_name("error_descriptions.json")


def grade_of(score: float) -> str:
    for minimum, label in GRADES:
        if score >= minimum:
            return label
    return GRADES[-1][1]


def load_descriptions(path: str | Path | None = None) -> dict[str, object]:
    with Path(path or DEFAULT_DESCRIPTIONS).open(encoding="utf-8") as file:
        return json.load(file)


def _top(task: str, probabilities: np.ndarray) -> tuple[str, float]:
    index = int(np.argmax(probabilities))
    return TASK_LABELS[task][index], round(float(probabilities[index]), 4)


def assess(
    probabilities: dict[str, np.ndarray],
    quality: float | None = None,
    *,
    expected_exercise: str | None = None,
    pass_score: float = DEFAULT_PASS_SCORE,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    descriptions: dict[str, object] | None = None,
) -> dict[str, object]:
    if not 0 < pass_score < 100:
        raise ValueError("pass_score phải nằm trong (0, 100)")
    if not 0 <= quality_weight <= 1:
        raise ValueError("quality_weight phải nằm trong [0, 1]")
    descriptions = descriptions or load_descriptions()
    probs = {task: np.asarray(probabilities[task], dtype=float) for task in TASK_LABELS}

    # Lớp 0 của 3 head này là "đúng"
    head_correctness = [
        probs[task][0] / max(probs[task][0] + probs[task][1:].max(), 1e-12)
        for task in CORRECT_HEADS
    ]
    correctness = 100.0 * float(np.mean(head_correctness))
    quality_score = None if quality is None else 100.0 * float(np.clip(quality, 0, 1))
    if quality_score is None:
        score = correctness
    else:
        score = (1 - quality_weight) * correctness + quality_weight * quality_score
    score = round(score, 1)
    passed = correctness >= pass_score

    exercise, exercise_conf = _top("exercise", probs["exercise"])
    result: dict[str, object] = {
        "exercise": exercise,
        "exercise_name": descriptions["exercise_names"].get(exercise, exercise),
        "exercise_confidence": exercise_conf,
        "verdict": "ĐÚNG" if passed else "SAI",
        "passed": passed,
        "score": score,
        "grade": grade_of(score),
        "correctness": round(correctness, 1),
        "quality": None if quality_score is None else round(quality_score, 1),
        "pass_score": pass_score,
        "head_predictions": {
            task: dict(zip(("label", "confidence"), _top(task, probs[task])))
            for task in TASK_LABELS
        },
        "warnings": [],
        "error": None,
    }
    if expected_exercise and expected_exercise != exercise:
        result["warnings"].append(
            f"Model nhận ra bài {exercise} ({exercise_conf:.0%}) nhưng bài cần tập là "
            f"{expected_exercise}: có thể tập nhầm bài hoặc động tác sai nhiều."
        )
    if exercise_conf < 0.6:
        result["warnings"].append(
            f"Model không chắc đây là bài nào (độ tin cậy {exercise_conf:.0%})."
        )
    if passed:
        return result

    # Gộp xác suất mã lỗi từ 2 head
    eec = dict(zip(TASK_LABELS["exercise_error_class"], probs["exercise_error_class"]))
    err = dict(zip(TASK_LABELS["error_type"], probs["error_type"]))
    error_probabilities = {
        code: float(np.mean([eec[code], err[code]])) if code in eec else float(err[code])
        for code in ERROR_CODES
    }
    error_code = max(error_probabilities, key=error_probabilities.get)
    reason = TASK_LABELS["global_evaluation"][1 + int(np.argmax(probs["global_evaluation"][1:]))]
    body_part, body_conf = _top("body_part", probs["body_part"])
    severity, severity_conf = _top("severity", probs["severity"])
    description = descriptions["errors"].get(exercise, {}).get(error_code, error_code)
    result["error"] = {
        "reason": reason,
        "error_type": error_code,
        "error_type_confidence": round(error_probabilities[error_code], 4),
        "description": description,
        "body_part": body_part,
        "body_part_confidence": body_conf,
        "severity": severity,
        "severity_confidence": severity_conf,
        "message": (
            f"{GLOBAL_EVALUATION_VI[reason]}: {description} "
            f"({error_probabilities[error_code]:.0%}) - bộ phận: "
            f"{BODY_PART_VI[body_part]} ({body_conf:.0%}) - mức độ: "
            f"{SEVERITY_VI[severity]} ({severity_conf:.0%})"
        ),
    }
    return result
