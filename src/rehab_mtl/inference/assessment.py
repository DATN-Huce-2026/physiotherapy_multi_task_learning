"""Đổi đầu ra của model thành kết quả: đúng/sai, sai ở đâu, điểm 0-100.

- Đúng/sai: trung bình P(đúng) / (P(đúng) + P(lỗi mạnh nhất)) của 3 head có lớp đúng.
- Điểm:
  + Chế độ 'penalty' (mặc định): trừ điểm từ 100 theo mức độ nghiêm trọng
    (đúng: 90-100, lỗi nhẹ SmallError: 70-85, lỗi nặng LargeError: 30-50,
    chưa hoàn thành: 35-50, không chuyển động: 5-20).
  + Chế độ 'strict': (1 - quality_weight) × độ đúng + quality_weight × quality.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any

import numpy as np

from rehab_mtl.datasets.label_maps import TASK_MAPPINGS

# Tên lớp theo thứ tự id lúc train
TASK_LABELS: dict[str, tuple[str, ...]] = {
    task: tuple(sorted(mapping, key=lambda k: mapping[k])) for task, mapping in TASK_MAPPINGS.items()
}
CORRECT_HEADS = ("global_evaluation", "exercise_error_class", "error_type")
ERROR_CODES = ("Error1", "Error2", "Error3", "Other")
DEFAULT_PASS_SCORE = 50.0
DEFAULT_QUALITY_WEIGHT = 0.5
DEFAULT_SCORING_MODE = "penalty"
VALID_SCORING_MODES = ("penalty", "strict")

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


def load_descriptions(path: str | Path | None = None) -> dict[str, Any]:
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
    scoring_mode: str = DEFAULT_SCORING_MODE,
    descriptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0 < pass_score < 100:
        raise ValueError("pass_score phải nằm trong (0, 100)")
    if not 0 <= quality_weight <= 1:
        raise ValueError("quality_weight phải nằm trong [0, 1]")
    if scoring_mode not in VALID_SCORING_MODES:
        raise ValueError(
            f"scoring_mode phải là một trong {VALID_SCORING_MODES}, nhận được: {scoring_mode!r}"
        )
    descriptions = descriptions or load_descriptions()
    probs = {task: np.asarray(probabilities[task], dtype=float) for task in TASK_LABELS}

    # Lớp 0 của 3 head này là "đúng"
    head_correctness = [
        probs[task][0] / max(probs[task][0] + probs[task][1:].max(), 1e-12)
        for task in CORRECT_HEADS
    ]
    correctness = 100.0 * float(np.mean(head_correctness))
    passed = correctness >= pass_score

    exercise, exercise_conf = _top("exercise", probs["exercise"])
    if expected_exercise and expected_exercise != exercise:
        return {
            "exercise": exercise,
            "exercise_name": descriptions["exercise_names"].get(exercise, exercise),
            "exercise_confidence": exercise_conf,
            "verdict": "CHƯA CHẤM",
            "passed": None,
            "score": None,
            "grade": None,
            "correctness": None,
            "quality": None,
            "pass_score": pass_score,
            "scoring_mode": scoring_mode,
            "head_predictions": {
                task: dict(zip(("label", "confidence"), _top(task, probs[task])))
                for task in TASK_LABELS
            },
            "warnings": [
                f"Model dự đoán {exercise} ({exercise_conf:.0%}), khác bài đã chọn "
                f"{expected_exercise}; chưa thể chấm đúng/sai hoặc gán lỗi cho lần tập này."
            ],
            "error": None,
            "unscored_reason": "exercise_mismatch",
            "expected_exercise": expected_exercise,
        }

    warnings: list[str] = []
    if exercise_conf < 0.6:
        warnings.append(
            f"Model không chắc đây là bài nào (độ tin cậy {exercise_conf:.0%})."
        )

    # Trích xuất thông tin lỗi nếu không passed
    error_info = None
    if not passed:
        eec = dict(zip(TASK_LABELS["exercise_error_class"], probs["exercise_error_class"]))
        err = dict(zip(TASK_LABELS["error_type"], probs["error_type"]))
        error_probabilities = {
            code: float(np.mean([eec[code], err[code]])) if code in eec else float(err[code])
            for code in ERROR_CODES
        }
        error_code = max(error_probabilities, key=lambda k: error_probabilities[k])
        error_conf = error_probabilities[error_code]
        reason = TASK_LABELS["global_evaluation"][1 + int(np.argmax(probs["global_evaluation"][1:]))]
        body_part, body_conf = _top("body_part", probs["body_part"])
        severity, severity_conf = _top("severity", probs["severity"])
        description = descriptions["errors"].get(exercise, {}).get(error_code, error_code)
        error_info = {
            "reason": reason,
            "error_type": error_code,
            "error_type_confidence": round(error_conf, 4),
            "description": description,
            "body_part": body_part,
            "body_part_confidence": body_conf,
            "severity": severity,
            "severity_confidence": severity_conf,
            "message": (
                f"{GLOBAL_EVALUATION_VI[reason]}: {description} "
                f"({error_conf:.0%}) - bộ phận: "
                f"{BODY_PART_VI[body_part]} ({body_conf:.0%}) - mức độ: "
                f"{SEVERITY_VI[severity]} ({severity_conf:.0%})"
            ),
        }

    # Tính điểm phân loại
    if scoring_mode == "penalty":
        if passed:
            norm = max(100.0 - pass_score, 1e-12)
            classification_score = 90.0 + 10.0 * float(
                np.clip((correctness - pass_score) / norm, 0.0, 1.0)
            )
        else:
            assert error_info is not None
            p_small = float(probs["severity"][0])
            p_large = float(probs["severity"][1])
            sev_sum = max(p_small + p_large, 1e-12)
            p_small /= sev_sum
            p_large /= sev_sum

            if error_info["reason"] == "Motionless":
                motionless_conf = float(probs["global_evaluation"][3])
                penalty = 80.0 + 15.0 * motionless_conf
            elif error_info["reason"] == "Incomplete":
                incomplete_conf = float(probs["global_evaluation"][2])
                penalty = 50.0 + 15.0 * incomplete_conf
            else:  # Incorrect: trừ điểm theo SmallError (15-30) hoặc LargeError (50-70)
                err_conf = float(error_info["error_type_confidence"])
                penalty_small = 15.0 + 15.0 * err_conf
                penalty_large = 50.0 + 20.0 * err_conf
                penalty = p_small * penalty_small + p_large * penalty_large

            classification_score = float(np.clip(100.0 - penalty, 0.0, 100.0))
    else:  # "strict"
        classification_score = correctness

    quality_score = None if quality is None else 100.0 * float(np.clip(quality, 0, 1))
    if quality_score is None:
        score = classification_score
    else:
        score = (1 - quality_weight) * classification_score + quality_weight * quality_score
    score = round(score, 1)

    return {
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
        "scoring_mode": scoring_mode,
        "head_predictions": {
            task: dict(zip(("label", "confidence"), _top(task, probs[task])))
            for task in TASK_LABELS
        },
        "warnings": warnings,
        "error": error_info,
    }
