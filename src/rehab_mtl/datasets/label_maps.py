from __future__ import annotations

from collections.abc import Mapping

EXERCISE_TO_ID = {
    "CTK": 0,
    "ELK": 1,
    "RTK": 2,
}

GLOBAL_EVALUATION_TO_ID = {
    "Correct": 0,
    "Incorrect": 1,
    "Incomplete": 2,
    "Motionless": 3,
}

ERROR_TYPE_TO_ID = {
    "none": 0,
    "Error1": 1,
    "Error2": 2,
    "Error3": 3,
    "Other": 4,
}

SEVERITY_TO_ID = {
    "SmallError": 0,
    "LargeError": 1,
}

BODY_PART_TO_ID = {
    "Spine": 0,
    "BothArms": 1,
    "RightArm": 2,
    "LeftArm": 3,
}

TASK_MAPPINGS = {
    "exercise": EXERCISE_TO_ID,
    "global_evaluation": GLOBAL_EVALUATION_TO_ID,
    "error_type": ERROR_TYPE_TO_ID,
    "severity": SEVERITY_TO_ID,
    "body_part": BODY_PART_TO_ID,
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