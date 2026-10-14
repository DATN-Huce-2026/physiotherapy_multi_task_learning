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

EXERCISE_ERROR_CLASS_TO_ID = {
    "Correct": 0,
    "Error1": 1,
    "Error2": 2,
    "Error3": 3,
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

TASK_MAPPINGS = {
    "exercise": EXERCISE_TO_ID,
    "exercise_error_class": EXERCISE_ERROR_CLASS_TO_ID,
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
