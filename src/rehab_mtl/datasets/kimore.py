"""Reader and parser for KiMoRe 3D Kinect skeleton sequences and clinical scores."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rehab_mtl.datasets.keraal import (
    JOINT_COUNT,
    JOINT_NAMES,
    offset_to_initial_spine_base,
)

VALUES_PER_JOINT_KIMORE = 4  # X, Y, Z, TrackingState
EXPECTED_VALUES_PER_FRAME_KIMORE = JOINT_COUNT * VALUES_PER_JOINT_KIMORE  # 100 values


def load_kimore_joint_xyz(
    path: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> list[list[tuple[float, float, float]]]:
    """Load KiMoRe 3D Kinect positions as ``[T][25][3]``.

    In KiMoRe `JointPosition*.csv`, each frame contains 25 joint blocks of:
    ``X, Y, Z, TrackingState``.
    This function extracts (X, Y, Z) in meters and discards the TrackingState.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KiMoRe skeleton file not found: {path}")

    frames: list[list[tuple[float, float, float]]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Tokens may have a trailing comma
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            if len(tokens) < EXPECTED_VALUES_PER_FRAME_KIMORE:
                raise ValueError(
                    f"{path}:{line_number}: expected at least {EXPECTED_VALUES_PER_FRAME_KIMORE} "
                    f"values, found {len(tokens)}"
                )

            try:
                values = [float(t) for t in tokens[:EXPECTED_VALUES_PER_FRAME_KIMORE]]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric coordinate value") from exc

            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{path}:{line_number}: non-finite coordinate value found")

            frame: list[tuple[float, float, float]] = []
            for joint_index in range(JOINT_COUNT):
                offset = joint_index * VALUES_PER_JOINT_KIMORE
                frame.append((values[offset], values[offset + 1], values[offset + 2]))
            frames.append(frame)

    if not frames:
        raise ValueError(f"{path}: contains no valid skeleton frames")

    if expected_frame_count is not None and len(frames) != expected_frame_count:
        # Allow minor discrepancy if any, but log or check
        pass

    return frames


def load_kimore_sample(
    manifest_row: dict[str, str],
    target_row: dict[str, str],
    kimore_root: str | Path,
    *,
    relative_to_initial_spine_base: bool = True,
) -> dict[str, Any]:
    """Load one KiMoRe sequence with its multi-task targets.

    Extracts 3D coordinates, applies initial SpineBase centering (if requested),
    and formats clinical quality target alongside any classification targets.
    """
    sample_id = manifest_row.get("sample_id", "")
    if sample_id != target_row.get("sample_id"):
        raise ValueError(
            f"sample_id mismatch: manifest={sample_id!r}, targets={target_row.get('sample_id')!r}"
        )

    root = Path(kimore_root)
    relative_kinect_path = manifest_row.get("kinect_path", "").strip()
    if not relative_kinect_path:
        raise ValueError(f"{sample_id}: missing kinect_path in manifest")

    full_kinect_path = root / relative_kinect_path
    xyz = load_kimore_joint_xyz(full_kinect_path)

    xyz_origin_offset = None
    xyz_transform = "none"
    if relative_to_initial_spine_base:
        xyz_origin_offset = xyz[0][0]
        xyz = offset_to_initial_spine_base(xyz)
        xyz_transform = "initial_spine_base_offset"

    # Targets
    quality_mask_text = target_row.get("quality_mask", "false").strip().lower()
    has_quality = quality_mask_text == "true"
    raw_quality_str = target_row.get("quality_target", "").strip()
    quality_target = float(raw_quality_str) if has_quality and raw_quality_str else 0.0

    targets: dict[str, str | None] = {
        "exercise": target_row.get("exercise_target") or manifest_row.get("exercise_code"),
        "exercise_error_class": None,
        "global_evaluation": None,
        "error_type": None,
        "severity": None,
        "body_part": None,
    }
    target_masks: dict[str, bool] = {
        "exercise": target_row.get("exercise_mask", "false").strip().lower() == "true",
        "exercise_error_class": False,
        "global_evaluation": False,
        "error_type": False,
        "severity": False,
        "body_part": False,
        "quality": has_quality,
    }

    return {
        "sample_id": sample_id,
        "dataset": "kimore",
        "cohort": manifest_row.get("cohort", ""),
        "group": manifest_row.get("group", ""),
        "subject_token": manifest_row.get("subject_token", ""),
        "subject_id": manifest_row.get("subject_id", ""),
        "exercise_code": manifest_row.get("exercise_code", ""),
        "xyz": xyz,
        "xyz_shape": (len(xyz), JOINT_COUNT, 3),
        "xyz_transform": xyz_transform,
        "xyz_origin_offset": xyz_origin_offset,
        "quality_target": quality_target,
        "quality_mask": has_quality,
        "targets": targets,
        "target_masks": target_masks,
    }
