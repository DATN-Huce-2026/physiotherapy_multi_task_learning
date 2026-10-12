"""Read-only KERAAL manifest and Kinect skeleton loader."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

JOINT_NAMES = (
    "SpineBase",
    "SpineMid",
    "Neck",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
    "SpineShoulder",
    "HandTipLeft",
    "ThumbLeft",
    "HandTipRight",
    "ThumbRight",
)

JOINT_COUNT = 25
VALUES_PER_JOINT = 7
VALUES_PER_FRAME = JOINT_COUNT * VALUES_PER_JOINT
XYZ_VALUES_PER_FRAME = JOINT_COUNT * 3
EXPECTED_SAMPLE_COUNT = 301
EXPECTED_GROUP_COUNTS = {"group1A": 250, "group2A": 51}

PATH_COLUMNS = (
    "kinect_path",
    "video_path",
    "openpose_path",
    "blazepose_path",
    "annotator_a_xml_path",
    "annotator_b_xml_path",
)

LABEL_FIELDS = {
    "global_evaluation": {
        "Correct",
        "Incorrect",
        "Incomplete",
        "Motionless",
    },
    "error_type": {"Error1", "Error2", "Error3", "Other", "none"},
    "severity": {"SmallError", "LargeError"},
    "body_part": {"Spine", "BothArms", "LeftArm", "RightArm"},
}
MTL_TARGET_FIELDS = (
    "exercise",
    "global_evaluation",
    "error_type",
    "severity",
    "body_part",
)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _source_path(root: Path, relative_path: str, *, sample_id: str, field: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"{sample_id}: {field} must be relative to the KERAAL root")

    resolved_root = root.resolve(strict=True)
    resolved_path = (resolved_root / candidate).resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{sample_id}: {field} escapes the KERAAL root")
    if not resolved_path.is_file():
        raise ValueError(f"{sample_id}: {field} is not a file: {relative_path}")
    return resolved_path


def load_kinect_xyz(
    path: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> list[list[tuple[float, float, float]]]:
    """Load Kinect positions as ``[T][25][3]`` without changing coordinates.

    Each source frame contains 25 joint blocks of ``XYZ + quaternion XYZW``.
    This reader keeps XYZ and drops quaternion values for the v0 input.
    """
    path = Path(path)
    frames: list[list[tuple[float, float, float]]] = []

    with path.open("r", encoding="utf-8", newline=None) as source:
        for line_number, line in enumerate(source, start=1):
            tokens = line.split()
            if len(tokens) != VALUES_PER_FRAME:
                raise ValueError(
                    f"{path}:{line_number}: expected {VALUES_PER_FRAME} values, "
                    f"found {len(tokens)}"
                )
            try:
                values = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric value") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number}: non-finite value")

            frame = []
            for joint_index in range(JOINT_COUNT):
                offset = joint_index * VALUES_PER_JOINT
                frame.append((values[offset], values[offset + 1], values[offset + 2]))
            frames.append(frame)

    if not frames:
        raise ValueError(f"{path}: contains no frames")
    if expected_frame_count is not None and len(frames) != expected_frame_count:
        raise ValueError(
            f"{path}: manifest says {expected_frame_count} frames, "
            f"file contains {len(frames)}"
        )
    return frames


def offset_to_initial_spine_base(
    frames: list[list[tuple[float, float, float]]],
) -> list[list[tuple[float, float, float]]]:
    """Return a copy with the first-frame SpineBase position as the origin.

    The same origin is subtracted from every joint at every frame, so movement
    of SpineBase over time remains in the sequence.
    """
    if not frames:
        raise ValueError("cannot offset an empty Kinect sequence")
    if any(len(frame) != JOINT_COUNT for frame in frames):
        raise ValueError(f"each Kinect frame must contain {JOINT_COUNT} joints")
    if any(len(joint) != 3 for frame in frames for joint in frame):
        raise ValueError("each Kinect joint must contain XYZ coordinates")

    origin = frames[0][0]  # Kinect joint index 0 is SpineBase.
    return [
        [tuple(point[axis] - origin[axis] for axis in range(3)) for point in frame]
        for frame in frames
    ]


def load_keraal_sample(
    row: dict[str, str],
    keraal_root: str | Path,
    *,
    relative_to_initial_spine_base: bool = False,
) -> dict[str, Any]:
    """Load one manifest row, preserving labels and optionally offsetting XYZ."""
    sample_id = row["sample_id"]
    root = Path(keraal_root)
    kinect_path = _source_path(
        root, row["kinect_path"], sample_id=sample_id, field="kinect_path"
    )
    try:
        expected_frames = int(row["kinect_frame_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: invalid kinect_frame_count") from exc

    xyz = load_kinect_xyz(kinect_path, expected_frame_count=expected_frames)
    xyz_origin_offset = None
    xyz_transform = "none"
    if relative_to_initial_spine_base:
        xyz_origin_offset = xyz[0][0]
        xyz = offset_to_initial_spine_base(xyz)
        xyz_transform = "initial_spine_base_offset"
    labels: dict[str, dict[str, str | None]] = {}
    label_masks: dict[str, dict[str, bool]] = {}
    temporal_annotations: dict[str, dict[str, str | int | None]] = {}

    for annotator in ("a", "b"):
        label_values = {
            field: _optional_text(row[f"annotator_{annotator}_{field}"])
            for field in LABEL_FIELDS
        }
        labels[annotator] = label_values
        # Explicit "none" remains a labeled value; an empty cell is unknown.
        label_masks[annotator] = {
            field: value is not None for field, value in label_values.items()
        }

        status = _optional_text(row[f"annotator_{annotator}_temporal_error_status"])
        count_text = _optional_text(
            row[f"annotator_{annotator}_temporal_error_event_count"]
        )
        event_count = int(count_text) if count_text is not None else None
        if status == "track_absent" and event_count is not None:
            raise ValueError(f"{sample_id}: absent temporal track must have blank event count")
        if status == "track_empty" and event_count != 0:
            raise ValueError(f"{sample_id}: empty temporal track must have event count 0")
        if status == "annotated" and (event_count is None or event_count < 1):
            raise ValueError(f"{sample_id}: annotated temporal track needs a positive event count")
        if status not in {"annotated", "track_empty", "track_absent"}:
            raise ValueError(f"{sample_id}: unexpected temporal status {status!r}")
        temporal_annotations[annotator] = {
            "status": status,
            "event_count": event_count,
        }

    return {
        "sample_id": sample_id,
        "group": row["group"],
        "subject_id": f"{row['group']}:{row['subject_token']}",
        "exercise_code": row["exercise_code"],
        "xyz": xyz,
        "xyz_shape": (len(xyz), JOINT_COUNT, 3),
        "xyz_transform": xyz_transform,
        "xyz_origin_offset": xyz_origin_offset,
        "labels": labels,
        "label_masks": label_masks,
        "temporal_annotations": temporal_annotations,
        "kinect_path": row["kinect_path"],
    }


def load_keraal_mtl_sample(
    manifest_row: dict[str, str],
    target_row: dict[str, str],
    keraal_root: str | Path,
    *,
    relative_to_initial_spine_base: bool = False,
) -> dict[str, Any]:
    """Join one Kinect sequence with its proposed multi-task targets by ID.

    Targets come from the review-only target preview CSV. A masked target is
    returned as ``None`` and its false mask is preserved for the training loss.
    """
    sample_id = manifest_row.get("sample_id", "")
    if sample_id != target_row.get("sample_id"):
        raise ValueError(
            f"sample_id mismatch: manifest={sample_id!r}, "
            f"targets={target_row.get('sample_id')!r}"
        )

    sample = load_keraal_sample(
        manifest_row,
        keraal_root,
        relative_to_initial_spine_base=relative_to_initial_spine_base,
    )
    targets: dict[str, str | None] = {}
    target_masks: dict[str, bool] = {}
    target_resolution: dict[str, str] = {}

    for field in MTL_TARGET_FIELDS:
        value = _optional_text(target_row.get(f"{field}_target"))
        mask_text = _optional_text(target_row.get(f"{field}_mask"))
        resolution = _optional_text(target_row.get(f"{field}_resolution"))
        if mask_text not in {"true", "false"}:
            raise ValueError(f"{sample_id}: invalid {field}_mask {mask_text!r}")
        mask = mask_text == "true"
        if mask != (value is not None):
            raise ValueError(
                f"{sample_id}: {field} target/mask disagree "
                f"(target={value!r}, mask={mask})"
            )
        if resolution is None:
            raise ValueError(f"{sample_id}: missing {field}_resolution")

        targets[field] = value
        target_masks[field] = mask
        target_resolution[field] = resolution

    sample["targets"] = targets
    sample["target_masks"] = target_masks
    sample["target_resolution"] = target_resolution
    sample["target_review_flags"] = tuple(
        flag for flag in (target_row.get("review_flags") or "").split(";") if flag
    )
    sample["target_policy_status"] = "provisional_review_preview"
    return sample


def _validate_label_value(value: str | None, *, sample_id: str, field: str) -> None:
    if value is None:
        return
    allowed = LABEL_FIELDS[field]
    values = [part.strip() for part in value.split("|")]
    unknown = [part for part in values if part not in allowed]
    if unknown:
        raise ValueError(f"{sample_id}: unknown {field} value(s): {unknown}")


def validate_keraal_manifest(manifest_path: str | Path, keraal_root: str | Path) -> dict[str, Any]:
    """Check manifest joins, labels, frame counts, and every Kinect sequence."""
    manifest_path = Path(manifest_path)
    keraal_root = Path(keraal_root)
    required_columns = {
        "sample_id",
        "group",
        "subject_token",
        "exercise_code",
        "kinect_frame_count",
        *PATH_COLUMNS,
        *(
            f"annotator_{annotator}_{field}"
            for annotator in ("a", "b")
            for field in LABEL_FIELDS
        ),
        *(
            f"annotator_{annotator}_temporal_error_{suffix}"
            for annotator in ("a", "b")
            for suffix in ("status", "event_count")
        ),
    }

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest_path}: missing CSV header")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{manifest_path}: missing columns {sorted(missing_columns)}")
        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"manifest has {len(rows)} rows; expected {EXPECTED_SAMPLE_COUNT}")

    sample_ids = [row["sample_id"] for row in rows]
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate sample_id values: {duplicates[:5]}")

    groups = Counter(row["group"] for row in rows)
    if dict(groups) != EXPECTED_GROUP_COUNTS:
        raise ValueError(f"unexpected group counts: {dict(groups)}")

    file_count = 0
    total_frames = 0
    frame_counts: list[int] = []
    global_agreements = 0
    global_comparisons = 0
    label_coverage: Counter[str] = Counter()

    for row in rows:
        sample_id = row["sample_id"]
        for column in PATH_COLUMNS:
            relative_path = _optional_text(row[column])
            if relative_path is None:
                raise ValueError(f"{sample_id}: missing {column}")
            _source_path(keraal_root, relative_path, sample_id=sample_id, field=column)
            file_count += 1

        for annotator in ("a", "b"):
            for field in LABEL_FIELDS:
                value = _optional_text(row[f"annotator_{annotator}_{field}"])
                _validate_label_value(value, sample_id=sample_id, field=field)
                if value is not None:
                    label_coverage[f"{annotator}.{field}"] += 1

        label_a = _optional_text(row["annotator_a_global_evaluation"])
        label_b = _optional_text(row["annotator_b_global_evaluation"])
        if label_a is not None and label_b is not None:
            global_comparisons += 1
            global_agreements += label_a == label_b

        sample = load_keraal_sample(row, keraal_root)
        count = sample["xyz_shape"][0]
        frame_counts.append(count)
        total_frames += count

    return {
        "sample_count": len(rows),
        "group_counts": dict(groups),
        "referenced_file_count": file_count,
        "total_kinect_frames": total_frames,
        "min_frames": min(frame_counts),
        "max_frames": max(frame_counts),
        "global_evaluation_agreement": (global_agreements, global_comparisons),
        "label_coverage": dict(sorted(label_coverage.items())),
    }
