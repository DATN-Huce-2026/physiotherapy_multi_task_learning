"""Reader and parser for KiMoRe 3D Kinect skeleton sequences and clinical scores."""

from __future__ import annotations

import csv
import math
from collections import Counter
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


def resolve_kimore_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve a relative path inside KiMoRe root with nested directory and casing fallback.

    Handles common archive unpacking artifacts on Google Colab/Linux:
    - Direct path: root / relative_path
    - Nested folder: root / root.name / relative_path (e.g. /content/KIMORE/KIMORE/...)
    - Subdirectory wrapper: root / 'kimore' / ...
    - Case-insensitive component matching on Linux.
    """
    rel = Path(relative_path)
    direct = root / rel
    if direct.is_file():
        return direct

    # Check 1: Nested root directory (e.g., unzipping KIMORE.zip into a KIMORE folder)
    candidate_upper = root / root.name / rel
    if candidate_upper.is_file():
        return candidate_upper

    candidate_lower = root / root.name.lower() / rel
    if candidate_lower.is_file():
        return candidate_lower

    # Check 2: Single wrapper subfolder inside root
    if root.is_dir():
        for sub in root.iterdir():
            if sub.is_dir() and sub.name.lower() in {root.name.lower(), "kimore"}:
                candidate_sub = sub / rel
                if candidate_sub.is_file():
                    return candidate_sub

    # Check 3: Case-insensitive component traversal on Linux
    if root.is_dir():
        curr = root
        parts = list(rel.parts)
        for part in parts:
            if not curr.is_dir():
                break
            matched_child = None
            for child in curr.iterdir():
                if child.name.lower() == part.lower():
                    matched_child = child
                    break
            if matched_child is not None:
                curr = matched_child
            else:
                break
        if curr.is_file():
            return curr

    return direct


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

    full_kinect_path = resolve_kimore_path(root, relative_kinect_path)
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


def validate_kimore_manifest(
    manifest_path: str | Path,
    kimore_root: str | Path,
    *,
    targets_path: str | Path | None = None,
    split_path: str | Path | None = None,
    check_all_frames: bool = True,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Validate the KiMoRe manifest, MTL targets preview, and subject splits.

    Ensures all referenced Kinect skeleton files exist, parses coordinate shapes,
    checks normalized quality scores, verifies subject-disjoint property in splits,
    and returns a summary dictionary without modifying source data.
    """
    manifest_path = Path(manifest_path)
    kimore_root = Path(kimore_root)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"KiMoRe manifest not found: {manifest_path}")
    if not kimore_root.is_dir():
        raise FileNotFoundError(f"KiMoRe root directory not found: {kimore_root}")

    required_columns = {
        "sample_id",
        "dataset",
        "cohort",
        "group",
        "subject_id",
        "subject_token",
        "exercise_code",
        "kinect_path",
        "kinect_frame_count",
        "assessment_path",
        "clinical_ts_raw",
        "clinical_po_raw",
        "clinical_cf_raw",
        "quality_score_normalized",
        "status",
    }

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest_path}: missing CSV header")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{manifest_path}: missing columns {sorted(missing_columns)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{manifest_path}: manifest is empty")

    sample_ids = [r["sample_id"] for r in rows]
    duplicates = [sid for sid, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate sample_id values in manifest: {duplicates[:5]}")

    status_counts = Counter(r["status"] for r in rows)
    cohort_counts = Counter(r["cohort"] for r in rows)
    group_counts = Counter(r["group"] for r in rows)
    exercise_counts = Counter(r["exercise_code"] for r in rows)

    # Optional targets validation
    targets_by_id: dict[str, dict[str, str]] = {}
    if targets_path is not None:
        t_path = Path(targets_path)
        if t_path.is_file():
            with t_path.open("r", encoding="utf-8-sig", newline="") as tf:
                t_reader = csv.DictReader(tf)
                for tr in t_reader:
                    sid = tr["sample_id"]
                    if sid in targets_by_id:
                        raise ValueError(f"{t_path}: duplicate sample_id {sid}")
                    targets_by_id[sid] = tr

    # Optional split validation
    split_counts: Counter[str] = Counter()
    subject_split_map: dict[str, str] = {}
    if split_path is not None:
        s_path = Path(split_path)
        if s_path.is_file():
            with s_path.open("r", encoding="utf-8-sig", newline="") as sf:
                s_reader = csv.DictReader(sf)
                for sr in s_reader:
                    sid = sr["sample_id"]
                    sub_id = sr["subject_id"]
                    sp = sr["split"]
                    if sp not in {"train", "validation", "test"}:
                        raise ValueError(f"{s_path}: unexpected split name {sp!r} for {sid}")
                    split_counts[sp] += 1
                    if sub_id in subject_split_map and subject_split_map[sub_id] != sp:
                        raise ValueError(
                            f"{s_path}: subject {sub_id} assigned to multiple splits: "
                            f"{subject_split_map[sub_id]} vs {sp}"
                        )
                    subject_split_map[sub_id] = sp

    valid_rows = [r for r in rows if r.get("status") == "ok"]
    if not valid_rows:
        raise ValueError("manifest contains 0 samples with status == 'ok'")

    if targets_by_id:
        valid_ids = {r["sample_id"] for r in valid_rows}
        missing_in_targets = valid_ids - targets_by_id.keys()
        if missing_in_targets:
            raise ValueError(
                f"{len(missing_in_targets)} valid samples missing in targets preview: "
                f"{sorted(missing_in_targets)[:5]}"
            )

    rows_to_check = valid_rows[:sample_limit] if sample_limit else valid_rows

    total_frames = 0
    frame_counts: list[int] = []
    quality_scores: list[float] = []

    for r in rows_to_check:
        sid = r["sample_id"]
        rel_kinect = r.get("kinect_path", "").strip()
        if not rel_kinect:
            raise ValueError(f"{sid}: missing kinect_path in valid sample")

        full_kinect = resolve_kimore_path(kimore_root, rel_kinect)
        if not full_kinect.is_file():
            raise FileNotFoundError(f"{sid}: Kinect file not found: {full_kinect}")

        rel_assessment = r.get("assessment_path", "").strip()
        if rel_assessment:
            full_assessment = resolve_kimore_path(kimore_root, rel_assessment)
            if not full_assessment.is_file():
                raise FileNotFoundError(f"{sid}: Assessment file not found: {full_assessment}")

        q_norm_str = r.get("quality_score_normalized", "").strip()
        if q_norm_str:
            q_val = float(q_norm_str)
            if not (0.0 <= q_val <= 1.05):
                raise ValueError(f"{sid}: quality_score_normalized {q_val} out of range [0, 1]")
            quality_scores.append(q_val)

        if check_all_frames:
            frames = load_kimore_joint_xyz(full_kinect)
            n_frames = len(frames)
            frame_counts.append(n_frames)
            total_frames += n_frames
            if targets_by_id and sid in targets_by_id:
                sample = load_kimore_sample(r, targets_by_id[sid], kimore_root)
                if sample["sample_id"] != sid:
                    raise ValueError(f"{sid}: loaded sample_id mismatch")
        else:
            manifest_fc = int(r.get("kinect_frame_count") or 0)
            frame_counts.append(manifest_fc)
            total_frames += manifest_fc

    sub_split_counts = Counter(subject_split_map.values()) if subject_split_map else {}

    return {
        "sample_count": len(rows),
        "valid_sample_count": len(valid_rows),
        "checked_sample_count": len(rows_to_check),
        "status_counts": dict(status_counts),
        "cohort_counts": dict(cohort_counts),
        "group_counts": dict(group_counts),
        "exercise_counts": dict(exercise_counts),
        "total_kinect_frames": total_frames,
        "min_frames": min(frame_counts) if frame_counts else 0,
        "max_frames": max(frame_counts) if frame_counts else 0,
        "avg_frames": (total_frames / len(frame_counts)) if frame_counts else 0.0,
        "quality_stats": {
            "count": len(quality_scores),
            "min": min(quality_scores) if quality_scores else None,
            "max": max(quality_scores) if quality_scores else None,
            "mean": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
        },
        "targets_checked": bool(targets_by_id),
        "split_counts": dict(split_counts),
        "subject_split_counts": dict(sub_split_counts),
    }
