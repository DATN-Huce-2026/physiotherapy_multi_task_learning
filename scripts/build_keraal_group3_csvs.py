#!/usr/bin/env python3
"""Add KERAAL group3 recordings to the manifest, target preview, and split preview.

Group3 has no annotator XML files. Its raw filename label code is preserved,
while the leading error code is mapped to shared clip-level label targets.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.keraal import load_kinect_xyz  # noqa: E402
from build_keraal_targets_preview import build_preview  # noqa: E402

EXPECTED_GROUP3_SAMPLES = 540
FILENAME_PATTERN = re.compile(
    r"^G3-Kinect-(?P<exercise>[^-]+)-(?P<participant>P\d+)"
    r"(?P<trial>T\d+)-(?P<context>[^-]+)-(?P<label>C|E[123]B[123])"
    r"-(?P<recording>\d+)\.txt$"
)
GROUP3_SPLITS = {"P1": "train", "P2": "validation", "P3": "test"}
EXTRA_MANIFEST_COLUMNS = (
    "participant_id",
    "trial_token",
    "group3_filename_context",
    "group3_label_code",
    "vicon_path",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_group3_rows(keraal_root: Path, fieldnames: list[str]) -> list[dict[str, str]]:
    group_root = keraal_root / "group3"
    kinect_dir = group_root / "kinect"
    if not kinect_dir.is_dir():
        raise ValueError(f"group3 Kinect directory not found: {kinect_dir}")

    rows: list[dict[str, str]] = []
    ids: set[str] = set()
    for kinect_file in sorted(kinect_dir.glob("*.txt")):
        match = FILENAME_PATTERN.match(kinect_file.name)
        if match is None:
            raise ValueError(f"unrecognized group3 Kinect filename: {kinect_file.name}")
        parts = match.groupdict()
        suffix = (
            f"{parts['exercise']}-{parts['participant']}{parts['trial']}-"
            f"{parts['context']}-{parts['label']}-{parts['recording']}"
        )
        sample_id = f"G3-{suffix}"
        if sample_id in ids:
            raise ValueError(f"duplicate group3 sample_id: {sample_id}")
        ids.add(sample_id)

        relative_paths = {
            "kinect_path": f"group3/kinect/{kinect_file.name}",
            "video_path": f"group3/videos/G3-Anon-{suffix}.avi",
            "openpose_path": f"group3/openpose/G3-OP-{suffix}.json",
            "blazepose_path": f"group3/blazepose/G3-BP-{suffix}.json",
            "vicon_path": f"group3/vicon/G3-Vicon-{suffix}.txt",
        }
        for field, relative_path in relative_paths.items():
            if not (keraal_root / Path(relative_path)).is_file():
                raise ValueError(f"{sample_id}: missing {field}: {relative_path}")

        frames = load_kinect_xyz(keraal_root / relative_paths["kinect_path"])
        row = {field: "" for field in fieldnames}
        row.update(
            {
                "group": "group3",
                "sample_id": sample_id,
                "exercise_code": parts["exercise"],
                "subject_token": parts["participant"],
                "participant_id": parts["participant"],
                "trial_token": parts["trial"],
                "site": "",
                "recording_token": suffix.split("-", 1)[1],
                **relative_paths,
                "group3_filename_context": parts["context"],
                "group3_label_code": parts["label"],
                "kinect_frame_count": str(len(frames)),
                "kinect_selected_for_v0_input": "true",
                "blazepose_excluded_for_now": "true",
            }
        )
        rows.append(row)

    if len(rows) != EXPECTED_GROUP3_SAMPLES:
        raise ValueError(
            f"group3 has {len(rows)} Kinect recordings; expected {EXPECTED_GROUP3_SAMPLES}"
        )
    return rows


def _build_split_rows(
    split_path: Path,
    manifest_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, existing_rows = _read_csv(split_path)
    if not {"sample_id", "subject_id", "group", "subject_token", "exercise_code", "split"}.issubset(fieldnames):
        raise ValueError(f"{split_path}: unexpected subject split schema")

    preserved_rows = [row for row in existing_rows if row["group"] != "group3"]
    new_rows = [row for row in manifest_rows if row["group"] == "group3"]
    participants = {row["subject_token"] for row in new_rows}
    if participants != set(GROUP3_SPLITS):
        raise ValueError(f"unexpected group3 participant IDs: {sorted(participants)}")

    additions = [
        {
            "sample_id": row["sample_id"],
            "subject_id": f"group3:{row['subject_token']}",
            "group": "group3",
            "subject_token": row["subject_token"],
            "exercise_code": row["exercise_code"],
            "split": GROUP3_SPLITS[row["subject_token"]],
        }
        for row in new_rows
    ]
    all_rows = preserved_rows + additions
    sample_ids = [row["sample_id"] for row in all_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{split_path}: duplicate sample IDs")
    expected_ids = {row["sample_id"] for row in manifest_rows}
    if set(sample_ids) != expected_ids:
        raise ValueError("split preview sample IDs do not match manifest")
    return fieldnames, all_rows


def build_csvs(
    keraal_root: Path,
    manifest_path: Path,
    targets_path: Path,
    split_path: Path,
) -> dict[str, Counter[str]]:
    manifest_fields, original_rows = _read_csv(manifest_path)
    base_rows = [row for row in original_rows if row["group"] != "group3"]
    base_counts = Counter(row["group"] for row in base_rows)
    if base_counts != Counter({"group1A": 250, "group2A": 51}):
        raise ValueError(f"unexpected existing manifest groups: {dict(base_counts)}")

    manifest_fields = list(dict.fromkeys([*manifest_fields, *EXTRA_MANIFEST_COLUMNS]))
    group3_rows = _build_group3_rows(keraal_root.resolve(strict=True), manifest_fields)
    manifest_rows = base_rows + group3_rows
    all_ids = [row["sample_id"] for row in manifest_rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("manifest would contain duplicate sample IDs")

    split_fields, split_rows = _build_split_rows(split_path, manifest_rows)
    manifest_counts = Counter(row["group"] for row in manifest_rows)
    if manifest_counts != Counter({"group1A": 250, "group2A": 51, "group3": 540}):
        raise ValueError(f"unexpected final manifest groups: {dict(manifest_counts)}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=manifest_path.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_manifest = temp_dir / manifest_path.name
        temp_targets = temp_dir / targets_path.name
        temp_split = temp_dir / split_path.name
        _write_csv(temp_manifest, manifest_fields, manifest_rows)
        _write_csv(temp_split, split_fields, split_rows)
        label_counts = build_preview(temp_manifest, temp_targets)
        os.replace(temp_manifest, manifest_path)
        os.replace(temp_targets, targets_path)
        os.replace(temp_split, split_path)
    return label_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keraal-root",
        type=Path,
        default=Path.home() / "Downloads" / "KERAAL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_manifest.csv",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_targets_preview.csv",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_subject_split_preview.csv",
    )
    args = parser.parse_args()

    try:
        label_counts = build_csvs(
            args.keraal_root,
            args.manifest,
            args.targets,
            args.split,
        )
    except (OSError, ValueError) as exc:
        print(f"Could not add KERAAL group3: {exc}", file=sys.stderr)
        return 1

    print("Updated manifest, target preview, and subject split preview (841 recordings).")
    print("Group3 source label codes were preserved; shared targets use the code prefix:")
    for label_code, count in sorted(label_counts["group3_label_code"].items()):
        print(f"  {label_code}: {count}")
    class_counts = label_counts["exercise_error_class_values"]
    usable_class_count = sum(class_counts.values())
    print(f"Shared four-class targets: {usable_class_count}/841 usable.")
    for label in ("Correct", "Error1", "Error2", "Error3"):
        print(f"  {label}: {class_counts[label]}")
    print("Group3 participant split: P1=train, P2=validation, P3=test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
