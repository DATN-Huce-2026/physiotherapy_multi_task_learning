#!/usr/bin/env python3
"""Build manifest, multi-task targets, and subject-disjoint splits for KiMoRe.

This script scans the KiMoRe dataset folder, extracts 3D skeleton frame counts from
`Raw/JointPosition*.csv`, parses clinical scores (TS, PO, CF) from
`Label/ClinicalAssessment*.xlsx`, normalizes the Total Score (TS) to [0.0, 1.0],
creates stratified subject-disjoint train/val/test splits, and outputs 3 CSV files
fully compatible with the multi-task learning pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KIMORE_ROOT = Path(r"D:\DATN\KIMORE")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "kimore"

EXERCISE_KERAAL_EQUIVALENTS = {
    "Es1": "ELK",  # Bilateral arm lifting (vươn/nâng hai tay)
    "Es2": "CTK",  # Lateral trunk tilt (nghiêng thân sang bên)
    "Es3": "RTK",  # Trunk rotation (xoay thân)
    "Es4": None,   # Pelvis rotation
    "Es5": None,   # Squat
}


def parse_assessment_xlsx(xlsx_path: Path) -> dict[int, dict[str, float]]:
    """Parse KiMoRe clinical assessment scores (TS, PO, CF) for exercises 1..5.

    Uses Python's standard zipfile and xml.etree so no external packages (openpyxl) are needed.
    Returns:
        dict: {ex_num: {"ts": float, "po": float, "cf": float}}
    """
    if not xlsx_path.is_file():
        return {}

    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
            for si in tree.findall(f"{ns}si"):
                parts = [t.text for t in si.findall(f".//{ns}t") if t.text]
                shared_strings.append("".join(parts))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in z.namelist():
            return {}

        tree = ET.fromstring(z.read(sheet_name))
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        rows: list[dict[str, str]] = []
        for r in tree.findall(f"{ns}sheetData/{ns}row"):
            row_dict: dict[str, str] = {}
            for c in r.findall(f"{ns}c"):
                r_ref = c.get("r", "")
                col_letter = "".join([ch for ch in r_ref if ch.isalpha()])
                t_attr = c.get("t")
                v = c.find(f"{ns}v")
                raw_text = v.text if (v is not None and v.text is not None) else ""
                if not raw_text and t_attr == "inlineStr":
                    t_el = c.find(f"{ns}is/{ns}t")
                    raw_text = t_el.text if (t_el is not None and t_el.text is not None) else ""
                val = raw_text.strip()
                if t_attr == "s" and val.isdigit() and int(val) < len(shared_strings):
                    val = shared_strings[int(val)]
                row_dict[col_letter] = val
            if any(row_dict.values()):
                rows.append(row_dict)

    if len(rows) < 2:
        return {}

    header = rows[0]
    data = rows[1]

    col_map: dict[str, tuple[str, int]] = {}
    for col, h in header.items():
        clean_h = h.lower().replace(" ", "").replace("_", "")
        # match clinical ts ex#1 or ts ex1
        match = re.search(r"(ts|po|cf).*?ex#?(\d+)", clean_h)
        if match:
            metric_type = match.group(1)
            ex_num = int(match.group(2))
            col_map[col] = (metric_type, ex_num)

    results: dict[int, dict[str, float]] = defaultdict(dict)
    for col, (metric_type, ex_num) in col_map.items():
        raw_val = data.get(col, "").strip()
        if raw_val:
            try:
                results[ex_num][metric_type] = float(raw_val)
            except ValueError:
                pass

    return dict(results)


def count_joint_frames(csv_path: Path) -> int:
    """Count non-empty frames in a JointPosition CSV file."""
    if not csv_path.is_file():
        return 0
    count = 0
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def scan_kimore_dataset(kimore_root: Path) -> list[dict[str, Any]]:
    """Scan the KiMoRe dataset directory and extract samples, frame counts, and scores."""
    samples: list[dict[str, Any]] = []

    # Valid cohorts in KiMoRe
    known_cohorts = {"CG", "GPP"}

    for cohort_dir in sorted(kimore_root.iterdir()):
        if not cohort_dir.is_dir() or cohort_dir.name not in known_cohorts:
            continue
        cohort = cohort_dir.name

        for group_dir in sorted(cohort_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group_name = group_dir.name  # Expert, NotExpert, BackPain, Parkinson, Stroke

            for subject_dir in sorted(group_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue
                subject_token = subject_dir.name  # E_ID1, NE_ID1, B_ID1, ...

                for ex_dir in sorted(subject_dir.iterdir()):
                    if not ex_dir.is_dir() or not ex_dir.name.startswith("Es"):
                        continue
                    ex_code = ex_dir.name  # Es1, Es2, Es3, ...
                    match_num = re.search(r"\d+", ex_code)
                    ex_num = int(match_num.group(0)) if match_num else 1

                    # Look for JointPosition
                    raw_dir = ex_dir / "Raw"
                    joint_files = sorted(raw_dir.glob("JointPosition*.csv")) if raw_dir.is_dir() else []
                    joint_path = joint_files[0] if joint_files else None

                    # Look for ClinicalAssessment
                    label_dir = ex_dir / "Label"
                    assessment_files = sorted(label_dir.glob("ClinicalAssessment*.xlsx")) if label_dir.is_dir() else []
                    assessment_path = assessment_files[0] if assessment_files else None

                    # Parse clinical assessment
                    scores: dict[str, float] = {}
                    if assessment_path is not None:
                        all_scores = parse_assessment_xlsx(assessment_path)
                        scores = all_scores.get(ex_num, {})

                    # Count frames
                    frame_count = count_joint_frames(joint_path) if joint_path is not None else 0

                    ts_raw = scores.get("ts")
                    po_raw = scores.get("po")
                    cf_raw = scores.get("cf")

                    ts_norm = (ts_raw / 50.0) if ts_raw is not None else None

                    sample_id = f"KMR-{cohort}-{group_name}-{subject_token}-{ex_code}"
                    status = "ok"
                    if joint_path is None or frame_count == 0:
                        status = "missing_joint_position"
                    elif ts_raw is None:
                        status = "missing_clinical_score"

                    samples.append({
                        "sample_id": sample_id,
                        "dataset": "kimore",
                        "cohort": cohort,
                        "group": group_name,
                        "subject_token": subject_token,
                        "subject_id": f"kimore:{cohort}:{group_name}:{subject_token}",
                        "exercise_code": ex_code,
                        "exercise_num": ex_num,
                        "kinect_path": joint_path.relative_to(kimore_root).as_posix() if joint_path else "",
                        "kinect_frame_count": frame_count,
                        "assessment_path": assessment_path.relative_to(kimore_root).as_posix() if assessment_path else "",
                        "clinical_ts_raw": ts_raw,
                        "clinical_po_raw": po_raw,
                        "clinical_cf_raw": cf_raw,
                        "quality_score_normalized": ts_norm,
                        "status": status,
                    })

    return samples


def split_subjects_stratified(
    samples: list[dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, str]:
    """Perform a strictly subject-disjoint stratified train/val/test split across cohorts/groups.

    Returns:
        dict: {subject_id: 'train' | 'validation' | 'test'}
    """
    rng = random.Random(seed)
    # Group subjects by their group category (Expert, NotExpert, BackPain, Parkinson, Stroke)
    subjects_by_group: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        subjects_by_group[s["group"]].add(s["subject_id"])

    subject_split_map: dict[str, str] = {}

    for group_name, sub_set in sorted(subjects_by_group.items()):
        sub_list = sorted(sub_set)
        rng.shuffle(sub_list)
        n = len(sub_list)

        n_test = max(1, round(n * test_ratio)) if n >= 3 else 1
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        n_train = n - n_test - n_val
        if n_train <= 0:
            n_train = 1
            n_test = max(0, n - 1)

        train_subs = sub_list[:n_train]
        val_subs = sub_list[n_train:n_train + n_val]
        test_subs = sub_list[n_train + n_val:]

        for sid in train_subs:
            subject_split_map[sid] = "train"
        for sid in val_subs:
            subject_split_map[sid] = "validation"
        for sid in test_subs:
            subject_split_map[sid] = "test"

    return subject_split_map


def write_csv_file(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write list of row dicts to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manifest, MTL targets preview, and subject split preview for KiMoRe."
    )
    parser.add_argument(
        "--kimore-root",
        type=Path,
        default=DEFAULT_KIMORE_ROOT,
        help=f"Root directory of KiMoRe dataset (default: {DEFAULT_KIMORE_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Primary directory to save CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject-disjoint split (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )

    args = parser.parse_args()

    kimore_root: Path = args.kimore_root
    output_dir: Path = args.output_dir

    if not kimore_root.is_dir():
        print(f"Error: KiMoRe root directory not found: {kimore_root}", file=sys.stderr)
        sys.exit(1)

    print(f"--> Scanning KiMoRe dataset at: {kimore_root}")
    samples = scan_kimore_dataset(kimore_root)
    print(f"--> Total clips discovered: {len(samples)}")

    valid_samples = [s for s in samples if s["status"] == "ok"]
    missing_samples = [s for s in samples if s["status"] != "ok"]

    print(f"    - Valid clips: {len(valid_samples)}")
    if missing_samples:
        print(f"    - Incomplete/missing clips: {len(missing_samples)}")
        for m in missing_samples:
            print(f"      * {m['sample_id']}: status={m['status']}")

    # Stratified subject-disjoint split
    subject_split_map = split_subjects_stratified(
        valid_samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # 1. Build Manifest Rows
    manifest_fields = [
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
    ]
    manifest_rows = []
    for s in samples:
        row = dict(s)
        row["clinical_ts_raw"] = f"{s['clinical_ts_raw']:.4f}" if s["clinical_ts_raw"] is not None else ""
        row["clinical_po_raw"] = f"{s['clinical_po_raw']:.4f}" if s["clinical_po_raw"] is not None else ""
        row["clinical_cf_raw"] = f"{s['clinical_cf_raw']:.4f}" if s["clinical_cf_raw"] is not None else ""
        row["quality_score_normalized"] = (
            f"{s['quality_score_normalized']:.4f}" if s["quality_score_normalized"] is not None else ""
        )
        manifest_rows.append(row)

    # 2. Build Targets Preview Rows (Only for valid samples)
    targets_fields = [
        "sample_id",
        "dataset",
        "group",
        "subject_token",
        "exercise_code",
        "kinect_frame_count",
        "quality_target",
        "quality_mask",
        "quality_resolution",
        "clinical_ts_raw",
        "clinical_po_raw",
        "clinical_cf_raw",
        "exercise_target",
        "exercise_mask",
        "exercise_resolution",
        "exercise_error_class_target",
        "exercise_error_class_mask",
        "exercise_error_class_resolution",
        "global_evaluation_target",
        "global_evaluation_mask",
        "global_evaluation_resolution",
        "error_type_target",
        "error_type_mask",
        "error_type_resolution",
        "severity_target",
        "severity_mask",
        "severity_resolution",
        "body_part_target",
        "body_part_mask",
        "body_part_resolution",
        "review_flags",
    ]
    targets_rows = []
    for s in valid_samples:
        ts_norm = s["quality_score_normalized"]
        has_quality = ts_norm is not None
        targets_rows.append({
            "sample_id": s["sample_id"],
            "dataset": s["dataset"],
            "group": s["group"],
            "subject_token": s["subject_token"],
            "exercise_code": s["exercise_code"],
            "kinect_frame_count": s["kinect_frame_count"],
            "quality_target": f"{ts_norm:.4f}" if has_quality else "",
            "quality_mask": "true" if has_quality else "false",
            "quality_resolution": "clinical_ts_div_50" if has_quality else "unlabeled",
            "clinical_ts_raw": f"{s['clinical_ts_raw']:.4f}" if s["clinical_ts_raw"] is not None else "",
            "clinical_po_raw": f"{s['clinical_po_raw']:.4f}" if s["clinical_po_raw"] is not None else "",
            "clinical_cf_raw": f"{s['clinical_cf_raw']:.4f}" if s["clinical_cf_raw"] is not None else "",
            "exercise_target": s["exercise_code"],
            "exercise_mask": "true",
            "exercise_resolution": "exercise_folder",
            "exercise_error_class_target": "",
            "exercise_error_class_mask": "false",
            "exercise_error_class_resolution": "not_applicable",
            "global_evaluation_target": "",
            "global_evaluation_mask": "false",
            "global_evaluation_resolution": "not_applicable",
            "error_type_target": "",
            "error_type_mask": "false",
            "error_type_resolution": "not_applicable",
            "severity_target": "",
            "severity_mask": "false",
            "severity_resolution": "not_applicable",
            "body_part_target": "",
            "body_part_mask": "false",
            "body_part_resolution": "not_applicable",
            "review_flags": "",
        })

    # 3. Build Subject Split Preview Rows (Matches keraal_subject_split_preview.csv schema)
    split_fields = [
        "sample_id",
        "subject_id",
        "group",
        "subject_token",
        "exercise_code",
        "split",
    ]
    split_rows = []
    for s in valid_samples:
        sid = s["subject_id"]
        split_name = subject_split_map.get(sid, "train")
        split_rows.append({
            "sample_id": s["sample_id"],
            "subject_id": sid,
            "group": s["group"],
            "subject_token": s["subject_token"],
            "exercise_code": s["exercise_code"],
            "split": split_name,
        })

    # Write files to output_dir
    manifest_path = output_dir / "kimore_manifest.csv"
    targets_path = output_dir / "kimore_targets_preview.csv"
    split_path = output_dir / "kimore_subject_split_preview.csv"

    write_csv_file(manifest_path, manifest_fields, manifest_rows)
    write_csv_file(targets_path, targets_fields, targets_rows)
    write_csv_file(split_path, split_fields, split_rows)

    print(f"\n--> Successfully generated 3 CSV files in: {output_dir}")
    print(f"    1. Manifest:     {manifest_path.name} ({len(manifest_rows)} rows)")
    print(f"    2. Targets:      {targets_path.name} ({len(targets_rows)} rows)")
    print(f"    3. Split:        {split_path.name} ({len(split_rows)} rows)")

    # Print split statistics
    split_counts = Counter(r["split"] for r in split_rows)
    sub_split_counts = Counter(subject_split_map.values())
    print("\n--> Split Breakdown (Subject-Disjoint):")
    for sp in ["train", "validation", "test"]:
        print(
            f"    - {sp.upper():10}: {sub_split_counts[sp]:2} subjects, "
            f"{split_counts[sp]:3} clips ({split_counts[sp] / len(split_rows) * 100:.1f}%)"
        )

if __name__ == "__main__":
    main()
