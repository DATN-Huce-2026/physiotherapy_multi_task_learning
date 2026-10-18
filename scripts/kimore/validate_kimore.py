#!/usr/bin/env python3
"""Validate the KiMoRe manifest, targets, splits, and Kinect skeleton files without modifying source data."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.kimore import validate_kimore_manifest  # noqa: E402

if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_manifest.csv",
        help="Path to kimore_manifest.csv (default: data/kimore/kimore_manifest.csv)",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_targets_preview.csv",
        help="Path to kimore_targets_preview.csv (default: data/kimore/kimore_targets_preview.csv)",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_subject_split_preview.csv",
        help="Path to kimore_subject_split_preview.csv (default: data/kimore/kimore_subject_split_preview.csv)",
    )
    parser.add_argument(
        "--kimore-root",
        type=Path,
        default=Path(r"D:\DATN\KIMORE"),
        help="Root directory of the KiMoRe dataset (e.g. /content/KIMORE or D:\\DATN\\KIMORE)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Perform a quick check on the first 5 samples instead of all samples",
    )
    args = parser.parse_args()

    sample_limit = 5 if args.quick else None

    try:
        summary = validate_kimore_manifest(
            args.manifest,
            args.kimore_root,
            targets_path=args.targets if args.targets.is_file() else None,
            split_path=args.split if args.split.is_file() else None,
            check_all_frames=True,
            sample_limit=sample_limit,
        )
    except (OSError, ValueError) as exc:
        print(f"KiMoRe validation failed: {exc}", file=sys.stderr)
        return 1

    print("============================================================")
    print("KiMoRe Dataset Validation Summary")
    print("============================================================")
    print(f"Total manifest samples: {summary['sample_count']}")
    print(f"Status breakdown: {summary['status_counts']}")
    print(f"Valid clips for training: {summary['valid_sample_count']}")
    print(f"Cohorts: {summary['cohort_counts']}")
    print(f"Groups: {summary['group_counts']}")
    print(f"Exercises: {summary['exercise_counts']}")
    print(
        f"Kinect frames parsed: {summary['total_kinect_frames']} "
        f"(min={summary['min_frames']}, max={summary['max_frames']}, avg={summary['avg_frames']:.1f} per clip)"
    )
    q_stats = summary["quality_stats"]
    if q_stats["count"] > 0:
        print(
            f"Quality target range (clinical TS / 50.0): "
            f"min={q_stats['min']:.4f}, max={q_stats['max']:.4f}, avg={q_stats['mean']:.4f} "
            f"({q_stats['count']} samples)"
        )
    if summary["targets_checked"]:
        print("MTL targets preview: join and format verified")
    if summary["split_counts"]:
        print(f"Subject-disjoint split: {summary['split_counts']} clips")
        if summary["subject_split_counts"]:
            print(f"Subject count per split: {summary['subject_split_counts']}")
    if args.quick:
        print("Note: Quick mode enabled (--quick); only first 5 samples were parsed.")
    print("Validation passed; source data was read only.")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
