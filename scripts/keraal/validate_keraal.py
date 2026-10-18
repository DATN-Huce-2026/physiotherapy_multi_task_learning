#!/usr/bin/env python3
"""Validate the KERAAL manifest and Kinect parser without changing source data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.keraal import validate_keraal_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal" / "keraal_manifest.csv",
    )
    parser.add_argument(
        "--keraal-root",
        type=Path,
        default=Path.home() / "Downloads" / "KERAAL",
    )
    args = parser.parse_args()

    try:
        summary = validate_keraal_manifest(args.manifest, args.keraal_root)
    except (OSError, ValueError) as exc:
        print(f"KERAAL validation failed: {exc}", file=sys.stderr)
        return 1

    agree, compared = summary["global_evaluation_agreement"]
    print(f"Samples: {summary['sample_count']} ({summary['group_counts']})")
    print(
        f"Referenced source files: {summary['referenced_file_count']}; "
        f"Kinect frames: {summary['total_kinect_frames']} "
        f"({summary['min_frames']}–{summary['max_frames']} per clip)"
    )
    print(f"Global Evaluation coder agreement: {agree}/{compared}")
    print("Label values present:")
    for label, count in summary["label_coverage"].items():
        print(f"  {label}: {count}")
    print("Validation passed; source data was read only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
