#!/usr/bin/env python3
"""Preview root-offset coordinates for one KERAAL Kinect clip in memory only."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.keraal import (  # noqa: E402
    load_kinect_xyz,
    offset_to_initial_spine_base,
)


def _axis_ranges(frames: list[list[tuple[float, float, float]]]) -> dict[str, tuple[float, float]]:
    values = {axis: [] for axis in range(3)}
    for frame in frames:
        for joint in frame:
            for axis, value in enumerate(joint):
                values[axis].append(value)
    return {
        axis_name: (min(values[index]), max(values[index]))
        for index, axis_name in enumerate("xyz")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_manifest.csv",
    )
    parser.add_argument(
        "--keraal-root",
        type=Path,
        default=Path.home() / "Documents" / "KERAAL",
    )
    parser.add_argument("--sample-id", default="G1A-CTK-R1-Brest-038")
    args = parser.parse_args()

    try:
        with args.manifest.open("r", encoding="utf-8-sig", newline="") as source:
            row = next(
                (item for item in csv.DictReader(source) if item["sample_id"] == args.sample_id),
                None,
            )
        if row is None:
            raise ValueError(f"sample_id not found in manifest: {args.sample_id}")
        raw = load_kinect_xyz(
            args.keraal_root / row["kinect_path"],
            expected_frame_count=int(row["kinect_frame_count"]),
        )
        transformed = offset_to_initial_spine_base(raw)
    except (OSError, StopIteration, ValueError) as exc:
        print(f"Could not preview sample: {exc}", file=sys.stderr)
        return 1

    print(f"Sample: {args.sample_id}")
    print(f"Shape before and after: ({len(raw)}, 25, 3)")
    print(f"First-frame SpineBase origin: {raw[0][0]}")
    print(f"SpineBase at frame 0 after transform: {transformed[0][0]}")
    if len(raw) > 1:
        print(f"SpineBase at frame 1 before: {raw[1][0]}")
        print(f"SpineBase at frame 1 after:  {transformed[1][0]}")
    print(f"Raw XYZ ranges: {_axis_ranges(raw)}")
    print(f"Offset XYZ ranges: {_axis_ranges(transformed)}")
    print("Preview only: no file was written and the KERAAL source was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
