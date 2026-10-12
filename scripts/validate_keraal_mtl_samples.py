#!/usr/bin/env python3
"""Check joins between KERAAL Kinect files and the proposed target preview."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.keraal import (  # noqa: E402
    EXPECTED_SAMPLE_COUNT,
    MTL_TARGET_FIELDS,
    load_keraal_mtl_sample,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"{path}: missing CSV header or sample_id column")
        rows = list(reader)
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path}: duplicate sample_id values")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--keraal-root",
        type=Path,
        default=Path.home() / "Documents" / "KERAAL",
    )
    parser.add_argument(
        "--relative-to-initial-spine-base",
        action="store_true",
        help="apply the proposed coordinate offset while loading each sample",
    )
    args = parser.parse_args()

    try:
        manifest_rows = _read_rows(args.manifest)
        target_rows = _read_rows(args.targets)
        if len(manifest_rows) != EXPECTED_SAMPLE_COUNT:
            raise ValueError(
                f"manifest has {len(manifest_rows)} samples; "
                f"expected {EXPECTED_SAMPLE_COUNT}"
            )
        manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
        targets_by_id = {row["sample_id"]: row for row in target_rows}
        missing_targets = manifest_by_id.keys() - targets_by_id.keys()
        extra_targets = targets_by_id.keys() - manifest_by_id.keys()
        if missing_targets or extra_targets:
            raise ValueError(
                "sample_id sets differ: "
                f"missing targets={sorted(missing_targets)[:5]}, "
                f"extra targets={sorted(extra_targets)[:5]}"
            )

        labeled_counts: Counter[str] = Counter()
        for sample_id, manifest_row in manifest_by_id.items():
            sample = load_keraal_mtl_sample(
                manifest_row,
                targets_by_id[sample_id],
                args.keraal_root,
                relative_to_initial_spine_base=args.relative_to_initial_spine_base,
            )
            if sample["sample_id"] != sample_id:
                raise ValueError(f"loader returned wrong sample_id for {sample_id}")
            for field in MTL_TARGET_FIELDS:
                if sample["target_masks"][field]:
                    labeled_counts[field] += 1

        transform_name = (
            "initial SpineBase offset"
            if args.relative_to_initial_spine_base
            else "raw XYZ"
        )
    except (OSError, ValueError) as exc:
        print(f"MTL sample validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Joined and loaded samples: {len(manifest_rows)}")
    print(f"Input coordinates: {transform_name}")
    print("Samples with a target per head:")
    for field in MTL_TARGET_FIELDS:
        print(f"  {field}: {labeled_counts[field]}/{len(manifest_rows)}")
    print("Join and target-mask checks passed. Targets remain provisional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
