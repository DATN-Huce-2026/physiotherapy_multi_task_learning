#!/usr/bin/env python3
"""Build a review-only CSV of proposed KERAAL clip-level targets.

This script reads the manifest and writes a derived preview. It never edits
the manifest or any KERAAL source file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SAMPLES = 301
ANNOTATORS = ("a", "b")
BODY_PART_ORDER = ("Spine", "BothArms", "LeftArm", "RightArm")
LABEL_VALUES = {
    "global_evaluation": {"Correct", "Incorrect", "Incomplete", "Motionless"},
    "error_type": {"Error1", "Error2", "Error3", "Other", "none"},
    "severity": {"SmallError", "LargeError"},
    "body_part": set(BODY_PART_ORDER),
}

LABEL_FIELDS = ("global_evaluation", "error_type", "severity", "body_part")


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _normalize(field: str, value: str | None) -> str | None:
    """Normalize values for comparison while preserving multi-part labels."""
    if value is None:
        return None
    if field != "body_part":
        if value not in LABEL_VALUES[field]:
            raise ValueError(f"unknown {field} value: {value!r}")
        return value

    parts = {part.strip() for part in value.split("|") if part.strip()}
    order = {part: index for index, part in enumerate(BODY_PART_ORDER)}
    unknown = parts.difference(order)
    if unknown:
        raise ValueError(f"unknown body-part value(s): {sorted(unknown)}")
    if not parts:
        raise ValueError("body_part contains no labels")
    return "|".join(sorted(parts, key=order.__getitem__))


def _resolve(field: str, value_a: str | None, value_b: str | None) -> tuple[str, bool, str]:
    """Resolve one head without converting missing labels into negatives."""
    value_a = _normalize(field, _clean(value_a))
    value_b = _normalize(field, _clean(value_b))

    if value_a is not None and value_b is not None:
        if value_a == value_b:
            return value_a, True, "both_agree"
        return "", False, "disagreement"
    if value_a is not None:
        return value_a, True, "annotator_a_only"
    if value_b is not None:
        return value_b, True, "annotator_b_only"
    return "", False, "unlabeled"


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {
            "sample_id",
            "group",
            "subject_token",
            "exercise_code",
            "kinect_frame_count",
            *(f"annotator_{coder}_{field}" for coder in ANNOTATORS for field in LABEL_FIELDS),
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader.fieldnames), list(reader)


def build_preview(manifest: Path, output: Path) -> dict[str, Counter[str]]:
    fieldnames, rows = _read_manifest(manifest)
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"manifest has {len(rows)} rows; expected {EXPECTED_SAMPLES}")

    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate sample_id values")

    columns = [
        "sample_id",
        "group",
        "subject_token",
        "exercise_code",
        "kinect_frame_count",
        "exercise_target",
        "exercise_mask",
        "exercise_resolution",
    ]
    for field in LABEL_FIELDS:
        columns.extend(
            (
                f"annotator_a_{field}",
                f"annotator_b_{field}",
                f"{field}_target",
                f"{field}_mask",
                f"{field}_resolution",
            )
        )
    columns.append("review_flags")
    counters: dict[str, Counter[str]] = {"exercise": Counter()}
    counters.update({field: Counter() for field in LABEL_FIELDS})
    counters["review"] = Counter()
    exercise_codes = {row["exercise_code"].strip() for row in rows}
    if not exercise_codes or "" in exercise_codes:
        raise ValueError("manifest contains a missing exercise_code")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = {column: row.get(column, "") for column in columns[:5]}
            record.update(
                {
                    "exercise_target": row["exercise_code"].strip(),
                    "exercise_mask": "true",
                    "exercise_resolution": "manifest_metadata",
                }
            )
            counters["exercise"]["labeled"] += 1

            for field in LABEL_FIELDS:
                column_a = f"annotator_a_{field}"
                column_b = f"annotator_b_{field}"
                value_a = _clean(row.get(column_a))
                value_b = _clean(row.get(column_b))
                target, mask, resolution = _resolve(field, value_a, value_b)
                record[column_a] = value_a or ""
                record[column_b] = value_b or ""
                record[f"{field}_target"] = target
                record[f"{field}_mask"] = str(mask).lower()
                record[f"{field}_resolution"] = resolution
                counters[field][resolution] += 1

            review_flags: list[str] = []
            for coder in ANNOTATORS:
                global_value = _clean(row.get(f"annotator_{coder}_global_evaluation"))
                error_value = _clean(row.get(f"annotator_{coder}_error_type"))
                body_value = _clean(row.get(f"annotator_{coder}_body_part"))
                if global_value in {"Incomplete", "Motionless"} and error_value == "none":
                    review_flags.append(f"{coder}:incomplete_or_motionless_with_none")
                if error_value in {"Error1", "Error2", "Error3", "Other"} and not body_value:
                    review_flags.append(f"{coder}:error_type_without_body_part")
                if body_value and "|" in body_value:
                    review_flags.append(f"{coder}:compound_body_part")

            part_a = _normalize("body_part", _clean(row.get("annotator_a_body_part")))
            part_b = _normalize("body_part", _clean(row.get("annotator_b_body_part")))
            if {part_a, part_b} == {"BothArms", "LeftArm|RightArm"}:
                review_flags.append("body_part_both_arms_equivalence_to_review")
            record["review_flags"] = ";".join(review_flags)
            counters["review"].update(review_flags)

            writer.writerow(record)

    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_targets_preview.csv",
    )
    args = parser.parse_args()

    try:
        counters = build_preview(args.manifest, args.output)
    except (OSError, ValueError) as exc:
        print(f"Could not build target preview: {exc}", file=sys.stderr)
        return 1

    print(f"Preview: {args.output} ({EXPECTED_SAMPLES} rows)")
    print("Head target counts:")
    for head, counts in counters.items():
        if head == "exercise":
            print(f"  exercise: {counts['labeled']} labeled")
            continue
        if head == "review":
            print("Review flags:")
            for flag, count in counts.items():
                print(f"  {flag}: {count} sample(s)")
            continue
        usable = sum(count for status, count in counts.items() if status != "disagreement" and status != "unlabeled")
        print(
            f"  {head}: {usable} usable / {EXPECTED_SAMPLES}; "
            f"both agree={counts['both_agree']}, "
            f"A only={counts['annotator_a_only']}, "
            f"B only={counts['annotator_b_only']}, "
            f"disagree={counts['disagreement']}, unlabeled={counts['unlabeled']}"
        )
    print("Preview built; source manifest and KERAAL files were read only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
