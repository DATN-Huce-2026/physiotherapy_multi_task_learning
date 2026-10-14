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
ANNOTATORS = ("a", "b")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.label_maps import (  # noqa: E402
    GROUP3_LABEL_CODE_TO_ERROR_TYPE,
    GROUP3_LABEL_CODE_TO_EXERCISE_ERROR_CLASS,
    GROUP3_LABEL_CODE_TO_GLOBAL_EVALUATION,
)

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


def _resolve_exercise_error_class(
    group: str,
    group3_label_code: str | None,
    global_evaluation: str,
    global_evaluation_mask: bool,
    error_type: str,
    error_type_mask: bool,
) -> tuple[str, bool, str]:
    """Resolve the shared four-class target without filling uncertain labels."""
    if group == "group3":
        if group3_label_code not in GROUP3_LABEL_CODE_TO_EXERCISE_ERROR_CLASS:
            raise ValueError(f"unmapped group3 filename code: {group3_label_code!r}")
        if group3_label_code == "C":
            return "Correct", True, "group3_filename_code_C"
        return (
            GROUP3_LABEL_CODE_TO_EXERCISE_ERROR_CLASS[group3_label_code],
            True,
            "group3_filename_error_prefix_mapping",
        )

    if global_evaluation_mask and global_evaluation in {
        "Incomplete",
        "Motionless",
    }:
        return "", False, f"excluded_{global_evaluation.lower()}"

    if error_type_mask and error_type in {"Error1", "Error2", "Error3"}:
        if global_evaluation_mask and global_evaluation == "Correct":
            return "", False, "conflicting_correct_and_error_type"
        return error_type, True, "annotator_error_type"

    if global_evaluation_mask and global_evaluation == "Correct":
        if error_type_mask and error_type not in {"none"}:
            return "", False, "conflicting_correct_and_error_type"
        return "Correct", True, "annotator_global_correct"

    if error_type_mask and error_type == "Other":
        return "", False, "other_outside_four_class_task"
    if global_evaluation_mask and global_evaluation == "Incorrect":
        return "", False, "incorrect_without_supported_error_type"
    return "", False, "insufficient_or_disputed_annotation"


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
    sample_count = len(rows)
    if sample_count == 0:
        raise ValueError("manifest contains no samples")

    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate sample_id values")

    columns = [
        "sample_id",
        "group",
        "subject_token",
        "exercise_code",
        "kinect_frame_count",
        "group3_filename_context",
        "group3_label_code",
        "group3_label_code_mask",
        "group3_label_code_resolution",
        "exercise_error_class_target",
        "exercise_error_class_mask",
        "exercise_error_class_resolution",
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
    counters["group3_label_code"] = Counter()
    counters["exercise_error_class"] = Counter()
    counters["exercise_error_class_values"] = Counter()
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
            group3_label = _clean(row.get("group3_label_code"))
            if row.get("group") == "group3":
                if group3_label is None:
                    raise ValueError(
                        f"{row.get('sample_id', '<unknown>')}: missing group3 label code"
                    )
                record["group3_filename_context"] = _clean(
                    row.get("group3_filename_context")
                ) or ""
                record["group3_label_code"] = group3_label
                record["group3_label_code_mask"] = "true"
                record["group3_label_code_resolution"] = "preserved_raw_filename_code"
                counters["group3_label_code"][group3_label] += 1
            else:
                record["group3_filename_context"] = ""
                record["group3_label_code"] = ""
                record["group3_label_code_mask"] = "false"
                record["group3_label_code_resolution"] = "not_applicable"
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
                if row.get("group") != "group3":
                    counters[field][resolution] += 1

            if row.get("group") == "group3":
                derived_group3_targets = {
                    "global_evaluation": GROUP3_LABEL_CODE_TO_GLOBAL_EVALUATION[
                        group3_label
                    ],
                    "error_type": GROUP3_LABEL_CODE_TO_ERROR_TYPE[group3_label],
                }
                for field, target in derived_group3_targets.items():
                    record[f"{field}_target"] = target
                    record[f"{field}_mask"] = "true"
                    record[f"{field}_resolution"] = "group3_filename_code_mapping"
                    counters[field]["group3_filename_code_mapping"] += 1

            class_target, class_mask, class_resolution = _resolve_exercise_error_class(
                row.get("group", ""),
                group3_label,
                record["global_evaluation_target"],
                record["global_evaluation_mask"] == "true",
                record["error_type_target"],
                record["error_type_mask"] == "true",
            )
            record["exercise_error_class_target"] = class_target
            record["exercise_error_class_mask"] = str(class_mask).lower()
            record["exercise_error_class_resolution"] = class_resolution
            counters["exercise_error_class"][class_resolution] += 1
            if class_mask:
                counters["exercise_error_class_values"][class_target] += 1

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

    sample_count = counters["exercise"]["labeled"]
    print(f"Preview: {args.output} ({sample_count} rows)")
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
        if head == "group3_label_code":
            print("  group3 filename label codes:")
            for label_code, count in sorted(counts.items()):
                print(f"    {label_code}: {count}")
            continue
        if head == "exercise_error_class":
            values = counters["exercise_error_class_values"]
            print(
                "  exercise_error_class: "
                f"{sum(values.values())} usable / {sample_count}; "
                + ", ".join(
                    f"{label}={values[label]}"
                    for label in ("Correct", "Error1", "Error2", "Error3")
                )
            )
            continue
        if head == "exercise_error_class_values":
            continue
        usable = sum(count for status, count in counts.items() if status != "disagreement" and status != "unlabeled")
        print(
            f"  {head}: {usable} usable / {sample_count}; "
            f"both agree={counts['both_agree']}, "
            f"A only={counts['annotator_a_only']}, "
            f"B only={counts['annotator_b_only']}, "
            f"disagree={counts['disagreement']}, unlabeled={counts['unlabeled']}"
        )
    print("Preview built; source manifest and KERAAL files were read only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
