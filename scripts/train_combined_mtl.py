#!/usr/bin/env python3
"""Train joint multi-task learning model combining KERAAL and KiMoRe datasets."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import Counter
from collections.abc import Sized
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.keraal_dataset import KeraalMTLDataset, collate_keraal_mtl  # noqa: E402
from rehab_mtl.datasets.kimore_dataset import KimoreMTLDataset  # noqa: E402
from rehab_mtl.datasets.label_maps import TASK_MAPPINGS  # noqa: E402
from rehab_mtl.models.shared_backbone import KeraalSharedBackbone  # noqa: E402
from rehab_mtl.training.engine import evaluate, fit  # noqa: E402

if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Default weights balancing classification tasks (KERAAL) and quality regression (KiMoRe)
DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "exercise": 1.0,
    "exercise_error_class": 1.2,
    "global_evaluation": 1.5,
    "error_type": 1.2,
    "severity": 0.8,
    "body_part": 0.5,
    "quality": 1.0,
}

DEFAULT_CLASS_WEIGHTS: dict[str, torch.Tensor] = {
    "global_evaluation": torch.tensor([0.70, 0.50, 1.80, 8.00], dtype=torch.float32),
    "error_type": torch.tensor([0.65, 0.95, 0.70, 1.05, 8.00], dtype=torch.float32),
    "body_part": torch.tensor([0.60, 0.80, 2.50, 5.00], dtype=torch.float32),
    "severity": torch.tensor([0.80, 1.25], dtype=torch.float32),
    "exercise_error_class": torch.tensor([0.85, 1.10, 0.90, 1.20], dtype=torch.float32),
}


def _parse_task_weights(arg_string: str | None) -> dict[str, float]:
    if not arg_string:
        return dict(DEFAULT_TASK_WEIGHTS)
    weights = dict(DEFAULT_TASK_WEIGHTS)
    for pair in arg_string.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"invalid task weight pair {pair!r}; expected task=weight")
        task, val_str = pair.split("=", 1)
        task = task.strip()
        try:
            val = float(val_str.strip())
        except ValueError as exc:
            raise ValueError(f"invalid weight {val_str!r} for task {task!r}") from exc
        weights[task] = val
    return weights


def compute_keraal_class_weights(
    train_dataset: KeraalMTLDataset,
    max_weight: float = 15.0,
) -> dict[str, torch.Tensor]:
    """Compute balanced class weights for classification tasks from KERAAL train split."""
    task_mappings = getattr(train_dataset, "task_mappings", TASK_MAPPINGS)
    target_rows = getattr(train_dataset, "target_rows", {})
    sample_ids = getattr(train_dataset, "sample_ids", [])

    computed: dict[str, torch.Tensor] = {}
    for task_name, mapping in task_mappings.items():
        if task_name == "quality":
            continue
        num_classes = len(mapping)
        counts: Counter[int] = Counter()
        for sid in sample_ids:
            row = target_rows.get(sid, {})
            if row.get(f"{task_name}_mask") == "true":
                raw_val = row.get(f"{task_name}_target")
                if raw_val in mapping:
                    counts[mapping[raw_val]] += 1
        total_samples = sum(counts.values())
        if total_samples == 0:
            continue
        weights = []
        for c in range(num_classes):
            cnt = counts.get(c, 0)
            if cnt > 0:
                raw_w = math.sqrt(total_samples / (num_classes * cnt))
                w = min(raw_w, max_weight)
            else:
                w = max_weight
            weights.append(w)
        tensor_w = torch.tensor(weights, dtype=torch.float32)
        mean_w = tensor_w.mean()
        if mean_w > 0:
            tensor_w = tensor_w / mean_w
        computed[task_name] = tensor_w
    return computed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint multi-task training on KERAAL (error assessment) and KiMoRe (quality scoring)."
    )
    # KERAAL paths
    parser.add_argument(
        "--keraal-root",
        type=Path,
        default=Path.home() / "Documents" / "KERAAL",
        help="Root directory of KERAAL dataset",
    )
    parser.add_argument(
        "--keraal-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal" / "keraal_manifest.csv",
        help="Path to keraal_manifest.csv",
    )
    parser.add_argument(
        "--keraal-targets",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal" / "keraal_targets_preview.csv",
        help="Path to keraal_targets_preview.csv",
    )
    parser.add_argument(
        "--keraal-splits",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal" / "keraal_subject_split_preview.csv",
        help="Path to keraal_subject_split_preview.csv",
    )

    # KiMoRe paths
    parser.add_argument(
        "--kimore-root",
        type=Path,
        default=Path(r"D:\DATN\KIMORE"),
        help="Root directory of KiMoRe dataset",
    )
    parser.add_argument(
        "--kimore-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_manifest.csv",
        help="Path to kimore_manifest.csv",
    )
    parser.add_argument(
        "--kimore-targets",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_targets_preview.csv",
        help="Path to kimore_targets_preview.csv",
    )
    parser.add_argument(
        "--kimore-splits",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_subject_split_preview.csv",
        help="Path to kimore_subject_split_preview.csv",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
        help="Number of epochs (default: 70)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["plateau", "cosine", "none"],
        default="plateau",
        help="LR scheduler type (default: plateau)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=6,
        help="Patience epochs for plateau scheduler (default: 6)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Decay factor for plateau scheduler (default: 0.5)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=5e-5,
        help="Minimum LR limit (default: 5e-5)",
    )
    parser.add_argument(
        "--task-weights",
        type=str,
        default=None,
        help="Comma-separated task weights (defaults to DEFAULT_TASK_WEIGHTS)",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=True,
        help="Apply balanced class weights for KERAAL classification tasks (default: True)",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "last"],
        default="mean",
        help="Temporal pooling: 'mean' (Masked GAP) or 'last' (default: mean)",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply 3D skeleton augmentation on train batches (default: True)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "combined_mtl_v1",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload datasets into memory for maximum training speed",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    task_weights = _parse_task_weights(args.task_weights)

    print("=" * 70)
    print("  JOINT MULTI-TASK LEARNING: KERAAL + KIMORE (SUBJECT-DISJOINT)")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"KERAAL Root:      {args.keraal_root}")
    print(f"KiMoRe Root:      {args.kimore_root}")
    print(f"Epochs:           {args.epochs} | Batch size: {args.batch_size} | Initial LR: {args.lr}")
    print(f"Pooling Strategy: {args.pooling.upper()} (Masked Global Average Pooling)")
    print(f"Augmentation:     {'ENABLED' if args.augment else 'DISABLED'}")
    print(f"Task Weights:     {task_weights}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)

    # 1. Initialize Datasets
    print("\n--> Loading KERAAL Datasets...")
    keraal_train = KeraalMTLDataset(
        args.keraal_root,
        split="train",
        manifest_path=args.keraal_manifest,
        targets_path=args.keraal_targets,
        split_path=args.keraal_splits,
        preload=args.preload,
    )
    keraal_val = KeraalMTLDataset(
        args.keraal_root,
        split="validation",
        manifest_path=args.keraal_manifest,
        targets_path=args.keraal_targets,
        split_path=args.keraal_splits,
        preload=args.preload,
    )
    keraal_test = KeraalMTLDataset(
        args.keraal_root,
        split="test",
        manifest_path=args.keraal_manifest,
        targets_path=args.keraal_targets,
        split_path=args.keraal_splits,
        preload=args.preload,
    )

    print("--> Loading KiMoRe Datasets...")
    kimore_train = KimoreMTLDataset(
        args.kimore_root,
        split="train",
        manifest_path=args.kimore_manifest,
        targets_path=args.kimore_targets,
        split_path=args.kimore_splits,
        preload=args.preload,
    )
    kimore_val = KimoreMTLDataset(
        args.kimore_root,
        split="validation",
        manifest_path=args.kimore_manifest,
        targets_path=args.kimore_targets,
        split_path=args.kimore_splits,
        preload=args.preload,
    )
    kimore_test = KimoreMTLDataset(
        args.kimore_root,
        split="test",
        manifest_path=args.kimore_manifest,
        targets_path=args.kimore_targets,
        split_path=args.kimore_splits,
        preload=args.preload,
    )

    # 2. Concatenate Datasets
    combined_train = ConcatDataset([keraal_train, kimore_train])
    combined_val = ConcatDataset([keraal_val, kimore_val])
    combined_test = ConcatDataset([keraal_test, kimore_test])

    print("\n--> Dataset Inventory (Subject-Disjoint):")
    print(f"    * TRAIN : KERAAL={len(keraal_train):3} + KiMoRe={len(kimore_train):3} => Total={len(combined_train):4} clips")
    print(f"    * VAL   : KERAAL={len(keraal_val):3} + KiMoRe={len(kimore_val):3} => Total={len(combined_val):4} clips")
    print(f"    * TEST  : KERAAL={len(keraal_test):3} + KiMoRe={len(kimore_test):3} => Total={len(combined_test):4} clips")

    train_loader = DataLoader(
        combined_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_keraal_mtl,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_keraal_mtl,
    )
    test_loader = DataLoader(
        combined_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_keraal_mtl,
    )

    # Class weights for KERAAL classification tasks
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_keraal_class_weights(keraal_train)

    # 3. Model & Optimizer
    model = KeraalSharedBackbone(pooling=args.pooling).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Fit Model
    print("\n--> Starting Joint Training...\n")
    history = fit(
        model=model,
        train_batches=train_loader,
        validation_batches=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        task_weights=task_weights,
        class_weights=class_weights,
        scheduler=scheduler,
        checkpoint_dir=args.output_dir,
        augment=args.augment,
        verbose=True,
    )

    history_path = args.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training history saved to: {history_path}")

    # 5. Evaluate on Unseen Test Sets
    best_model_path = args.output_dir / "best_model.pt"
    if best_model_path.is_file():
        print(f"\n--> Loading best model checkpoint: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n--> Evaluating on Combined TEST Set...")
    combined_test_results = evaluate(
        model=model,
        batches=test_loader,
        device=device,
        task_weights=task_weights,
    )

    print("\n" + "=" * 65)
    print("           COMBINED TEST SET EVALUATION RESULTS")
    print("=" * 65)
    print(f"Overall Test Loss: {combined_test_results['loss']:.4f}")
    print("\nClassification Tasks (KERAAL):")
    for t_name, acc in combined_test_results.get("task_accuracies", {}).items():
        f1 = combined_test_results.get("task_macro_f1", {}).get(t_name, 0.0)
        cnt = combined_test_results.get("task_sample_counts", {}).get(t_name, 0)
        print(f"  * {t_name:22}: Accuracy={acc * 100:5.1f}% | Macro F1={f1:.4f} (N={cnt})")

    if "quality" in combined_test_results.get("task_mae", {}):
        mae = combined_test_results["task_mae"]["quality"]
        rmse = combined_test_results["task_rmse"]["quality"]
        cnt = combined_test_results.get("task_sample_counts", {}).get("quality", 0)
        print("\nMovement Quality Scoring Task (KiMoRe):")
        print(f"  * Normalized Quality MAE:  {mae:.4f}  (Scale 0.0 - 1.0)")
        print(f"  * Normalized Quality RMSE: {rmse:.4f}  (Scale 0.0 - 1.0)")
        print(f"  * Raw EAAQ Score MAE:      {mae * 50.0:.2f} / 50 clinical score")
        print(f"  * Raw EAAQ Score RMSE:     {rmse * 50.0:.2f} / 50 clinical score (N={cnt})")
    print("=" * 65)

    test_eval_path = args.output_dir / "test_evaluation.json"
    with test_eval_path.open("w", encoding="utf-8") as f:
        json.dump(combined_test_results, f, indent=2)
    print(f"[OK] Test evaluation report saved to: {test_eval_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
