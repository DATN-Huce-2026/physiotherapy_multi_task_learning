#!/usr/bin/env python3
"""Train the baseline KERAAL multi-task model using subject-disjoint splits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sized
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets import TASK_MAPPINGS, build_keraal_dataloaders  # noqa: E402
from rehab_mtl.models.shared_backbone import KeraalSharedBackbone  # noqa: E402
from rehab_mtl.training.engine import evaluate, fit  # noqa: E402

# Trọng số mặc định cho từng head (bạn có thể trực tiếp sửa các số này ở đây):
DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "exercise": 1.0,
    "exercise_error_class": 1.2,
    "global_evaluation": 1.5,
    "error_type": 1.2,
    "severity": 0.8,
    "body_part": 0.5,
}

# Trọng số cân bằng lớp mặc định (hiệu chỉnh theo phân bố tập train 841 clips):
# Dùng làm fallback khi không dùng chế độ auto-weights
DEFAULT_CLASS_WEIGHTS: dict[str, torch.Tensor] = {
    # global_evaluation: [Correct, Incorrect, Incomplete, Motionless] (train: 111, 178, 22, 1)
    "global_evaluation": torch.tensor([0.70, 0.50, 1.80, 8.00], dtype=torch.float32),
    # error_type: [none, Error1, Error2, Error3, Other] (train: 124, 62, 104, 51, 1)
    "error_type": torch.tensor([0.65, 0.95, 0.70, 1.05, 8.00], dtype=torch.float32),
    # body_part: [Spine, BothArms, RightArm, LeftArm] (train: 52, 33, 3, 1)
    "body_part": torch.tensor([0.60, 0.80, 2.50, 5.00], dtype=torch.float32),
    # severity: [SmallError, LargeError] (train: 60, 25)
    "severity": torch.tensor([0.80, 1.25], dtype=torch.float32),
    # exercise_error_class: [Correct, Error1, Error2, Error3] (train: 111, 62, 97, 51)
    "exercise_error_class": torch.tensor([0.85, 1.10, 0.90, 1.20], dtype=torch.float32),
}


def compute_train_class_weights(
    train_dataset: Any,
    tasks: list[str] | None = None,
    max_weight: float = 15.0,
) -> dict[str, torch.Tensor]:
    """Tự động tính class weights từ train set theo nghịch đảo căn bậc hai tần suất (Smoothed Inverse Frequency).

    w_c = sqrt(total / (num_classes * count_c)), chặn trên bởi max_weight để tránh
    bùng nổ gradient đối với các lớp chỉ có 1 mẫu (như Motionless hay Other),
    sau đó chuẩn hóa để trọng số trung bình mỗi task bằng 1.0.
    """
    task_mappings = getattr(train_dataset, "task_mappings", TASK_MAPPINGS)
    target_rows = getattr(train_dataset, "target_rows", {})
    sample_ids = getattr(train_dataset, "sample_ids", [])

    selected_tasks = tasks or list(task_mappings.keys())
    computed_weights: dict[str, torch.Tensor] = {}

    for task_name in selected_tasks:
        if task_name not in task_mappings:
            continue
        mapping = task_mappings[task_name]
        num_classes = len(mapping)
        counts: Counter[int] = Counter()

        for sid in sample_ids:
            row = target_rows.get(sid, {})
            if row.get(f"{task_name}_mask") == "true":
                raw_val = row.get(f"{task_name}_target")
                if raw_val in mapping:
                    counts[mapping[raw_val]] += 1

        total_valid = sum(counts.values())
        if total_valid == 0:
            continue

        weights = torch.zeros(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            cnt = counts.get(c, 0)
            if cnt > 0:
                w = math.sqrt(total_valid / (num_classes * cnt))
                w = min(w, max_weight)
                weights[c] = w
            else:
                weights[c] = 1.0

        weights = weights / weights.mean()
        computed_weights[task_name] = weights

    return computed_weights


def _parse_task_weights(text: str | None) -> dict[str, float] | None:
    """Parse key=value,key2=value2 or JSON string into dict."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("{"):
        import json
        return json.loads(text)
    result: dict[str, float] = {}
    for item in text.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            result[k.strip()] = float(v.strip())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keraal-root",
        type=Path,
        default=Path.home() / "Documents" / "KERAAL",
        help="Path to root directory of KERAAL dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_manifest.csv",
        help="Path to KERAAL manifest CSV",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_targets_preview.csv",
        help="Path to KERAAL targets preview CSV",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "data" / "keraal_subject_split_preview.csv",
        help="Path to subject split CSV",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for AdamW optimizer. Default: 1e-4",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["none", "plateau", "cosine"],
        help="Learning rate scheduler ('plateau', 'cosine', or 'none'). Default: 'plateau'",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="Number of epochs with no validation loss improvement before reducing LR (for plateau). Default: 3",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Decay factor for plateau scheduler (new_lr = lr * lr_factor). Default: 0.5",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate bound for scheduler. Default: 1e-6",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baseline",
        help="Directory to save training logs and model checkpoints",
    )
    parser.add_argument(
        "--no-origin-offset",
        action="store_true",
        help="Do not offset XYZ coordinates to first-frame SpineBase",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload all samples into memory during dataset initialization",
    )
    parser.add_argument(
        "--task-weights",
        type=str,
        default=None,
        help="Comma-separated task weights, e.g. 'exercise=1.0,global_evaluation=1.5' (defaults to DEFAULT_TASK_WEIGHTS)",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Apply class weights to penalize errors on minority classes (e.g. Incomplete/Motionless)",
    )
    parser.add_argument(
        "--class-weights-mode",
        type=str,
        default="auto",
        choices=["auto", "default"],
        help="Mode for class weights: 'auto' (computed dynamically from train split) or 'default' (calibrated DEFAULT_CLASS_WEIGHTS). Default: 'auto'",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "last"],
        help="Temporal pooling method: 'mean' (Masked Global Average Pooling) or 'last' (final hidden state). Default: 'mean'",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply skeleton data augmentation (random Y-rotation, jitter, scaling) to train batches. (default: True, use --no-augment to disable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_weights = _parse_task_weights(args.task_weights) if args.task_weights else DEFAULT_TASK_WEIGHTS
    print("=" * 60)
    print("Rehab MTL — KERAAL Baseline Multi-Task Training")
    print(f"Device: {args.device}")
    print(f"KERAAL root: {args.keraal_root}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr} | Weight decay: {args.weight_decay}")
    if args.lr_scheduler == "plateau":
        print(f"LR Scheduler:  ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor}, min_lr={args.min_lr})")
    elif args.lr_scheduler == "cosine":
        print(f"LR Scheduler:  CosineAnnealingLR (T_max={args.epochs}, min_lr={args.min_lr})")
    else:
        print("LR Scheduler:  Disabled (fixed LR)")
    print(f"Pooling:       {'MEAN (Masked Global Average Pooling)' if args.pooling == 'mean' else 'LAST (Final BiGRU hidden state)'}")
    print(f"Data augment:  {'Enabled (Y-rotation ±15°, jitter 8mm, scale 0.95-1.05)' if args.augment else 'Disabled'}")
    print(f"Task weights:  {task_weights}")
    print(f"Class weights: {f'Enabled (mode={args.class_weights_mode})' if args.use_class_weights else 'Disabled (uniform)'}")
    print("=" * 60)

    try:
        loaders = build_keraal_dataloaders(
            keraal_root=args.keraal_root,
            manifest_path=args.manifest,
            targets_path=args.targets,
            split_path=args.splits,
            batch_size=args.batch_size,
            relative_to_initial_spine_base=not args.no_origin_offset,
            preload=args.preload,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Failed to initialize DataLoaders: {exc}", file=sys.stderr)
        return 1

    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    test_loader = loaders["test"]

    assert isinstance(train_loader.dataset, Sized)
    assert isinstance(val_loader.dataset, Sized)
    assert isinstance(test_loader.dataset, Sized)

    print("Dataset split sizes:")
    print(f"  Train:      {len(train_loader.dataset)} clips ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_loader.dataset)} clips ({len(val_loader)} batches)")
    print(f"  Test:       {len(test_loader.dataset)} clips ({len(test_loader)} batches)")

    class_weights = None
    if args.use_class_weights:
        if args.class_weights_mode == "auto":
            class_weights = compute_train_class_weights(train_loader.dataset)
            print("\nAuto-computed class weights from train split (smoothed inverse frequency):")
            for task, w in class_weights.items():
                print(f"  {task:20s}: {w.round(decimals=3).tolist()}")
        else:
            class_weights = DEFAULT_CLASS_WEIGHTS
            print("\nUsing calibrated static DEFAULT_CLASS_WEIGHTS:")
            for task, w in class_weights.items():
                print(f"  {task:20s}: {w.round(decimals=3).tolist()}")

    model = KeraalSharedBackbone(pooling=args.pooling).to(args.device)
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

    print("\nStarting training loop...")
    history = fit(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=args.device,
        task_weights=task_weights,
        class_weights=class_weights,
        checkpoint_dir=args.output_dir,
        best_model_filename="best_model.pt",
        augment=args.augment,
        verbose=True,
    )

    history_path = args.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    last_model_path = args.output_dir / "last_model.pt"
    torch.save(model.state_dict(), last_model_path)
    print(f"Final epoch model saved to: {last_model_path}")
    print(f"Best validation model saved to: {args.output_dir / 'best_model.pt'}")

    best_model_path = args.output_dir / "best_model.pt"
    if best_model_path.is_file():
        print("\n" + "=" * 65)
        print("Evaluating Best Checkpoint on Test Set (Subject-Disjoint)...")
        print("=" * 65)
        model.load_state_dict(
            torch.load(best_model_path, map_location=args.device, weights_only=True)
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device=args.device,
            task_weights=task_weights,
            class_weights=class_weights,
        )

        print(f"\nTest Multi-Task Overall Loss: {test_metrics['loss']:.4f} (across {test_metrics['sample_count']} test clips)\n")
        print(f"{'Task Name':<24} | {'Loss':<8} | {'Acc (%)':<8} | {'Macro F1':<8} | {'Valid Samples'}")
        print("-" * 65)
        for task_name in test_metrics["task_losses"]:
            loss_val = test_metrics["task_losses"].get(task_name, 0.0)
            acc_val = test_metrics["task_accuracies"].get(task_name, 0.0) * 100.0
            f1_val = test_metrics["task_macro_f1"].get(task_name, 0.0)
            cnt_val = test_metrics["task_sample_counts"].get(task_name, 0)
            print(f"{task_name:<24} | {loss_val:<8.4f} | {acc_val:<7.2f}% | {f1_val:<8.4f} | {cnt_val}")

        test_results_path = args.output_dir / "test_evaluation.json"
        with test_results_path.open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nTest evaluation results saved to: {test_results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
