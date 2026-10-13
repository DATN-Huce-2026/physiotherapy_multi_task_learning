#!/usr/bin/env python3
"""Train the baseline KERAAL multi-task model using subject-disjoint splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sized
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets import build_keraal_dataloaders  # noqa: E402
from rehab_mtl.models.shared_backbone import KeraalSharedBackbone  # noqa: E402
from rehab_mtl.training.engine import fit  # noqa: E402

# Trọng số mặc định cho từng head (bạn có thể trực tiếp sửa các số này ở đây):
DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "exercise": 1.0,
    "global_evaluation": 1.5,
    "error_type": 1.2,
    "severity": 0.8,
    "body_part": 0.5,
}

# Trọng số cân bằng lớp (chống mất cân bằng class trong từng head):
# Cho global_evaluation: [Correct, Incorrect, Incomplete, Motionless]
# Tăng phạt khi đoán sai Incomplete và Motionless (lớp thiểu số)
DEFAULT_CLASS_WEIGHTS: dict[str, torch.Tensor] = {
    "global_evaluation": torch.tensor([0.50, 0.77, 1.50, 5.00], dtype=torch.float32),
}


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_weights = _parse_task_weights(args.task_weights) if args.task_weights else DEFAULT_TASK_WEIGHTS
    class_weights = DEFAULT_CLASS_WEIGHTS if args.use_class_weights else None
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
    print(f"Task weights:  {task_weights}")
    print(f"Class weights: {'Enabled (minority penalty)' if class_weights else 'Disabled (uniform)'}")
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

    model = KeraalSharedBackbone().to(args.device)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
