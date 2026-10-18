#!/usr/bin/env python3
"""Train the baseline KiMoRe movement quality regression model using subject-disjoint splits."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections.abc import Sized
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.datasets.kimore_dataset import build_kimore_dataloaders  # noqa: E402
from rehab_mtl.models.shared_backbone import KeraalSharedBackbone  # noqa: E402
from rehab_mtl.training.engine import evaluate, fit  # noqa: E402

if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "quality": 1.0,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the baseline KiMoRe movement quality model with subject-disjoint splits."
    )
    parser.add_argument(
        "--kimore-root",
        type=Path,
        default=Path(r"D:\DATN\KIMORE"),
        help="Root directory of the KiMoRe dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_manifest.csv",
        help="Path to kimore_manifest.csv",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_targets_preview.csv",
        help="Path to kimore_targets_preview.csv",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "data" / "kimore" / "kimore_subject_split_preview.csv",
        help="Path to kimore_subject_split_preview.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
        help="Number of training epochs (default: 70)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau", "cosine"],
        default="plateau",
        help="Learning rate scheduler (default: plateau)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Patience epochs before reducing LR (default: 5)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Decay factor for ReduceLROnPlateau (default: 0.5)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Minimum LR limit (default: 1e-5)",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "last"],
        default="mean",
        help="Backbone pooling strategy: 'mean' (Masked GAP) or 'last' (default: mean)",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply 3D skeleton data augmentation on training batches (default: True)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda', 'cpu', etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "kimore_baseline",
        help="Directory to save training checkpoints and history",
    )
    parser.add_argument(
        "--no-origin-offset",
        action="store_true",
        help="Do NOT offset skeleton positions relative to initial SpineBase",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload entire dataset into RAM",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)

    print("=" * 65)
    print("  KIMORE MOVEMENT QUALITY REGRESSION TRAINING (SUBJECT-DISJOINT)")
    print("=" * 65)
    print(f"Device:           {device}")
    print(f"KiMoRe Root:      {args.kimore_root}")
    print(f"Manifest:         {args.manifest}")
    print(f"Epochs:           {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"Pooling Strategy: {args.pooling.upper()} (Masked Global Average Pooling)")
    print(f"Augmentation:     {'ENABLED' if args.augment else 'DISABLED'}")
    kimore_root = args.kimore_root
    if not (kimore_root / "CG").is_dir() and not (kimore_root / "cg").is_dir():
        for candidate in [kimore_root / kimore_root.name, kimore_root / "KIMORE", kimore_root / "kimore"]:
            if (candidate / "CG").is_dir() or (candidate / "cg").is_dir():
                print(f"[INFO] Auto-detected nested KiMoRe root: {candidate}")
                kimore_root = candidate
                break

    print(f"Output Dir:       {args.output_dir}")
    print("=" * 65)

    loaders = build_kimore_dataloaders(
        kimore_root=kimore_root,
        manifest_path=args.manifest,
        targets_path=args.targets,
        split_path=args.splits,
        batch_size=args.batch_size,
        relative_to_initial_spine_base=not args.no_origin_offset,
        preload=args.preload,
    )

    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    test_loader = loaders["test"]

    assert isinstance(train_loader.dataset, Sized)
    assert isinstance(val_loader.dataset, Sized)
    assert isinstance(test_loader.dataset, Sized)

    train_count = len(train_loader.dataset)
    val_count = len(val_loader.dataset)
    test_count = len(test_loader.dataset)
    print(f"\n--> Dataset Splits: Train={train_count} | Val={val_count} | Test={test_count}")

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

    print("\n--> Starting Training...\n")
    history = fit(
        model=model,
        train_batches=train_loader,
        validation_batches=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        task_weights=DEFAULT_TASK_WEIGHTS,
        scheduler=scheduler,
        checkpoint_dir=args.output_dir,
        augment=args.augment,
        verbose=True,
    )

    history_path = args.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training history saved to: {history_path}")

    # Final evaluation on the unseen Test Set
    best_model_path = args.output_dir / "best_model.pt"
    if best_model_path.is_file():
        print(f"\n--> Loading best checkpoint for Test evaluation: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n--> Evaluating on Subject-Disjoint TEST Set...")
    test_results = evaluate(
        model=model,
        batches=test_loader,
        device=device,
        task_weights=DEFAULT_TASK_WEIGHTS,
    )

    print("\n" + "=" * 50)
    print("           TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Test Loss: {test_results['loss']:.4f}")
    if "quality" in test_results.get("task_mae", {}):
        mae = test_results["task_mae"]["quality"]
        rmse = test_results["task_rmse"]["quality"]
        raw_scale_mae = mae * 50.0
        raw_scale_rmse = rmse * 50.0
        print(f"Quality Normalized MAE:  {mae:.4f}  (Scale 0-1)")
        print(f"Quality Normalized RMSE: {rmse:.4f}  (Scale 0-1)")
        print(f"Quality Raw Scale MAE:   {raw_scale_mae:.2f} / 50 EAAQ clinical score")
        print(f"Quality Raw Scale RMSE:  {raw_scale_rmse:.2f} / 50 EAAQ clinical score")
    print("=" * 50)

    test_eval_path = args.output_dir / "test_evaluation.json"
    with test_eval_path.open("w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"[OK] Test evaluation saved to: {test_eval_path}")


if __name__ == "__main__":
    main()
