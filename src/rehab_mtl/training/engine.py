"""Epoch and batch loops for KERAAL multi-task training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rehab_mtl.training.multitask_loss import masked_multitask_cross_entropy


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Move the documented batch fields to the selected device."""
    required_fields = {"xyz", "lengths", "target_ids", "target_masks"}
    missing_fields = required_fields.difference(batch)
    if missing_fields:
        raise ValueError(f"batch is missing fields: {sorted(missing_fields)}")

    xyz = batch["xyz"].to(device=device, dtype=torch.float32)
    lengths = batch["lengths"].to(device=device, dtype=torch.long)
    target_ids = {
        task: values.to(device=device, dtype=torch.long)
        for task, values in batch["target_ids"].items()
    }
    target_masks = {
        task: values.to(device=device, dtype=torch.bool)
        for task, values in batch["target_masks"].items()
    }
    return xyz, lengths, target_ids, target_masks


def _run_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    task_weights: Mapping[str, float] | None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
    max_grad_norm: float | None = 1.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss_sum = 0.0
    sample_count = 0
    task_loss_sums: dict[str, float] = {}
    task_target_counts: dict[str, int] = {}
    batch_count = 0

    for batch in batches:
        xyz, lengths, target_ids, target_masks = _move_batch_to_device(batch, device)
        batch_size = xyz.shape[0]
        if batch_size == 0:
            continue

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(xyz, lengths)
            total_loss, task_losses = masked_multitask_cross_entropy(
                logits,
                target_ids,
                target_masks,
                task_weights=task_weights,
                class_weights=class_weights,
            )

            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("multi-task loss became non-finite")

            if optimizer is not None:
                total_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        total_loss_sum += float(total_loss.detach()) * batch_size
        sample_count += batch_size
        batch_count += 1

        for task_name, task_loss in task_losses.items():
            valid_count = int(target_masks[task_name].sum().item())
            task_loss_sums[task_name] = (
                task_loss_sums.get(task_name, 0.0)
                + float(task_loss.detach()) * valid_count
            )
            task_target_counts[task_name] = (
                task_target_counts.get(task_name, 0) + valid_count
            )

    if batch_count == 0:
        raise ValueError("the data loader produced no non-empty batches")

    mean_task_losses = {
        task_name: loss_sum / task_target_counts[task_name]
        for task_name, loss_sum in task_loss_sums.items()
        if task_target_counts[task_name] > 0
    }
    return {
        "loss": total_loss_sum / sample_count,
        "task_losses": mean_task_losses,
        "task_target_counts": task_target_counts,
        "batch_count": batch_count,
        "sample_count": sample_count,
    }


def fit(
    model: nn.Module,
    train_batches: Iterable[Mapping[str, Any]],
    validation_batches: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    device: torch.device | str,
    task_weights: Mapping[str, float] | None = None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
    scheduler: Any | None = None,
    max_grad_norm: float | None = 1.0,
    checkpoint_dir: str | Path | None = None,
    best_model_filename: str = "best_model.pt",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the epoch loop and return train/validation metrics for each epoch.

    The caller must move ``model`` to ``device`` before constructing the
    optimizer. Each batch must contain ``xyz``, ``lengths``, ``target_ids``
    and ``target_masks`` as documented by the model and loss functions. Pass
    re-iterable loaders, such as a PyTorch ``DataLoader``, so batches are
    available again at the start of each epoch.

    If ``checkpoint_dir`` is specified, the model checkpoint with the lowest
    validation loss is automatically saved as ``best_model_filename``.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    selected_device = torch.device(device)
    history: list[dict[str, Any]] = []

    save_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch_index in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = _run_epoch(
            model,
            train_batches,
            selected_device,
            optimizer=optimizer,
            task_weights=task_weights,
            class_weights=class_weights,
            max_grad_norm=max_grad_norm,
        )
        validation_metrics = _run_epoch(
            model,
            validation_batches,
            selected_device,
            optimizer=None,
            task_weights=task_weights,
            class_weights=class_weights,
            max_grad_norm=None,
        )

        val_loss = validation_metrics["loss"]
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch_index
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir / best_model_filename)

        lr_before_step = current_lr
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        lr_after_step = optimizer.param_groups[0]["lr"]

        if verbose:
            best_tag = f" -> [BEST SAVED ({best_model_filename})]" if (is_best and save_dir is not None) else ""
            print(
                f"Epoch {epoch_index:3d}/{epochs} | "
                f"LR: {lr_before_step:.6f} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_loss:.4f}"
                f"{best_tag}"
            )
            if lr_after_step < lr_before_step:
                print(
                    f"    ↳ [LR Scheduler] Learning rate reduced: {lr_before_step:.6f} -> {lr_after_step:.6f}"
                )

        history.append(
            {
                "epoch": epoch_index,
                "lr": lr_before_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "is_best": is_best,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
        )

    if verbose and save_dir is not None and best_epoch != -1:
        print(f"\n[OK] Best model checkpoint saved at Epoch {best_epoch} with Val Loss: {best_val_loss:.4f} to: {save_dir / best_model_filename}")

    return history

