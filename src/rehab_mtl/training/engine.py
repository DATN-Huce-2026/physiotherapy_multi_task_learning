"""Epoch and batch loops for KERAAL multi-task training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(xyz, lengths)
            total_loss, task_losses = masked_multitask_cross_entropy(
                logits,
                target_ids,
                target_masks,
                task_weights=task_weights,
            )

            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("multi-task loss became non-finite")

            if training:
                total_loss.backward()
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
) -> list[dict[str, Any]]:
    """Run the epoch loop and return train/validation metrics for each epoch.

    The caller must move ``model`` to ``device`` before constructing the
    optimizer. Each batch must contain ``xyz``, ``lengths``, ``target_ids``
    and ``target_masks`` as documented by the model and loss functions. Pass
    re-iterable loaders, such as a PyTorch ``DataLoader``, so batches are
    available again at the start of each epoch.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    selected_device = torch.device(device)
    history: list[dict[str, Any]] = []

    for epoch_index in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_batches,
            selected_device,
            optimizer=optimizer,
            task_weights=task_weights,
        )
        validation_metrics = _run_epoch(
            model,
            validation_batches,
            selected_device,
            optimizer=None,
            task_weights=task_weights,
        )
        history.append(
            {
                "epoch": epoch_index,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

    return history
