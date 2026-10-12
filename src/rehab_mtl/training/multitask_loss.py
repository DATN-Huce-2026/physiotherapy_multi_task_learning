"""Masked loss calculation for partially labeled multi-task batches."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def masked_multitask_cross_entropy(
    logits: Mapping[str, torch.Tensor],
    target_ids: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    *,
    task_weights: Mapping[str, float] | None = None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute mean cross-entropy per labeled task, then average tasks.

    For each task, only batch rows with a true target mask contribute to that
    task's loss. Target IDs at masked rows are ignored and may use any valid
    placeholder value. Tasks with no labeled rows in the batch are skipped.

    `class_weights` allows specifying a 1D tensor of class weights per task
    (e.g., to handle severe class imbalance within a task).

    Returns the combined loss and the individual active task losses.
    """
    if not logits:
        raise ValueError("logits must contain at least one task")

    losses: dict[str, torch.Tensor] = {}
    weighted_loss_sum: torch.Tensor | None = None
    active_weight_sum = 0.0

    for task_name, task_logits in logits.items():
        if task_name not in target_ids or task_name not in target_masks:
            raise ValueError(f"missing target IDs or mask for task {task_name!r}")
        if task_logits.ndim != 2:
            raise ValueError(f"{task_name} logits must have shape [batch, classes]")

        task_targets = target_ids[task_name].to(
            device=task_logits.device,
            dtype=torch.long,
        )
        task_mask = target_masks[task_name].to(
            device=task_logits.device,
            dtype=torch.bool,
        )
        batch_size = task_logits.shape[0]
        if task_targets.shape != (batch_size,):
            raise ValueError(f"{task_name} target IDs must have shape [batch]")
        if task_mask.shape != (batch_size,):
            raise ValueError(f"{task_name} mask must have shape [batch]")

        if not bool(task_mask.any()):
            continue

        cw: torch.Tensor | None = None
        if class_weights is not None and task_name in class_weights:
            cw = class_weights[task_name].to(
                device=task_logits.device,
                dtype=task_logits.dtype,
            )

        task_loss = F.cross_entropy(
            task_logits[task_mask],
            task_targets[task_mask],
            weight=cw,
        )
        weight = 1.0 if task_weights is None else task_weights.get(task_name, 1.0)
        if weight <= 0:
            raise ValueError(f"weight for active task {task_name!r} must be positive")

        losses[task_name] = task_loss
        weighted = task_loss * weight
        weighted_loss_sum = (
            weighted
            if weighted_loss_sum is None
            else weighted_loss_sum + weighted
        )
        active_weight_sum += weight

    if weighted_loss_sum is None:
        raise ValueError("this batch has no labeled targets for any task")

    return weighted_loss_sum / active_weight_sum, losses
