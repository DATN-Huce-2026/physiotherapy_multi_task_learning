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
    regression_targets: Mapping[str, torch.Tensor] | None = None,
    task_weights: Mapping[str, float] | None = None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute loss per labeled task, then average tasks with task weights.

    - Classification tasks: Cross-Entropy loss with optional class_weights.
    - Regression tasks (e.g. 'quality'): Smooth L1 (Huber) loss.

    For each task, only batch rows with a true target mask contribute to that
    task's loss. Tasks with no labeled rows in the batch are skipped.
    """
    if not logits:
        raise ValueError("logits must contain at least one task")

    losses: dict[str, torch.Tensor] = {}
    weighted_loss_sum: torch.Tensor | None = None
    active_weight_sum = 0.0

    for task_name, task_logits in logits.items():
        if task_name not in target_masks:
            continue

        task_mask = target_masks[task_name].to(
            device=task_logits.device,
            dtype=torch.bool,
        )
        if not bool(task_mask.any()):
            continue

        batch_size = task_logits.shape[0]
        if task_mask.shape != (batch_size,):
            raise ValueError(f"{task_name} mask must have shape [batch]")

        if task_name == "quality":
            # Continuous movement quality regression with Smooth L1 (Huber) Loss
            if regression_targets is not None and "quality" in regression_targets:
                task_targets = regression_targets["quality"].to(
                    device=task_logits.device,
                    dtype=task_logits.dtype,
                )
            elif "quality" in target_ids:
                task_targets = target_ids["quality"].to(
                    device=task_logits.device,
                    dtype=task_logits.dtype,
                )
            else:
                continue

            preds = task_logits.squeeze(-1) if task_logits.ndim > 1 else task_logits
            task_loss = F.smooth_l1_loss(
                preds[task_mask],
                task_targets[task_mask],
                beta=0.1,
            )
        else:
            # Multi-class classification task with Cross-Entropy Loss
            if task_name not in target_ids:
                continue
            if task_logits.ndim != 2:
                continue

            task_targets = target_ids[task_name].to(
                device=task_logits.device,
                dtype=torch.long,
            )
            if task_targets.shape != (batch_size,):
                raise ValueError(f"{task_name} target IDs must have shape [batch]")

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
