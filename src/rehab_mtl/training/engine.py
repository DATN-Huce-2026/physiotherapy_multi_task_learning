"""Epoch and batch loops for KERAAL multi-task training."""

from __future__ import annotations

import math
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
    raw_regression = batch.get("regression_targets", {})
    regression_targets = {
        task: values.to(device=device, dtype=torch.float32)
        for task, values in raw_regression.items()
    }
    return xyz, lengths, target_ids, target_masks, regression_targets


def augment_skeleton_batch(
    xyz: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_rotation_degrees: float = 15.0,
    jitter_std: float = 0.008,
    scale_range: tuple[float, float] = (0.95, 1.05),
) -> torch.Tensor:
    """Tăng cường dữ liệu skeleton ngẫu nhiên trực tiếp trên batch (GPU/CPU).

    1. Xoay ngẫu nhiên quanh trục đứng Y trong khoảng [-max_rotation, +max_rotation] độ.
    2. Co giãn ngẫu nhiên tỷ lệ cơ thể (scale) trong khoảng [min_scale, max_scale].
    3. Thêm nhiễu rung tọa độ Gaussian jitter (~8mm) mô phỏng sai số đo của camera Kinect.
    Các frame đệm (zero padding) được giữ nguyên giá trị 0.
    """
    batch_size, time_steps, joints, _ = xyz.shape
    device = xyz.device

    # 1. Co giãn ngẫu nhiên (Scaling)
    scale = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(scale_range[0], scale_range[1])
    out = xyz * scale

    # 2. Xoay ngẫu nhiên quanh trục đứng Y (Y-axis rotation)
    max_rad = math.radians(max_rotation_degrees)
    theta = torch.empty(batch_size, 1, 1, device=device).uniform_(-max_rad, max_rad)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x = out[..., 0]
    y = out[..., 1]
    z = out[..., 2]

    new_x = x * cos_t + z * sin_t
    new_z = -x * sin_t + z * cos_t
    out = torch.stack([new_x, y, new_z], dim=-1)

    # 3. Thêm nhiễu Gaussian jitter trên các frame hợp lệ
    if jitter_std > 0:
        noise = torch.randn_like(out) * jitter_std
        time_indices = torch.arange(time_steps, device=device).unsqueeze(0)
        time_mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
        out = out + noise * time_mask

    return out


def _run_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    task_weights: Mapping[str, float] | None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
    max_grad_norm: float | None = 1.0,
    augment: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss_sum = 0.0
    sample_count = 0
    task_loss_sums: dict[str, float] = {}
    task_target_counts: dict[str, int] = {}
    batch_count = 0

    for batch in batches:
        xyz, lengths, target_ids, target_masks, regression_targets = _move_batch_to_device(batch, device)
        batch_size = xyz.shape[0]
        if batch_size == 0:
            continue

        if training and augment:
            xyz = augment_skeleton_batch(xyz, lengths)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(xyz, lengths)
            total_loss, task_losses = masked_multitask_cross_entropy(
                logits,
                target_ids,
                target_masks,
                regression_targets=regression_targets,
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
    augment: bool = True,
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
            augment=augment,
        )
        validation_metrics = _run_epoch(
            model,
            validation_batches,
            selected_device,
            optimizer=None,
            task_weights=task_weights,
            class_weights=class_weights,
            max_grad_norm=None,
            augment=False,
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    *,
    task_weights: Mapping[str, float] | None = None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Chạy đánh giá mô hình trên tập dữ liệu (test hoặc validation), tính loss, accuracy và macro F1.

    Trả về từ điển gồm:
        - "loss": weighted multi-task loss trung bình
        - "task_losses": loss trung bình theo từng task
        - "task_accuracies": độ chính xác (accuracy) theo từng task (chỉ tính trên mẫu có mask hợp lệ)
        - "task_macro_f1": Macro-averaged F1 score theo từng task
        - "task_sample_counts": số lượng mẫu có nhãn hợp lệ theo từng task
        - "sample_count": tổng số clip được đánh giá
        - "batch_count": tổng số batch
    """
    selected_device = torch.device(device)
    model.eval()

    total_loss_sum = 0.0
    sample_count = 0
    task_loss_sums: dict[str, float] = {}
    task_target_counts: dict[str, int] = {}
    batch_count = 0

    all_preds: dict[str, list[torch.Tensor]] = {}
    all_targets: dict[str, list[torch.Tensor]] = {}
    all_masks: dict[str, list[torch.Tensor]] = {}

    for batch in batches:
        xyz, lengths, target_ids, target_masks, regression_targets = _move_batch_to_device(batch, selected_device)
        batch_size = xyz.shape[0]
        if batch_size == 0:
            continue

        logits = model(xyz, lengths)
        total_loss, task_losses = masked_multitask_cross_entropy(
            logits,
            target_ids,
            target_masks,
            regression_targets=regression_targets,
            task_weights=task_weights,
            class_weights=class_weights,
        )

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

        for task_name, task_logits in logits.items():
            if task_name == "quality":
                if "quality" in target_masks:
                    q_targets = (
                        regression_targets.get("quality")
                        if "quality" in regression_targets
                        else target_ids.get("quality")
                    )
                    if q_targets is not None:
                        preds = task_logits.squeeze(-1).detach().cpu()
                        targs = q_targets.detach().cpu()
                        masks = target_masks["quality"].detach().cpu()
                        all_preds.setdefault("quality", []).append(preds)
                        all_targets.setdefault("quality", []).append(targs)
                        all_masks.setdefault("quality", []).append(masks)
            elif task_name in target_ids and task_name in target_masks:
                preds = task_logits.argmax(dim=-1).detach().cpu()
                targs = target_ids[task_name].detach().cpu()
                masks = target_masks[task_name].detach().cpu()

                all_preds.setdefault(task_name, []).append(preds)
                all_targets.setdefault(task_name, []).append(targs)
                all_masks.setdefault(task_name, []).append(masks)

    if batch_count == 0:
        raise ValueError("the data loader produced no non-empty batches")

    mean_task_losses = {
        task_name: round(loss_sum / task_target_counts[task_name], 4)
        for task_name, loss_sum in task_loss_sums.items()
        if task_target_counts[task_name] > 0
    }

    task_accuracies: dict[str, float] = {}
    task_macro_f1: dict[str, float] = {}
    task_mae: dict[str, float] = {}
    task_rmse: dict[str, float] = {}

    for task_name in all_preds:
        preds = torch.cat(all_preds[task_name], dim=0)
        targets = torch.cat(all_targets[task_name], dim=0)
        masks = torch.cat(all_masks[task_name], dim=0)

        valid_preds = preds[masks]
        valid_targets = targets[masks]
        num_valid = len(valid_targets)

        if num_valid == 0:
            if task_name == "quality":
                task_mae[task_name] = 0.0
                task_rmse[task_name] = 0.0
            else:
                task_accuracies[task_name] = 0.0
                task_macro_f1[task_name] = 0.0
            continue

        if task_name == "quality":
            mae = float(torch.abs(valid_preds - valid_targets).mean().item())
            rmse = float(torch.sqrt(torch.mean((valid_preds - valid_targets) ** 2)).item())
            task_mae[task_name] = round(mae, 4)
            task_rmse[task_name] = round(rmse, 4)
            continue

        acc = float((valid_preds == valid_targets).sum().item()) / num_valid
        task_accuracies[task_name] = round(acc, 4)

        unique_classes = torch.unique(torch.cat([valid_preds, valid_targets])).tolist()
        f1_scores: list[float] = []
        for c in unique_classes:
            tp = float(((valid_preds == c) & (valid_targets == c)).sum().item())
            fp = float(((valid_preds == c) & (valid_targets != c)).sum().item())
            fn = float(((valid_preds != c) & (valid_targets == c)).sum().item())
            support = tp + fn
            if support == 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        task_macro_f1[task_name] = round(macro_f1, 4)

    return {
        "loss": round(total_loss_sum / sample_count, 4),
        "task_losses": mean_task_losses,
        "task_accuracies": task_accuracies,
        "task_macro_f1": task_macro_f1,
        "task_mae": task_mae,
        "task_rmse": task_rmse,
        "task_sample_counts": task_target_counts,
        "sample_count": sample_count,
        "batch_count": batch_count,
    }


