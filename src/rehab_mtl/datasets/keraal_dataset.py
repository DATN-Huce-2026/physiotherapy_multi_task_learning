"""PyTorch Dataset and DataLoader integration for KERAAL multi-task learning."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from rehab_mtl.datasets.keraal import load_keraal_mtl_sample
from rehab_mtl.datasets.label_maps import TASK_MAPPINGS, encode_target

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "keraal" / "keraal_manifest.csv"
DEFAULT_TARGETS_PATH = PROJECT_ROOT / "data" / "keraal" / "keraal_targets_preview.csv"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "keraal" / "keraal_subject_split_preview.csv"
DEFAULT_KERAAL_ROOT = Path.home() / "Documents" / "KERAAL"


def _read_csv_by_sample_id(path: Path) -> dict[str, dict[str, str]]:
    """Read a CSV file into a dictionary indexed by sample_id."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"{path}: missing CSV header or sample_id column")
        rows = list(reader)

    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"].strip()
        if sample_id in result:
            raise ValueError(f"{path}: duplicate sample_id {sample_id!r}")
        result[sample_id] = row
    return result


class KeraalMTLDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset for multi-task learning with KERAAL skeleton sequences.

    Filters samples by the specified train/validation/test split. For each sample,
    it returns the 3D skeleton coordinates [T, 25, 3] and mapped multi-task target IDs
    along with their validity masks.
    """

    def __init__(
        self,
        keraal_root: str | Path = DEFAULT_KERAAL_ROOT,
        *,
        split: str | None = None,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        targets_path: str | Path = DEFAULT_TARGETS_PATH,
        split_path: str | Path = DEFAULT_SPLIT_PATH,
        relative_to_initial_spine_base: bool = True,
        task_mappings: Mapping[str, Mapping[str, int]] | None = None,
        preload: bool = False,
    ) -> None:
        super().__init__()
        self.keraal_root = Path(keraal_root)
        self.manifest_path = Path(manifest_path)
        self.targets_path = Path(targets_path)
        self.split_path = Path(split_path)
        self.split = split
        self.relative_to_initial_spine_base = relative_to_initial_spine_base
        self.task_mappings = dict(task_mappings or TASK_MAPPINGS)
        self.preload = preload

        self.manifest_rows = _read_csv_by_sample_id(self.manifest_path)
        self.target_rows = _read_csv_by_sample_id(self.targets_path)
        self.split_rows = _read_csv_by_sample_id(self.split_path)

        # Filter sample IDs by split if requested
        if self.split is not None:
            valid_splits = {"train", "validation", "test"}
            if self.split not in valid_splits:
                raise ValueError(
                    f"invalid split: {self.split!r}; expected one of {sorted(valid_splits)}"
                )
            self.sample_ids = [
                sample_id
                for sample_id, split_row in self.split_rows.items()
                if split_row.get("split") == self.split
            ]
        else:
            self.sample_ids = list(self.manifest_rows.keys())

        # Validate that all required sample_ids exist across files
        for sample_id in self.sample_ids:
            if sample_id not in self.manifest_rows:
                raise ValueError(f"sample_id {sample_id!r} not found in manifest")
            if sample_id not in self.target_rows:
                raise ValueError(f"sample_id {sample_id!r} not found in targets")

        self._cache: dict[str, dict[str, Any]] = {}
        if self.preload:
            for idx in range(len(self.sample_ids)):
                self[idx]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        if self.preload and sample_id in self._cache:
            return self._cache[sample_id]

        manifest_row = self.manifest_rows[sample_id]
        target_row = self.target_rows[sample_id]

        sample = load_keraal_mtl_sample(
            manifest_row,
            target_row,
            self.keraal_root,
            relative_to_initial_spine_base=self.relative_to_initial_spine_base,
        )

        xyz_tensor = torch.tensor(sample["xyz"], dtype=torch.float32)

        target_ids: dict[str, int] = {}
        target_masks: dict[str, bool] = {}

        for task_name, mapping in self.task_mappings.items():
            raw_target = sample["targets"].get(task_name)
            is_valid = bool(sample["target_masks"].get(task_name, False))
            target_masks[task_name] = is_valid

            if is_valid:
                encoded = encode_target(raw_target, mapping, mask=True)
                assert encoded is not None
                target_ids[task_name] = encoded
            else:
                # Placeholder integer for masked target; masked out during loss calculation
                target_ids[task_name] = 0

        # KERAAL does not contain continuous clinical quality scores
        target_masks["quality"] = False

        item = {
            "sample_id": sample_id,
            "xyz": xyz_tensor,
            "length": len(sample["xyz"]),
            "target_ids": target_ids,
            "target_masks": target_masks,
            "quality_target": torch.tensor(0.0, dtype=torch.float32),
            "group": sample["group"],
            "subject_id": sample["subject_id"],
            "exercise_code": sample["exercise_code"],
        }

        if self.preload:
            self._cache[sample_id] = item

        return item


def collate_keraal_mtl(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length KERAAL sequence items into a padded batch.

    Pads the time dimension of `xyz` sequences with zeros up to the maximum sequence
    length in the batch. Formats `target_ids` and `target_masks` as dictionary of tensors
    directly compatible with `KeraalSharedBackbone` and `masked_multitask_cross_entropy`.
    """
    if not batch:
        raise ValueError("batch cannot be empty")

    batch_size = len(batch)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    # Shape: [batch, max_len, 25, 3]
    num_joints = batch[0]["xyz"].shape[1]
    coordinates = batch[0]["xyz"].shape[2]
    padded_xyz = torch.zeros(
        batch_size, max_len, num_joints, coordinates, dtype=torch.float32
    )

    for i, item in enumerate(batch):
        seq_len = item["length"]
        padded_xyz[i, :seq_len] = item["xyz"]

    task_names = list(batch[0]["target_ids"].keys())
    target_ids = {
        task: torch.tensor(
            [item["target_ids"][task] for item in batch], dtype=torch.long
        )
        for task in task_names
    }
    all_mask_keys = set(batch[0]["target_masks"].keys())
    target_masks = {
        task: torch.tensor(
            [item["target_masks"].get(task, False) for item in batch], dtype=torch.bool
        )
        for task in all_mask_keys
    }
    regression_targets = {
        "quality": torch.tensor(
            [float(item.get("quality_target", 0.0)) for item in batch], dtype=torch.float32
        )
    }

    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "xyz": padded_xyz,
        "lengths": lengths,
        "target_ids": target_ids,
        "target_masks": target_masks,
        "regression_targets": regression_targets,
        "exercise_codes": [item["exercise_code"] for item in batch],
    }


def build_keraal_dataloaders(
    keraal_root: str | Path = DEFAULT_KERAAL_ROOT,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    targets_path: str | Path = DEFAULT_TARGETS_PATH,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    batch_size: int = 16,
    eval_batch_size: int | None = None,
    relative_to_initial_spine_base: bool = True,
    task_mappings: Mapping[str, Mapping[str, int]] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    preload: bool = False,
) -> dict[str, DataLoader[dict[str, Any]]]:
    """Construct train, validation, and test DataLoaders for KERAAL MTL.

    Returns a dictionary with keys 'train', 'validation', and 'test'.
    """
    eval_bs = eval_batch_size if eval_batch_size is not None else batch_size

    train_dataset = KeraalMTLDataset(
        keraal_root,
        split="train",
        manifest_path=manifest_path,
        targets_path=targets_path,
        split_path=split_path,
        relative_to_initial_spine_base=relative_to_initial_spine_base,
        task_mappings=task_mappings,
        preload=preload,
    )
    val_dataset = KeraalMTLDataset(
        keraal_root,
        split="validation",
        manifest_path=manifest_path,
        targets_path=targets_path,
        split_path=split_path,
        relative_to_initial_spine_base=relative_to_initial_spine_base,
        task_mappings=task_mappings,
        preload=preload,
    )
    test_dataset = KeraalMTLDataset(
        keraal_root,
        split="test",
        manifest_path=manifest_path,
        targets_path=targets_path,
        split_path=split_path,
        relative_to_initial_spine_base=relative_to_initial_spine_base,
        task_mappings=task_mappings,
        preload=preload,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_keraal_mtl,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collate_keraal_mtl,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collate_keraal_mtl,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }
