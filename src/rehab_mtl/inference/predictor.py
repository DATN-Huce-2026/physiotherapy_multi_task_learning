"""Load checkpoint ``KeraalSharedBackbone`` và dự đoán cho một chuỗi khung xương."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rehab_mtl.models.shared_backbone import KeraalSharedBackbone


def load_model(path: str | Path, *, pooling: str = "mean") -> KeraalSharedBackbone:
    """Load state_dict an toàn (``weights_only``), chuyển sang chế độ eval."""
    model = KeraalSharedBackbone(pooling=pooling)
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _forward(model: KeraalSharedBackbone, xyz: np.ndarray) -> dict[str, torch.Tensor]:
    # Giống lúc train: lấy SpineBase (khớp 0) của frame đầu làm gốc tọa độ.
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz = xyz - xyz[0, 0]
    return model(
        torch.from_numpy(xyz).unsqueeze(0),
        torch.tensor([len(xyz)], dtype=torch.long),
    )


def predict(
    model: KeraalSharedBackbone,
    keraal_sequence: np.ndarray,
    kimore_sequence: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], float | None]:
    """Trả (xác suất softmax của các head phân loại, điểm quality trong [0, 1]).

    Các head phân loại học từ KERAAL (10 fps) nên dùng ``keraal_sequence``.
    Head ``quality`` học từ KiMoRe (30 fps) nên dùng ``kimore_sequence`` nếu có.
    """
    outputs = _forward(model, keraal_sequence)
    probabilities = {
        task: torch.softmax(value[0], dim=-1).numpy()
        for task, value in outputs.items()
        if task != "quality"
    }
    quality_outputs = (
        outputs if kimore_sequence is None else _forward(model, kimore_sequence)
    )
    quality = quality_outputs.get("quality")
    return probabilities, (float(quality[0]) if quality is not None else None)
