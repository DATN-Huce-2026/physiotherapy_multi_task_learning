"""Load model và dự đoán cho một chuỗi khung xương."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rehab_mtl.features.angles import NUM_ANGLES, transform_kinect_sequence_to_angles  # [BỔ SUNG MỚI]
from rehab_mtl.models.shared_backbone import KeraalSharedBackbone


def load_model(path: str | Path, *, pooling: str = "mean") -> KeraalSharedBackbone:
    """Load model checkpoint và tự động nhận diện input_dim (góc hoặc 75 tọa độ)."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    # [BỔ SUNG]: Tự động phát hiện kích thước lớp đầu vào từ trọng số checkpoint
    in_features = state.get("frame_encoder.0.weight", torch.empty(0, NUM_ANGLES)).shape[1] if "frame_encoder.0.weight" in state else NUM_ANGLES
    model = KeraalSharedBackbone(input_dim=in_features, pooling=pooling)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _forward(model: KeraalSharedBackbone, sequence: np.ndarray) -> dict[str, torch.Tensor]:
    """Forward một chuỗi qua mô hình (tự động chuyển sang góc & 90 frames nếu model dùng góc)."""
    seq = np.asarray(sequence, dtype=np.float32)

    # [BỔ SUNG / CHỈNH SỬA]: Nếu mô hình học trên góc (13 hoặc checkpoint cũ 12 góc)
    if model.input_dim in (NUM_ANGLES, 12):
        if seq.ndim == 3 and seq.shape[1] == 25:
            # Chuỗi [T, 25, 3] -> Chuyển thành góc và nội suy về 90 frames
            features = transform_kinect_sequence_to_angles(seq, target_length=90)[:, :model.input_dim]
        elif seq.ndim == 2 and seq.shape[1] == model.input_dim:
            features = seq if len(seq) == 90 else transform_kinect_sequence_to_angles(seq, target_length=90)[:, :model.input_dim]
        elif seq.ndim == 2 and seq.shape[1] == NUM_ANGLES and model.input_dim == 12:
            features = seq[:, :12]
        else:
            raise ValueError(f"Không thể chuyển đổi chuỗi shape {seq.shape} sang {model.input_dim} góc")
        tensor_in = torch.from_numpy(features).unsqueeze(0)  # [1, 90, input_dim]
        lengths = torch.tensor([90], dtype=torch.long)
        return model(tensor_in, lengths)

    # [GIỮ NGUYÊN]: Trường hợp mô hình cũ dùng 75 tọa độ XYZ
    xyz = seq - seq[0, 0]
    return model(
        torch.from_numpy(xyz).unsqueeze(0),
        torch.tensor([len(xyz)], dtype=torch.long),
    )


def predict(
    model: KeraalSharedBackbone,
    keraal_sequence: np.ndarray,
    kimore_sequence: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], float | None]:
    """Trả (xác suất các head phân loại, điểm quality 0-1).

    Head phân loại dùng chuỗi 10 fps (KERAAL), head quality dùng chuỗi 30 fps (KiMoRe).
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
