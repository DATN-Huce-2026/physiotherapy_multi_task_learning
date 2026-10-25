"""Clip-level shared-backbone model for KERAAL Kinect XYZ sequences."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rehab_mtl.datasets.label_maps import (
    BODY_PART_TO_ID,
    ERROR_TYPE_TO_ID,
    EXERCISE_ERROR_CLASS_TO_ID,
    EXERCISE_TO_ID,
    GLOBAL_EVALUATION_TO_ID,
    SEVERITY_TO_ID,
)


TASK_MAPPINGS = {
    "exercise": EXERCISE_TO_ID,
    "exercise_error_class": EXERCISE_ERROR_CLASS_TO_ID,
    "global_evaluation": GLOBAL_EVALUATION_TO_ID,
    "error_type": ERROR_TYPE_TO_ID,
    "severity": SEVERITY_TO_ID,
    "body_part": BODY_PART_TO_ID,
}


class KeraalSharedBackbone(nn.Module):
    """Predict clip-level labels from a Kinect XYZ sequence.

    Input ``xyz`` has shape ``[batch, time, 25, 3]``. ``lengths`` gives the
    real frame count for each sequence before batch padding. The frame encoder
    and bidirectional GRU are shared by all task heads.

    Supports Masked Global Average Pooling over all valid frames to capture
    motion deviations throughout the entire exercise clip (rather than only the final frame).
    """

    def __init__(
        self,
        *,
        input_dim: int = 12,  # [BỔ SUNG MỚI]: Tham số input_dim=12 cho 12 góc lâm sàng (thay thế 75 tọa độ)
        num_joints: int | None = None,  # [TÙY CHỌN]: Giữ tương thích ngược với mô hình cũ
        coordinates_per_joint: int | None = None,
        frame_embedding_size: int = 64,
        recurrent_hidden_size: int = 64,
        dropout: float = 0.4,
        pooling: str = "mean",  # "mean" (Masked Global Average Pooling) or "last"
    ) -> None:
        super().__init__()
        # [CHỈNH SỬA]: Xác định kích thước frame đầu vào: 12 (góc) hoặc 25*3=75 (tọa độ cũ)
        if num_joints is not None and coordinates_per_joint is not None:
            self.input_dim = num_joints * coordinates_per_joint
            self.num_joints = num_joints
            self.coordinates_per_joint = coordinates_per_joint
        else:
            self.input_dim = input_dim
            self.num_joints = None
            self.coordinates_per_joint = None

        self.pooling = pooling

        # [CHỈNH SỬA]: Lớp Linear đầu vào nhận self.input_dim (12 hoặc 75)
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.input_dim, frame_embedding_size),
            nn.LayerNorm(frame_embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.shared_temporal_backbone = nn.GRU(
            input_size=frame_embedding_size,
            hidden_size=recurrent_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.feature_dropout = nn.Dropout(dropout)

        shared_feature_size = recurrent_hidden_size * 2
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Linear(shared_feature_size, len(class_to_id))
                for task_name, class_to_id in TASK_MAPPINGS.items()
            }
        )

        # Quality assessment regression head (predicts normalized score in [0.0, 1.0])
        self.quality_head = nn.Sequential(
            nn.Linear(shared_feature_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # [CHỈNH SỬA]: Hỗ trợ cả tensor 3D [batch, time, 12] (12 góc) lẫn 4D [batch, time, 25, 3] (tọa độ cũ)
        if xyz.ndim == 3:
            # Dạng mới: [batch_size, time_steps, input_dim] (ví dụ [batch, 90, 12])
            batch_size, padded_time, feat_dim = xyz.shape
            if feat_dim != self.input_dim:
                raise ValueError(
                    f"Frame feature dim is {feat_dim}, expected input_dim={self.input_dim}"
                )
            frame_features = xyz
        elif xyz.ndim == 4:
            # Dạng cũ: [batch_size, time_steps, joints, coordinates]
            batch_size, padded_time, joint_count, coordinate_count = xyz.shape
            if self.num_joints is not None and self.coordinates_per_joint is not None:
                expected_shape = (self.num_joints, self.coordinates_per_joint)
                if (joint_count, coordinate_count) != expected_shape:
                    raise ValueError(
                        f"xyz ends with {(joint_count, coordinate_count)}, expected {expected_shape}"
                    )
            frame_features = xyz.reshape(batch_size, padded_time, -1)
            if frame_features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Flattened frame feature dim is {frame_features.shape[-1]}, "
                    f"expected input_dim={self.input_dim}"
                )
        else:
            raise ValueError(
                f"xyz must have shape [batch, time, 12] or [batch, time, 25, 3], got {xyz.shape}"
            )

        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError("lengths must have one value per batch item")
        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
        if bool((lengths_cpu < 1).any()):
            raise ValueError("every sequence length must be at least 1")
        if bool((lengths_cpu > padded_time).any()):
            raise ValueError("a sequence length cannot exceed the padded time size")

        # [CHỈNH SỬA]: Đưa qua frame_encoder để ánh xạ từ input_dim sang frame_embedding_size (64)
        frame_features = self.frame_encoder(frame_features)

        # Packing tells the GRU to ignore padded frames in each sequence.
        packed_features = pack_padded_sequence(
            frame_features,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, final_hidden = self.shared_temporal_backbone(packed_features)

        if self.pooling == "mean":
            # Unpack GRU sequence outputs: [batch, padded_time, 2 * hidden_size]
            gru_outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=padded_time,
            )

            # Masked Global Average Pooling across valid frames (ignoring zero pads)
            time_indices = torch.arange(padded_time, device=xyz.device).unsqueeze(0)
            time_mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1)
            masked_outputs = gru_outputs * time_mask
            summed = masked_outputs.sum(dim=1)
            pooled_features = summed / lengths.unsqueeze(1).clamp(min=1).to(dtype=summed.dtype)
            shared_features = self.feature_dropout(pooled_features)
        else:
            # Fallback: concatenate the final forward and backward summaries
            shared_features = torch.cat(
                (final_hidden[-2], final_hidden[-1]),
                dim=1,
            )
            shared_features = self.feature_dropout(shared_features)

        outputs = {
            task_name: head(shared_features)
            for task_name, head in self.heads.items()
        }
        outputs["quality"] = self.quality_head(shared_features).squeeze(-1)
        return outputs
