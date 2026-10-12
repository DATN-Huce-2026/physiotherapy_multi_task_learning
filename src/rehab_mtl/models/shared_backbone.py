"""Clip-level shared-backbone model for KERAAL Kinect XYZ sequences."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from rehab_mtl.datasets.label_maps import (
    BODY_PART_TO_ID,
    ERROR_TYPE_TO_ID,
    EXERCISE_TO_ID,
    GLOBAL_EVALUATION_TO_ID,
    SEVERITY_TO_ID,
)


TASK_MAPPINGS = {
    "exercise": EXERCISE_TO_ID,
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

    Each head returns raw class logits with shape ``[batch, class_count]``.
    This model predicts one label per whole clip; it does not predict a label
    for each frame or locate the time of an error.
    """

    def __init__(
        self,
        *,
        num_joints: int = 25,
        coordinates_per_joint: int = 3,
        frame_embedding_size: int = 64,
        recurrent_hidden_size: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.coordinates_per_joint = coordinates_per_joint

        frame_input_size = num_joints * coordinates_per_joint
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_input_size, frame_embedding_size),
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

        shared_feature_size = recurrent_hidden_size * 2
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Linear(shared_feature_size, len(class_to_id))
                for task_name, class_to_id in TASK_MAPPINGS.items()
            }
        )

    def forward(
        self,
        xyz: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if xyz.ndim != 4:
            raise ValueError(
                "xyz must have shape [batch, time, joints, coordinates]"
            )

        batch_size, padded_time, joint_count, coordinate_count = xyz.shape
        expected_shape = (self.num_joints, self.coordinates_per_joint)
        if (joint_count, coordinate_count) != expected_shape:
            raise ValueError(
                f"xyz ends with {(joint_count, coordinate_count)}, "
                f"expected {expected_shape}"
            )

        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError("lengths must have one value per batch item")
        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
        if bool((lengths_cpu < 1).any()):
            raise ValueError("every sequence length must be at least 1")
        if bool((lengths_cpu > padded_time).any()):
            raise ValueError("a sequence length cannot exceed the padded time size")

        # Flatten the 25 XYZ joints into 75 values for each frame.
        frame_features = xyz.reshape(batch_size, padded_time, -1)
        frame_features = self.frame_encoder(frame_features)

        # Packing tells the GRU to ignore padded frames in each sequence.
        packed_features = pack_padded_sequence(
            frame_features,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _, final_hidden = self.shared_temporal_backbone(packed_features)

        # For a bidirectional GRU, the final layer's forward and backward
        # summaries are the last two entries in final_hidden.
        shared_features = torch.cat(
            (final_hidden[-2], final_hidden[-1]),
            dim=1,
        )

        return {
            task_name: head(shared_features)
            for task_name, head in self.heads.items()
        }
