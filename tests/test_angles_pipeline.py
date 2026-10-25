"""Kiểm thử tính đúng đắn của pipeline trích xuất 12 góc khớp và nội suy 90 frames."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from rehab_mtl.datasets.keraal_dataset import collate_keraal_mtl
from rehab_mtl.datasets.kimore_dataset import collate_kimore_mtl
from rehab_mtl.features.angles import (
    calculate_angle_3d,
    extract_angles_from_kinect_frame,
    resample_sequence_to_fixed_length,
    transform_kinect_sequence_to_angles,
)
from rehab_mtl.models.shared_backbone import KeraalSharedBackbone


class TestAnglesPipeline(unittest.TestCase):
    def test_calculate_angle_3d_known_values(self):
        # Góc vuông 90 độ (pi / 2 rad)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        angle = calculate_angle_3d(a, b, c)
        self.assertAlmostEqual(angle, math.pi / 2, places=5)

        # Góc thẳng 180 độ (pi rad)
        c_straight = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        angle_straight = calculate_angle_3d(a, b, c_straight)
        self.assertAlmostEqual(angle_straight, math.pi, places=5)

        # Trùng điểm (suy biến) không bị crash / nan
        angle_zero = calculate_angle_3d(b, b, c)
        self.assertEqual(angle_zero, 0.0)

    def test_extract_angles_from_kinect_frame(self):
        rng = np.random.default_rng(42)
        frame = rng.normal(size=(25, 3)).astype(np.float32)

        angles = extract_angles_from_kinect_frame(frame)
        self.assertEqual(angles.shape, (12,))
        self.assertTrue(np.all(angles >= 0.0))
        self.assertTrue(np.all(angles <= math.pi))

    def test_resample_sequence_to_fixed_length(self):
        rng = np.random.default_rng(42)
        # Chuỗi dài 50 frames
        seq_short = rng.normal(size=(50, 12)).astype(np.float32)
        resampled_50 = resample_sequence_to_fixed_length(seq_short, target_length=90)
        self.assertEqual(resampled_50.shape, (90, 12))

        # Chuỗi dài 150 frames
        seq_long = rng.normal(size=(150, 12)).astype(np.float32)
        resampled_150 = resample_sequence_to_fixed_length(seq_long, target_length=90)
        self.assertEqual(resampled_150.shape, (90, 12))

    def test_transform_kinect_sequence_to_angles(self):
        rng = np.random.default_rng(42)
        sequence_xyz = rng.normal(size=(110, 25, 3)).astype(np.float32)

        result = transform_kinect_sequence_to_angles(sequence_xyz, target_length=90)
        self.assertEqual(result.shape, (90, 12))
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(bool(np.isnan(result).any()))

    def test_shared_backbone_with_12_angles(self):
        model = KeraalSharedBackbone(input_dim=12)
        batch_size = 4
        x = torch.randn(batch_size, 90, 12)
        lengths = torch.tensor([90] * batch_size, dtype=torch.long)

        outputs = model(x, lengths)
        self.assertIn("exercise", outputs)
        self.assertIn("quality", outputs)
        self.assertEqual(outputs["exercise"].shape, (batch_size, 3))
        self.assertEqual(outputs["quality"].shape, (batch_size,))

    def test_collate_fn_with_12_angles(self):
        items = [
            {
                "sample_id": f"sample_{i}",
                "xyz": torch.randn(90, 12),
                "length": 90,
                "target_ids": {"exercise": 0, "global_evaluation": 1},
                "target_masks": {"exercise": True, "global_evaluation": True},
                "quality_target": torch.tensor(0.75, dtype=torch.float32),
                "exercise_code": "CTK",
            }
            for i in range(3)
        ]
        batch = collate_keraal_mtl(items)
        self.assertEqual(batch["xyz"].shape, (3, 90, 12))
        self.assertEqual(batch["lengths"].tolist(), [90, 90, 90])

        batch_kimore = collate_kimore_mtl(items)
        self.assertEqual(batch_kimore["xyz"].shape, (3, 90, 12))
        self.assertEqual(batch_kimore["lengths"].tolist(), [90, 90, 90])


if __name__ == "__main__":
    unittest.main()
