"""Kiểm thử tính đúng đắn của pipeline trích xuất 12 góc khớp và nội suy 90 frames."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from rehab_mtl.datasets.keraal_dataset import collate_keraal_mtl
from rehab_mtl.datasets.kimore_dataset import collate_kimore_mtl
from rehab_mtl.features.angles import (
    ANGLE_NAMES,
    HIP_LEFT,
    HIP_RIGHT,
    NUM_ANGLES,
    SHOULDER_LEFT,
    SHOULDER_RIGHT,
    SPINE_BASE,
    SPINE_SHOULDER,
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
        self.assertEqual(angles.shape, (NUM_ANGLES,))
        self.assertTrue(np.all(angles >= 0.0))
        self.assertTrue(np.all(angles <= math.pi))

    def test_trunk_tilt_symmetry_and_torso_twist(self):
        tilt_idx = ANGLE_NAMES.index("trunk_tilt")
        twist_idx = ANGLE_NAMES.index("torso_twist")

        def make_pose(tilt_x: float = 0.0, shoulder_yaw_rad: float = 0.0) -> np.ndarray:
            frame = np.zeros((25, 3), dtype=np.float32)
            frame[SPINE_BASE] = [0.0, 0.0, 0.0]
            frame[HIP_LEFT] = [-0.15, 0.0, 0.0]
            frame[HIP_RIGHT] = [0.15, 0.0, 0.0]
            frame[SPINE_SHOULDER] = [tilt_x, 0.5, 0.0]
            half_w = 0.2
            dx = half_w * math.cos(shoulder_yaw_rad)
            dz = half_w * math.sin(shoulder_yaw_rad)
            frame[SHOULDER_LEFT] = [tilt_x - dx, 0.5, -dz]
            frame[SHOULDER_RIGHT] = [tilt_x + dx, 0.5, dz]
            return frame

        tilt_left = extract_angles_from_kinect_frame(make_pose(tilt_x=-0.25))
        tilt_right = extract_angles_from_kinect_frame(make_pose(tilt_x=0.25))
        self.assertAlmostEqual(float(tilt_left[tilt_idx]), float(tilt_right[tilt_idx]), places=5)
        self.assertGreater(float(tilt_left[tilt_idx]), 0.4)
        self.assertAlmostEqual(float(tilt_left[twist_idx]), 0.0, places=3)

        rotated = extract_angles_from_kinect_frame(make_pose(shoulder_yaw_rad=math.pi / 4))
        self.assertAlmostEqual(float(rotated[tilt_idx]), 0.0, places=3)
        self.assertAlmostEqual(float(rotated[twist_idx]), math.pi / 4, places=5)

    def test_resample_sequence_to_fixed_length(self):
        rng = np.random.default_rng(42)
        # Chuỗi dài 50 frames
        seq_short = rng.normal(size=(50, NUM_ANGLES)).astype(np.float32)
        resampled_50 = resample_sequence_to_fixed_length(seq_short, target_length=90)
        self.assertEqual(resampled_50.shape, (90, NUM_ANGLES))

        # Chuỗi dài 150 frames
        seq_long = rng.normal(size=(150, NUM_ANGLES)).astype(np.float32)
        resampled_150 = resample_sequence_to_fixed_length(seq_long, target_length=90)
        self.assertEqual(resampled_150.shape, (90, NUM_ANGLES))

    def test_transform_kinect_sequence_to_angles(self):
        rng = np.random.default_rng(42)
        sequence_xyz = rng.normal(size=(110, 25, 3)).astype(np.float32)

        result = transform_kinect_sequence_to_angles(sequence_xyz, target_length=90)
        self.assertEqual(result.shape, (90, NUM_ANGLES))
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(bool(np.isnan(result).any()))

    def test_shared_backbone_with_angles(self):
        model = KeraalSharedBackbone(input_dim=NUM_ANGLES)
        batch_size = 4
        x = torch.randn(batch_size, 90, NUM_ANGLES)
        lengths = torch.tensor([90] * batch_size, dtype=torch.long)

        outputs = model(x, lengths)
        self.assertIn("exercise", outputs)
        self.assertIn("quality", outputs)
        self.assertEqual(outputs["exercise"].shape, (batch_size, 3))
        self.assertEqual(outputs["quality"].shape, (batch_size,))

    def test_collate_fn_with_angles(self):
        items = [
            {
                "sample_id": f"sample_{i}",
                "xyz": torch.randn(90, NUM_ANGLES),
                "length": 90,
                "target_ids": {"exercise": 0, "global_evaluation": 1},
                "target_masks": {"exercise": True, "global_evaluation": True},
                "quality_target": torch.tensor(0.75, dtype=torch.float32),
                "exercise_code": "CTK",
            }
            for i in range(3)
        ]
        batch = collate_keraal_mtl(items)
        self.assertEqual(batch["xyz"].shape, (3, 90, NUM_ANGLES))
        self.assertEqual(batch["lengths"].tolist(), [90, 90, 90])

        batch_kimore = collate_kimore_mtl(items)
        self.assertEqual(batch_kimore["xyz"].shape, (3, 90, NUM_ANGLES))
        self.assertEqual(batch_kimore["lengths"].tolist(), [90, 90, 90])

    def test_blazepose_json_loader_and_keraal_dataset_sources(self):
        import csv
        import json
        import tempfile
        from pathlib import Path
        from rehab_mtl.datasets.keraal import BLAZEPOSE_LANDMARK_NAMES, load_blazepose_xyz
        from rehab_mtl.datasets.keraal_dataset import KeraalMTLDataset

        def make_bp_frame(offset: float) -> dict[str, list[float]]:
            frame = {}
            for idx, name in enumerate(BLAZEPOSE_LANDMARK_NAMES):
                frame[name] = [0.5 + 0.01 * idx + offset, 0.5 + 0.01 * idx, -0.1]
            return frame

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "group1A" / "kinect").mkdir(parents=True)
            (root / "group1A" / "blazepose").mkdir(parents=True)

            kinect_line = " ".join("0.1" for _ in range(25 * 7))
            for fname in ("k1.txt", "k2.txt"):
                (root / "group1A" / "kinect" / fname).write_text(
                    f"{kinect_line}\n{kinect_line}\n", encoding="utf-8"
                )

            bp_payload = {
                "positions": {
                    "10.0": make_bp_frame(0.3),
                    "5.0": {},
                    "1.0": make_bp_frame(0.1),
                    "2.0": make_bp_frame(0.2),
                }
            }
            bp_path = root / "group1A" / "blazepose" / "bp1.json"
            bp_path.write_text(json.dumps(bp_payload), encoding="utf-8")

            bp_empty_path = root / "group1A" / "blazepose" / "bp_empty.json"
            bp_empty_path.write_text(
                json.dumps({"positions": {str(i): {} for i in range(20)}}),
                encoding="utf-8",
            )

            xyz = load_blazepose_xyz(bp_path)
            self.assertEqual(len(xyz), 3)
            self.assertEqual(len(xyz[0]), 25)
            self.assertEqual(len(xyz[0][0]), 3)
            self.assertNotEqual(xyz[0], xyz[2])

            manifest_path = root / "manifest.csv"
            targets_path = root / "targets.csv"
            splits_path = root / "splits.csv"

            manifest_fields = [
                "sample_id", "group", "subject_token", "exercise_code",
                "kinect_path", "kinect_frame_count", "blazepose_path",
                "annotator_a_global_evaluation", "annotator_a_error_type",
                "annotator_a_severity", "annotator_a_body_part",
                "annotator_a_temporal_error_status", "annotator_a_temporal_error_event_count",
                "annotator_b_global_evaluation", "annotator_b_error_type",
                "annotator_b_severity", "annotator_b_body_part",
                "annotator_b_temporal_error_status", "annotator_b_temporal_error_event_count",
            ]
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=manifest_fields)
                writer.writeheader()
                for sid, ex_code, kf, bf in [
                    ("S1", "ELK", "group1A/kinect/k1.txt", "group1A/blazepose/bp1.json"),
                    ("S2", "CTK", "group1A/kinect/k2.txt", "group1A/blazepose/bp_empty.json"),
                ]:
                    writer.writerow({
                        "sample_id": sid, "group": "group1A", "subject_token": "R1",
                        "exercise_code": ex_code, "kinect_path": kf, "kinect_frame_count": "2",
                        "blazepose_path": bf,
                        "annotator_a_global_evaluation": "Correct", "annotator_a_error_type": "none",
                        "annotator_a_severity": "", "annotator_a_body_part": "",
                        "annotator_a_temporal_error_status": "track_absent",
                        "annotator_a_temporal_error_event_count": "",
                        "annotator_b_global_evaluation": "Correct", "annotator_b_error_type": "none",
                        "annotator_b_severity": "", "annotator_b_body_part": "",
                        "annotator_b_temporal_error_status": "track_absent",
                        "annotator_b_temporal_error_event_count": "",
                    })

            target_fields = [
                "sample_id", "review_flags",
                "exercise_target", "exercise_mask", "exercise_resolution",
                "exercise_error_class_target", "exercise_error_class_mask", "exercise_error_class_resolution",
                "global_evaluation_target", "global_evaluation_mask", "global_evaluation_resolution",
                "error_type_target", "error_type_mask", "error_type_resolution",
                "severity_target", "severity_mask", "severity_resolution",
                "body_part_target", "body_part_mask", "body_part_resolution",
            ]
            with targets_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=target_fields)
                writer.writeheader()
                for sid, ex_code in (("S1", "ELK"), ("S2", "CTK")):
                    writer.writerow({
                        "sample_id": sid, "review_flags": "",
                        "exercise_target": ex_code, "exercise_mask": "true", "exercise_resolution": "manifest_metadata",
                        "exercise_error_class_target": "Correct", "exercise_error_class_mask": "true", "exercise_error_class_resolution": "both_agree",
                        "global_evaluation_target": "Correct", "global_evaluation_mask": "true", "global_evaluation_resolution": "both_agree",
                        "error_type_target": "none", "error_type_mask": "true", "error_type_resolution": "both_agree",
                        "severity_target": "", "severity_mask": "false", "severity_resolution": "not_applicable",
                        "body_part_target": "", "body_part_mask": "false", "body_part_resolution": "not_applicable",
                    })

            with splits_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_id", "split"])
                writer.writeheader()
                writer.writerow({"sample_id": "S1", "split": "train"})
                writer.writerow({"sample_id": "S2", "split": "train"})

            ds_kinect = KeraalMTLDataset(
                root, split="train", manifest_path=manifest_path,
                targets_path=targets_path, split_path=splits_path,
                skeleton_source="kinect",
            )
            ds_bp = KeraalMTLDataset(
                root, split="train", manifest_path=manifest_path,
                targets_path=targets_path, split_path=splits_path,
                skeleton_source="blazepose",
            )
            ds_both = KeraalMTLDataset(
                root, split="train", manifest_path=manifest_path,
                targets_path=targets_path, split_path=splits_path,
                skeleton_source="both", preload=True,
            )

            self.assertEqual(len(ds_kinect), 2)
            self.assertEqual(len(ds_bp), 1)
            self.assertEqual(len(ds_both), 3)
            self.assertEqual(ds_bp[0]["skeleton_source"], "blazepose")
            self.assertEqual(ds_bp[0]["xyz"].shape, (90, NUM_ANGLES))
            self.assertEqual({ds_both[i]["skeleton_source"] for i in range(len(ds_both))}, {"kinect", "blazepose"})

            ds_filtered = KeraalMTLDataset(
                root, split="train", manifest_path=manifest_path,
                targets_path=targets_path, split_path=splits_path,
                skeleton_source="kinect",
                allowed_exercises=["ELK", "RTK"],
            )
            self.assertEqual(len(ds_filtered), 1)
            self.assertEqual(ds_filtered[0]["exercise_code"], "ELK")


if __name__ == "__main__":
    unittest.main()
