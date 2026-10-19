import unittest
from types import SimpleNamespace

import numpy as np

from rehab_mtl.inference.mediapipe_to_kinect import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    mediapipe_world_to_kinect,
    resample_sequence,
)
from rehab_mtl.inference.recorder import SkeletonRecorder


def standing_pose():
    """Người đứng đối diện camera, tọa độ MediaPipe world (y hướng xuống)."""
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[LEFT_HIP] = (0.1, 0.0, 0.0)  # bên trái người = +x MediaPipe
    landmarks[RIGHT_HIP] = (-0.1, 0.0, 0.0)
    landmarks[LEFT_SHOULDER] = (0.18, -0.5, 0.0)  # vai cao hơn hông = y âm
    landmarks[RIGHT_SHOULDER] = (-0.18, -0.5, 0.0)
    landmarks[7] = (0.07, -0.7, 0.0)  # tai trái
    landmarks[8] = (-0.07, -0.7, 0.0)
    landmarks[LEFT_WRIST] = (0.3, -0.1, -0.2)  # tay trái đưa về phía camera
    return landmarks


def fake_results(landmarks, visibility=1.0):
    points = [
        SimpleNamespace(x=float(x), y=float(y), z=float(z), visibility=visibility)
        for x, y, z in landmarks
    ]
    return SimpleNamespace(pose_world_landmarks=SimpleNamespace(landmark=points))


class TestMediapipeToKinect(unittest.TestCase):
    def test_doi_truc_sang_kinect(self):
        kinect = mediapipe_world_to_kinect(standing_pose(), pitch_deg=0)

        # Kinect: vai trái có x âm (bên phải người là +x), vai cao hơn hông (y dương)
        self.assertLess(kinect[4, 0], kinect[8, 0])
        self.assertGreater(kinect[4, 1], kinect[12, 1])
        # Tay đưa về phía camera -> z giảm (Kinect +z hướng ra xa camera)
        self.assertLess(kinect[6, 2], kinect[4, 2])
        self.assertTrue(np.isfinite(kinect).all())

    def test_khop_noi_suy_nam_dung_ti_le(self):
        kinect = mediapipe_world_to_kinect(standing_pose(), pitch_deg=0)
        spine_base, spine_mid, spine_shoulder = kinect[0], kinect[1], kinect[20]
        ratio = (spine_mid[1] - spine_base[1]) / (spine_shoulder[1] - spine_base[1])
        self.assertAlmostEqual(ratio, 0.57, places=5)
        self.assertGreater(kinect[2, 1], spine_shoulder[1])  # cổ trên vai
        self.assertGreater(kinect[3, 1], kinect[2, 1])  # đầu trên cổ

    def test_pitch_xoay_quanh_truc_x(self):
        straight = mediapipe_world_to_kinect(standing_pose(), pitch_deg=0)
        tilted = mediapipe_world_to_kinect(standing_pose(), pitch_deg=90)
        np.testing.assert_allclose(tilted[:, 0], straight[:, 0], atol=1e-6)
        np.testing.assert_allclose(tilted[:, 2], straight[:, 1], atol=1e-6)


class TestResample(unittest.TestCase):
    def test_lay_mau_10fps_va_lap_frame_mat(self):
        timestamps = np.arange(7) / 30.0  # 30 fps, 0.2 giây
        frames = np.stack([np.full((25, 3), t, dtype=np.float32) for t in range(7)])
        frames[3] = np.nan  # một frame không thấy người

        result = resample_sequence(timestamps, frames, fps=10.0)

        self.assertEqual(result.shape, (3, 25, 3))  # t = 0, 0.1, 0.2
        np.testing.assert_allclose(result[:, 0, 0], [0, 3, 6], atol=1e-4)

    def test_qua_it_frame_thi_bao_loi(self):
        frames = np.full((3, 25, 3), np.nan, dtype=np.float32)
        with self.assertRaises(ValueError):
            resample_sequence(np.arange(3), frames)


class TestSkeletonRecorder(unittest.TestCase):
    def test_ghi_frame_bo_qua_frame_thieu_khop_va_xuat_hai_toc_do(self):
        recorder = SkeletonRecorder(min_visibility=0.5, pitch_deg=0)
        self.assertTrue(recorder.add(0.0, fake_results(standing_pose())))
        self.assertFalse(recorder.add(0.05, fake_results(standing_pose(), 0.1)))
        self.assertFalse(recorder.add(0.1, None))
        self.assertTrue(recorder.add(0.2, fake_results(standing_pose())))

        self.assertEqual(len(recorder), 4)
        self.assertEqual(recorder.visible_frames, 2)
        self.assertAlmostEqual(recorder.duration, 0.2)
        self.assertEqual(recorder.sequence(10.0).shape, (3, 25, 3))  # KERAAL
        self.assertEqual(recorder.sequence(30.0).shape, (7, 25, 3))  # KiMoRe
        recorder.reset()
        self.assertEqual(len(recorder), 0)


if __name__ == "__main__":
    unittest.main()
