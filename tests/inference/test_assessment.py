import unittest

import numpy as np

from rehab_mtl.inference.assessment import TASK_LABELS, assess


def probabilities(**overrides):
    base = {
        "exercise": [0.9, 0.05, 0.05],
        "exercise_error_class": [0.7, 0.1, 0.1, 0.1],
        "global_evaluation": [0.7, 0.2, 0.05, 0.05],
        "error_type": [0.6, 0.1, 0.1, 0.1, 0.1],
        "severity": [0.8, 0.2],
        "body_part": [0.1, 0.7, 0.1, 0.1],
    }
    base.update(overrides)
    return {task: np.asarray(values) for task, values in base.items()}


class TestAssess(unittest.TestCase):
    def test_thu_tu_nhan_lay_tu_label_maps(self):
        self.assertEqual(TASK_LABELS["exercise"], ("CTK", "ELK", "RTK"))
        self.assertEqual(TASK_LABELS["global_evaluation"][0], "Correct")
        self.assertEqual(TASK_LABELS["error_type"][0], "none")

    def test_diem_ket_hop_do_dung_va_quality_strict_mode(self):
        result = assess(
            probabilities(), quality=0.9, expected_exercise="CTK", scoring_mode="strict"
        )

        # độ đúng = (0.7/0.9 + 0.7/0.8 + 0.6/0.7) / 3 = 83.66
        self.assertAlmostEqual(result["correctness"], 83.7, places=1)
        self.assertEqual(result["quality"], 90.0)
        self.assertAlmostEqual(result["score"], 86.8, places=1)
        self.assertEqual(result["verdict"], "ĐÚNG")
        self.assertIsNone(result["error"])
        self.assertEqual(result["warnings"], [])

    def test_strict_mode_khong_co_quality_thi_diem_bang_do_dung(self):
        result = assess(probabilities(), quality=None, scoring_mode="strict")
        self.assertIsNone(result["quality"])
        self.assertEqual(result["score"], result["correctness"])

    def test_penalty_dong_tac_dung_thang_diem_90_100(self):
        result = assess(probabilities(), quality=None)
        self.assertEqual(result["verdict"], "ĐÚNG")
        self.assertIsNone(result["error"])
        # Độ đúng 83.7 -> chuẩn hóa theo (83.7 - 50) / 50 -> 90 + 10 * 0.674 = 96.7
        self.assertAlmostEqual(result["score"], 96.7, places=1)
        self.assertEqual(result["grade"], "Tốt")

    def test_penalty_dong_tac_dung_ket_hop_quality(self):
        result = assess(probabilities(), quality=0.9, expected_exercise="CTK")
        self.assertEqual(result["verdict"], "ĐÚNG")
        # 0.5 * 96.7 + 0.5 * 90.0 = 93.4
        self.assertAlmostEqual(result["score"], 93.4, places=1)
        self.assertEqual(result["grade"], "Tốt")

    def test_penalty_loi_nhe_thang_diem_70_85(self):
        # Mô phỏng trường hợp trong thực tế của người dùng: SmallError chiếm ưu thế (84%)
        result = assess(
            probabilities(
                exercise_error_class=[0.3, 0.3, 0.2, 0.2],
                global_evaluation=[0.25, 0.65, 0.05, 0.05],
                error_type=[0.2, 0.3, 0.2, 0.2, 0.1],
                severity=[0.84, 0.16],
                body_part=[0.1, 0.74, 0.08, 0.08],
            ),
            quality=None,
        )
        self.assertEqual(result["verdict"], "SAI")
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["error"]["severity"], "SmallError")
        self.assertGreaterEqual(result["score"], 70.0)
        self.assertLessEqual(result["score"], 85.0)
        self.assertIn(result["grade"], ("Khá", "Đạt"))

    def test_penalty_loi_nang_thang_diem_30_50(self):
        # Lỗi nặng LargeError chiếm ưu thế (90%)
        result = assess(
            probabilities(
                exercise_error_class=[0.1, 0.6, 0.2, 0.1],
                global_evaluation=[0.1, 0.7, 0.1, 0.1],
                error_type=[0.1, 0.6, 0.1, 0.1, 0.1],
                severity=[0.1, 0.9],
                body_part=[0.1, 0.7, 0.1, 0.1],
            ),
            quality=None,
        )
        self.assertEqual(result["verdict"], "SAI")
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["error"]["severity"], "LargeError")
        self.assertGreaterEqual(result["score"], 30.0)
        self.assertLessEqual(result["score"], 50.0)
        self.assertIn(result["grade"], ("Chưa đạt", "Kém"))

    def test_quality_cao_khong_lam_lan_tap_sai_thanh_dung(self):
        result = assess(
            probabilities(
                exercise_error_class=[0.1, 0.1, 0.1, 0.7],
                global_evaluation=[0.1, 0.2, 0.6, 0.1],
                error_type=[0.1, 0.1, 0.1, 0.6, 0.1],
                severity=[0.3, 0.7],
                body_part=[0.1, 0.1, 0.1, 0.7],
            ),
            quality=0.95,
        )
        self.assertEqual(result["verdict"], "SAI")
        error = result["error"]
        self.assertEqual(error["error_type"], "Error3")
        self.assertEqual(error["reason"], "Incomplete")
        self.assertEqual(error["body_part"], "LeftArm")
        self.assertEqual(error["severity"], "LargeError")
        self.assertIn("tay trái", error["message"])

    def test_khong_cham_khi_du_doan_khac_bai_da_chon(self):
        result = assess(probabilities(exercise=[0.1, 0.8, 0.1]), expected_exercise="CTK")
        self.assertEqual(result["exercise"], "ELK")
        self.assertEqual(result["verdict"], "CHƯA CHẤM")
        self.assertIsNone(result["passed"])
        self.assertIsNone(result["score"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["unscored_reason"], "exercise_mismatch")
        self.assertTrue(any("CTK" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
