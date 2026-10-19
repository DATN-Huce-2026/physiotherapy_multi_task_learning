#!/usr/bin/env python3
"""Test một lần tập vật lý trị liệu bằng model đã train (Model v6).

Webcam: SPACE bắt đầu/kết thúc một lần tập, 'q' thoát. Video: cả video là một lần tập.

    python scripts/assess_exercise.py --source 0 --exercise CTK
    python scripts/assess_exercise.py --source video.mp4 --exercise ELK --no-display
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rehab_mtl.inference.assessment import (  # noqa: E402
    DEFAULT_PASS_SCORE,
    DEFAULT_QUALITY_WEIGHT,
    assess,
    load_descriptions,
)
from rehab_mtl.inference.mediapipe_to_kinect import (  # noqa: E402
    DEFAULT_CAMERA_PITCH_DEG,
    KERAAL_FPS,
    KIMORE_FPS,
)
from rehab_mtl.inference.predictor import load_model, predict  # noqa: E402
from rehab_mtl.inference.recorder import SkeletonRecorder  # noqa: E402

if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WINDOW_NAME = "Physical Therapy Assessment"
COUNTDOWN_SECONDS = 3.0
MIN_CLIP_SECONDS = 3.0
MIN_VISIBLE_RATIO = 0.7


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Đánh giá một lần tập vật lý trị liệu từ webcam/video bằng model đã train."
    )
    parser.add_argument("--source", default="0", help="Chỉ số webcam hoặc file video")
    parser.add_argument(
        "--exercise",
        choices=("CTK", "ELK", "RTK"),
        help="Bài người tập được giao (cảnh báo nếu model nhận ra bài khác)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "model_v6" / "best_model.pt",
    )
    parser.add_argument("--pass-score", type=float, default=DEFAULT_PASS_SCORE)
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=DEFAULT_QUALITY_WEIGHT,
        help="Tỉ trọng head quality trong điểm cuối (0 = chỉ dùng head phân loại)",
    )
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=DEFAULT_CAMERA_PITCH_DEG,
        help="Góc bù (độ) giữa camera và Kinect của dữ liệu train",
    )
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--report", type=Path, help="Lưu kết quả các lần tập ra file JSON")
    parser.add_argument("--no-display", action="store_true")
    return parser


def evaluate_attempt(recorder, model, args, descriptions) -> dict[str, object]:
    total = len(recorder)
    visible_ratio = recorder.visible_frames / total if total else 0.0
    attempt: dict[str, object] = {
        "duration_seconds": round(recorder.duration, 2),
        "visible_ratio": round(visible_ratio, 3),
    }
    if recorder.duration < MIN_CLIP_SECONDS:
        attempt["skipped"] = f"Lần tập quá ngắn (< {MIN_CLIP_SECONDS:g} giây)"
        return attempt
    try:
        keraal_sequence = recorder.sequence(KERAAL_FPS)
        kimore_sequence = recorder.sequence(KIMORE_FPS)
    except ValueError as error:
        attempt["skipped"] = str(error)
        return attempt

    probabilities, quality = predict(model, keraal_sequence, kimore_sequence)
    result = assess(
        probabilities,
        quality,
        expected_exercise=args.exercise,
        pass_score=args.pass_score,
        quality_weight=args.quality_weight,
        descriptions=descriptions,
    )
    if visible_ratio < MIN_VISIBLE_RATIO:
        result["warnings"].append(
            f"Chỉ thấy đủ khớp ở {visible_ratio:.0%} số frame; hãy đứng lùi ra để "
            "camera thấy từ đầu đến hông."
        )
    attempt["assessment"] = result
    return attempt


def print_attempt(number: int, attempt: dict[str, object]) -> None:
    print("-" * 72)
    print(f"Lần {number}: {attempt['duration_seconds']} giây")
    if "skipped" in attempt:
        print(f"  Bỏ qua: {attempt['skipped']}")
        return
    result = attempt["assessment"]
    print(
        f"  Bài tập : {result['exercise_name']} "
        f"(độ tin cậy {result['exercise_confidence']:.0%})"
    )
    detail = f"độ đúng {result['correctness']}"
    if result["quality"] is not None:
        detail += f", chất lượng {result['quality']}"
    print(
        f"  Kết quả : {result['verdict']} - {result['score']}/100 "
        f"({result['grade']}) [{detail}]"
    )
    if result["error"]:
        print(f"  Sai ở   : {result['error']['message']}")
    for warning in result["warnings"]:
        print(f"  Cảnh báo: {warning}")


def overlay_lines(state, recorder, last_attempt, now, countdown_end, is_webcam):
    # OpenCV không vẽ được chữ có dấu nên dùng tiếng Anh
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if state == "recording":
        lines.append((f"RECORDING {recorder.duration:4.1f}s", (0, 0, 255)))
    elif state == "countdown":
        lines.append((f"Get ready... {max(0.0, countdown_end - now):.1f}", (0, 255, 255)))
    if last_attempt and state != "recording":
        if "skipped" in last_attempt:
            lines.append(("Last attempt skipped", (0, 180, 255)))
        else:
            result = last_attempt["assessment"]
            color = (0, 220, 0) if result["passed"] else (0, 0, 255)
            label = "CORRECT" if result["passed"] else "INCORRECT"
            lines.append(
                (f"{label} | Score {result['score']:.0f}/100 | {result['exercise']}", color)
            )
            error = result["error"]
            if error:
                lines.append(
                    (
                        f"Error: {error['error_type']} | {error['body_part']} | "
                        f"{error['severity']}",
                        (0, 180, 255),
                    )
                )
    if is_webcam:
        hint = "SPACE: stop" if state == "recording" else "SPACE: start"
        lines.append((f"{hint} | q: quit", (200, 200, 200)))
    return lines


def draw_panel(frame, lines) -> None:
    if not lines:
        return
    width = frame.shape[1]
    scale = max(0.45, min(0.7, width / 1280 * 0.7))
    thickness = 1 if width < 900 else 2
    line_height = int(32 * scale / 0.7)
    box_width = min(
        width - 16,
        max(
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
            for text, _ in lines
        )
        + 20,
    )
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (8, 8), (8 + box_width, 24 + line_height * len(lines)), (0, 0, 0), cv2.FILLED
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, 8 + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def run(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    descriptions = load_descriptions()
    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        print(f"Không mở được nguồn video: {args.source}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp.solutions.pose.Pose(model_complexity=1)
    recorder = SkeletonRecorder(args.min_visibility, args.camera_pitch)

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    state = "idle" if is_webcam else "recording"
    countdown_end = 0.0
    attempts: list[dict[str, object]] = []
    last_attempt: dict[str, object] | None = None
    frame_index = 0

    def finish_attempt() -> None:
        nonlocal last_attempt
        last_attempt = evaluate_attempt(recorder, model, args, descriptions)
        attempts.append(last_attempt)
        print_attempt(len(attempts), last_attempt)
        recorder.reset()

    if is_webcam:
        print("Nhấn SPACE để bắt đầu một lần tập, SPACE lần nữa để kết thúc, 'q' để thoát.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic() if is_webcam else frame_index / fps
            frame_index += 1
            # Dùng ảnh chưa lật để trái/phải đúng
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if state == "countdown" and now >= countdown_end:
                state = "recording"
                recorder.reset()
            if state == "recording":
                recorder.add(now, results)
                if recorder.duration >= args.max_seconds:
                    finish_attempt()
                    if not is_webcam:
                        state = "done"
                        break
                    state = "idle"

            if args.no_display:
                continue
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )
            if is_webcam:
                frame = cv2.flip(frame, 1)  # lật như gương khi hiển thị
            draw_panel(
                frame,
                overlay_lines(state, recorder, last_attempt, now, countdown_end, is_webcam),
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1 if is_webcam else max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and is_webcam:
                if state == "recording":
                    finish_attempt()
                    state = "idle"
                elif state == "idle":
                    state = "countdown"
                    countdown_end = now + COUNTDOWN_SECONDS
    except KeyboardInterrupt:
        print("Đã dừng theo yêu cầu.")
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()

    if state == "recording" and len(recorder):
        finish_attempt()

    assessed = [a["assessment"] for a in attempts if "assessment" in a]
    print("=" * 72)
    print(
        f"TỔNG KẾT: {len(attempts)} lần tập, chấm được {len(assessed)}, "
        f"ĐÚNG {sum(bool(r['passed']) for r in assessed)}"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(attempts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu kết quả: {args.report}")
    return 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
