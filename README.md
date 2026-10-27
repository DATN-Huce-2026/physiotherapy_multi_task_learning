# Rehab MTL

This project is being built in verified stages. The current implementation covers
the KERAAL inventory and read-only Kinect XYZ loading. The model and training
components described in [the research plan](README_MTL_Rehabilitation_Plan.md)
are still pending.

## Current KERAAL files

- `data/keraal/keraal_manifest.csv`: one row per recording (841 rows) from group1A,
  group2A, and group3, with source paths relative to the KERAAL dataset root.
  Annotator labels are kept apart; group3 filename labels are stored as raw codes.
- `data/keraal/keraal_targets_preview.csv`: a review table of clip-level targets and
  masks. It preserves raw Group3 filename codes and adds a shared four-class
  `exercise_error_class` target, plus mapped global-evaluation and error-type
  targets for Group3.
- `data/keraal/keraal_subject_split_preview.csv`: a candidate participant-level split
  for review; all clips from each participant share one split. Group3 P1/P2/P3
  are assigned to train/validation/test respectively.
- `src/rehab_mtl/datasets/keraal.py`: strict Kinect frame parser and sample reader.
- `src/rehab_mtl/datasets/keraal_dataset.py`: PyTorch `KeraalMTLDataset`, batch collation (`collate_keraal_mtl`), and `build_keraal_dataloaders`.
- `scripts/keraal/validate_keraal.py`: checks the manifest, source-file joins, label
  values, temporal annotation states, and all Kinect frames.
- `scripts/keraal/build_keraal_targets_preview.py`: builds the target preview without
  changing the manifest or source dataset.
- `scripts/keraal/build_keraal_group3_csvs.py`: reads group3 Kinect and matching
  modality files, then updates the three CSVs. It preserves raw filename codes
  and writes mapped targets to the derived target preview.
- `scripts/keraal/preview_kinect_preprocessing.py`: shows a proposed coordinate offset
  for one clip in memory; it writes no transformed data.
- `scripts/keraal/validate_keraal_mtl_samples.py`: joins each Kinect sequence to its
  proposed targets by `sample_id` and checks masks across the dataset.
- `scripts/keraal/train_baseline.py`: end-to-end multi-task training runner using PyTorch DataLoaders.

The source dataset remains in its original location. The loader does not rewrite,
normalize, interpolate, or resample it. For v0 it reads the XYZ position values
from each joint and leaves out the quaternion values. Joint order follows the
KERAAL format README.

## Validate the local KERAAL data

From the project root, run:

```bash
python3 scripts/keraal/validate_keraal.py
```

The default dataset root is `~/Downloads/KERAAL`. Override it when the dataset
is stored elsewhere:

```bash
python3 scripts/keraal/validate_keraal.py --keraal-root /path/to/KERAAL
```

To rebuild the manifest, target preview, and subject split after adding group3:

```bash
python3 scripts/keraal/build_keraal_group3_csvs.py --keraal-root /path/to/KERAAL
```

Group3 has 540 recordings with Kinect, video, OpenPose, BlazePose, and Vicon
files, but no annotator XML files. Code `C` maps to `Correct`; the `E1`, `E2`,
and `E3` prefixes in codes such as `E1B1` map to `Error1`, `Error2`, and
`Error3`. The `B1`-`B3` suffix remains in the raw code and is not interpreted as
another target. Each derived target records its mapping in a resolution column.
Severity and body-part targets remain masked for Group3.

The validator is read-only. It fails with the sample ID and line number when a
Kinect frame does not contain exactly 175 finite numeric values, or when its
frame count differs from the manifest.

## Label handling

The loader preserves each annotator's original `Global Evaluation`, error type,
severity, and body-part labels. Blank values become `None` with a false label
mask. An explicit `none` error label stays labeled and has a true mask. A missing
temporal track remains unknown; temporal annotations are not converted into
frame targets because their timing has not been aligned to Kinect frames.
`load_keraal_sample` returns source XYZ by default. Its
`relative_to_initial_spine_base=True` option applies the reviewed-in-preview
translation in memory and records the origin used in the returned sample.

The pipe-separated label strings in the manifest are preserved as strings. A
training label mapper must split and interpret them according to a reviewed
ontology; it must not silently treat them as one class.

The current preview policy is provisional: if both annotators agree, their value
is used; if only one has a value, that value is shown; if both have different
values, that head is masked for that clip. A blank is not converted to `none`.
Exercise code comes from the manifest metadata. The preview also records review
flags for label combinations that need human interpretation. These rules must be
reviewed before the preview is used for training.

The shared `exercise_error_class` target uses `Correct`, `Error1`, `Error2`, and
`Error3`. Group3 codes map through their leading error prefix; Group1A/Group2A
use a resolved `Correct` global label or a resolved `Error1`-`Error3` label.
Incomplete, motionless, `Other`, unsupported, and disputed cases are masked for
this head. The four-class task follows the KERAAL experiment described in
[Marusic et al. (2025)](https://arxiv.org/abs/2504.13866). Raw labels and
per-task masks remain available for review.

Rebuild the preview with:

```bash
python3 scripts/keraal/build_keraal_targets_preview.py
```

Preview a coordinate offset for one clip with:

```bash
python3 scripts/keraal/preview_kinect_preprocessing.py --sample-id G1A-CTK-R1-Brest-038
```

Check Kinect-to-target joins for all samples with:

```bash
python3 scripts/keraal/validate_keraal_mtl_samples.py --relative-to-initial-spine-base
```

The validator reads one clip at a time. The target preview is still provisional;
this command checks data wiring and masks, not whether the annotation policy is
final.

## Test một lần tập bằng model đã train (Model v7)

`scripts/assess_exercise.py` mặc định dùng checkpoint `artifacts/Model v7/best_model.pt`
(train gộp KERAAL + KiMoRe, có head `quality`) để đánh giá một lần tập từ webcam
hoặc video: **ĐÚNG/SAI**, **sai ở đâu** (loại lỗi, bộ phận, mức độ) và **điểm 0-100**.

Trên Windows PowerShell, dùng Python 3.12 trong `.venv` của project:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[inference]"
.venv/Scripts/python.exe scripts/assess_exercise.py --source 0 --exercise RTK
.venv/Scripts/python.exe scripts/assess_exercise.py --source video.mp4 --exercise ELK --no-display
```

- Webcam mặc định không cần chạm bàn phím: giữ yên khoảng 1 giây để lấy tư thế
  gốc, thực hiện một lần tập, rồi trở về tư thế gốc và giữ khoảng 0.8 giây để
  hệ thống chấm. `q` thoát. Có thể dùng `--webcam-mode manual` để thử bằng SPACE.
  Video file: cả video là một lần tập. `--report ket_qua.json` để lưu kết quả.
- Bộ tự tách lần tập dùng ngưỡng chuyển động của thân và tay, chưa được hiệu chỉnh
  trên video webcam thật. Nếu không trở về tư thế gốc trước `--max-seconds`, lần
  tập được bỏ qua thay vì chấm một đoạn chưa hoàn chỉnh.
- MediaPipe Pose được chuyển sang khung xương Kinect v2 25 khớp
  (`src/rehab_mtl/inference/mediapipe_to_kinect.py`). Với model 12 góc, frame
  thiếu khớp được lấp ở mốc thời gian gốc, rồi dùng cùng hàm tiền xử lý như
  training: tính 12 góc từng frame và nội suy chuỗi góc về `[90, 12]`.
- ĐÚNG/SAI lấy từ ba head có lớp "đúng" (`global_evaluation`, `exercise_error_class`,
  `error_type`). Mặc định điểm chỉ dùng độ đúng của các head này; có thể bật
  thành phần `quality` bằng `--quality-weight`. Head `quality` chỉ có nhãn ở
  KiMoRe nên chưa phân biệt được đúng/sai trên bài KERAAL.
- Nếu bài model dự đoán khác `--exercise` đã chọn, kết quả là `CHƯA CHẤM`:
  chương trình vẫn báo bài dự đoán nhưng không xuất điểm hay mô tả lỗi của bài khác.
- Riêng RTK là phép xoay thân sang trái/phải; bộ 12 góc hiện tại không có góc
  xoay thân quanh trục dọc. Vì vậy kết quả RTK từ webcam chỉ phù hợp để thử
  luồng ghi hình/tiền xử lý, chưa đủ tin cậy để đánh giá người tập. Cần thêm
  đặc trưng xoay thân, train lại checkpoint và kiểm định trên dữ liệu webcam.
  Các video RTK đúng đã kiểm tra trong KERAAL thực hiện ở tư thế ngồi; khi thử
  luồng ghi hình hãy để camera thấy rõ cả tay, thân, hông, chân và bàn chân.
- Mô tả cụ thể của Error1/2/3 sửa trong `src/rehab_mtl/inference/error_descriptions.json`.
- Đây là phép chuyển đổi xấp xỉ từ MediaPipe sang Kinect, cần kiểm tra với các lần
  tập webcam thật trước khi tin vào ngưỡng đúng/sai. MediaPipe không có `SpineMid`:
  khớp này được nội suy thẳng, nên góc `spine_flexion` trên webcam gần như cố định.
- Trên Windows, đặt virtualenv ở đường dẫn không dấu: MediaPipe không mở được file model
  khi đường dẫn thư viện có ký tự như "Đ".
- Chạy test: `python -m unittest discover -s tests -t .` (sau `pip install -e .`).

## Next implementation steps

1. Review the target preview and finalize label and disagreement rules.
2. Make fixed subject-disjoint train/validation/test splits.
3. Add the PyTorch dataset, batching, shared backbone, and task heads, then
   compare multi-task training with single-task baselines.

See [README_MTL_Rehabilitation_Plan.md](README_MTL_Rehabilitation_Plan.md) for
the broader research plan and the roles of KERAAL, KIMORE, and other datasets.
