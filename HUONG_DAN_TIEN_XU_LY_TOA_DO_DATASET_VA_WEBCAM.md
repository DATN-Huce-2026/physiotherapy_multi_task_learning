# Hướng Dẫn Kỹ Thuật: Tiền Xử Lý Tọa Độ Khung Xương Từ Dataset & Đồng Bộ Hóa Webcam Cho Mô Hình MTL Phục Hồi Chức Năng

Tài liệu này trình bày toàn bộ kiến trúc, lý thuyết toán học và mã nguồn thực thi về quy trình tiền xử lý tọa độ khung xương (skeleton preprocessing) từ các tập dữ liệu huấn luyện ([KERAAL](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/datasets/keraal.py), [KiMoRe](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/datasets/kimore.py)) và cách chuẩn hóa dữ liệu từ **Webcam thời gian thực (qua MediaPipe Pose)** nhằm phục vụ triển khai mô hình học đa nhiệm (Multi-Task Learning) trong hệ thống phục hồi chức năng.

---

## Mục Lục
1. [Tổng Quan Kiến Trúc & Vấn Đề Lệch Miền (Domain Gap)](#1-tổng-quan-kiến-trúc--vấn-đề-lệch-miền-domain-gap)
2. [Chi Tiết Dữ Liệu Đầu Vào Giữa Dataset Và Webcam](#2-chi-tiết-dữ-liệu-đầu-vào-giữa-dataset-và-webcam)
3. [Hai Phương Pháp Tiền Xử Lý Trong Repository](#3-hai-phương-pháp-tiền-xử-lý-trong-repository)
   - [3.1. Phương Pháp Cũ: Tọa Độ Gốc 3D (75 chiều) & Hạn Chế Khi Dùng Webcam](#31-phương-pháp-cũ-tọa-độ-gốc-3d-75-chiều--hạn-chế-khi-dùng-webcam)
   - [3.2. Phương Pháp Mới: 12 Góc Khớp Sinh Cơ Học & Khắc Phục Lệch Miền Webcam](#32-phương-pháp-mới-12-góc-khớp-sinh-cơ-học--khắc-phục-lệch-miền-webcam)
4. [Công Thức Toán Học & Danh Sách 12 Góc Lâm Sàng](#4-công-thức-toán-học--danh-sách-12-góc-lâm-sàng)
5. [Quy Trình Chuyển Đổi MediaPipe (33 Mốc) Sang Kinect v2 (25 Khớp)](#5-quy-trình-chuyển-đổi-mediapipe-33-mốc-sang-kinect-v2-25-khớp)
   - [5.1. Đảo Trục Hệ Tọa Độ (Axis Flipping)](#51-đảo-trục-hệ-tọa-độ-axis-flipping)
   - [5.2. Bù Góc Nghiêng Camera (Pitch Rotation Compensation)](#52-bù-góc-nghiêng-camera-pitch-rotation-compensation)
   - [5.3. Nội Suy Các Khớp Ảo Cột Sống (Synthetic Spine Joints)](#53-nội-suy-các-khớp-ảo-cột-sống-synthetic-spine-joints)
6. [Đồng Bộ Hóa Thời Gian & Tần Số Lấy Mẫu (Temporal Resampling)](#6-đồng-bộ-hóa-thời-gian--tần-số-lấy-mẫu-temporal-resampling)
7. [Luồng Tích Hợp Dự Đoán Tại `predictor.py` & `assess_exercise.py`](#7-luồng-tích-hợp-dự-đoán-tại-predictorpy--assess_exercisepy)
8. [Hướng Dẫn Triển Khai Thực Tế & Code Mẫu Thời Gian Thực](#8-hướng-dẫn-triển-khai-thực-tế--code-mẫu-thời-gian-thực)
9. [Checklist Môi Trường & Thiết Lập Phòng Tập Cho Người Bệnh](#9-checklist-môi-trường--thiết-lập-phòng-tập-cho-người-bệnh)

---

## 1. Tổng Quan Kiến Trúc & Vấn Đề Lệch Miền (Domain Gap)

Hệ thống phục hồi chức năng MTL giải quyết đồng thời nhiều nhiệm vụ lâm sàng:
- **Phân loại bài tập** (`exercise`: CTK, ELK, RTK).
- **Đánh giá tổng thể** (`global_evaluation`: Correct, Incorrect, Incomplete, Motionless).
- **Phát hiện lỗi sai cụ thể** (`error_type`, `exercise_error_class`, `severity`, `body_part`).
- **Chấm điểm chất lượng vận động** (`quality`: Thang điểm liên tục $0.0 - 1.0$ từ tập KiMoRe).

```mermaid
flowchart TD
    subgraph Huấn Luyện (Training)
        D1[KERAAL Dataset<br>Kinect v2: 10 FPS, Mét] --> P_TRAIN[Tiền Xử Lý Dữ Liệu]
        D2[KiMoRe Dataset<br>Kinect v2: 30 FPS, Mét] --> P_TRAIN
        P_TRAIN --> F_EXTRACT[Trích Xuất 12 Góc Khớp<br>rehab_mtl.features.angles]
        F_EXTRACT --> RESAMPLE_90[Nội Suy Co Giãn Scipy<br>Cố định T=90 frames]
        RESAMPLE_90 --> MODEL[Mô hình Shared Backbone<br>KeraalSharedBackbone: input_dim=12]
    end

    subgraph Thực Tế (Webcam Inference)
        W[Webcam RGB 2D Camera] --> MP[Google MediaPipe Pose<br>33 World Landmarks]
        MP --> FILTER[Kiểm Tra Độ Tin Cậy<br>Visibility >= 0.5]
        FILTER --> ALIGN[Căn Chỉnh Hệ Tọa Độ & Khớp Ảo<br>mediapipe_world_to_kinect]
        ALIGN --> LIVE_FEAT[Trích Xuất 12 Góc Khớp Lâm Sàng]
        LIVE_FEAT --> RESAMPLE_LIVE[Resample T=90 frames]
        RESAMPLE_LIVE --> MODEL
        MODEL --> OUT[Kết Quả: Bài tập, Lỗi sai, Điểm số 0-100]
    end
```

### Thách thức cốt lõi (Domain Gap):
1. **Khác biệt cảm biến**: Kinect v2 dùng cảm biến hồng ngoại ToF (Time-of-Flight) đo khoảng cách vật lý $Z$ cực kỳ chính xác. Webcam RGB thông thường chỉ là camera 2D, các mô hình Deep Learning (như MediaPipe) chỉ ước tính tọa độ $Z$ mang tính tương đối.
2. **Khác biệt tỉ lệ (Scale)**: Người dùng đứng xa/gần camera, người cao/thấp sẽ có biên độ tọa độ $X, Y, Z$ chênh lệch nhau rất lớn.
3. **Khác biệt góc nhìn (Viewpoint & Pitch)**: Kinect phòng thí nghiệm thường đặt ở độ cao $1.0\text{m} - 1.2\text{m}$, vuông góc với sàn. Webcam người dùng cá nhân thường là camera laptop để trên bàn hướng lên (nghiêng khoảng $8^\circ - 15^\circ$) hoặc camera điện thoại đặt lệch góc.
4. **Khác biệt tần số lấy mẫu (FPS)**: KERAAL thu ở $\approx 10$ FPS, KiMoRe ở $\approx 30$ FPS, Webcam biến thiên từ $15 - 60$ FPS tùy ánh sáng và phần cứng.

---

## 2. Chi Tiết Dữ Liệu Đầu Vào Giữa Dataset Và Webcam

| Đặc điểm | Kinect v2 (KERAAL & KiMoRe) | Webcam Thực Tế (MediaPipe Pose) |
|:---|:---|:---|
| **Định dạng gốc** | File `.txt` (KERAAL: 175 giá trị/frame) hoặc `.csv` (KiMoRe: 100 giá trị/frame) | Luồng khung hình video RGB thời gian thực |
| **Số lượng khớp** | 25 khớp xương chuẩn Kinect v2 | 33 mốc cơ thể (landmarks) |
| **Hệ tọa độ $X, Y, Z$** | $X$ sang phải, $Y$ hướng lên, $Z$ hướng tới người (mét) | $X$ sang trái, $Y$ hướng xuống, $Z$ hướng xa camera |
| **Đơn vị đo lường** | Mét vật lý chính xác theo Depth ToF | Tọa độ chuẩn hóa $[0, 1]$ hoặc mét giả lập tại hông |
| **Tần số lấy mẫu** | Cố định (KERAAL: 10 FPS, KiMoRe: 30 FPS) | Biến động thời gian thực (tùy thuộc tải CPU/GPU) |
| **Nhiễu dữ liệu** | Thi thoảng mất khớp khi quay lưng | Nhiễu rung giật (jitter) do điều kiện ánh sáng kém |

---

## 3. Hai Phương Pháp Tiền Xử Lý Trong Repository

### 3.1. Phương Pháp Cũ: Tọa Độ Gốc 3D (75 chiều) & Hạn Chế Khi Dùng Webcam
* **Được định nghĩa tại**: [`src/rehab_mtl/datasets/keraal.py:offset_to_initial_spine_base`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/datasets/keraal.py#L141-L150).
* **Cơ chế**: Lấy tọa độ 25 khớp $\times$ 3 chiều $(X, Y, Z) = 75$ giá trị. Trừ tọa độ của khớp gốc hông `SpineBase` ở frame đầu tiên $t=0$:
  $$P_{t, j}' = P_{t, j} - P_{0, \text{SpineBase}}$$
* **Hạn chế lớn khi triển khai Webcam**:
  - Dù đã triệt tiêu vị trí đứng ban đầu, nhưng kích thước người (chiều dài xương tay, chân, thân) vẫn thể hiện trực tiếp qua khoảng cách mét giữa các khớp. Người cao $1\text{m}80$ sẽ có biên độ tọa độ lớn hơn người cao $1\text{m}55$, khiến mô hình phân loại nhầm thành "biên độ quá đà" hoặc "lỗi cử động".
  - Ước lượng trục $Z$ của MediaPipe trên ảnh 2D có sai số lớn. Khi mô hình phụ thuộc vào tọa độ $Z$ tuyệt đối, sai số này sẽ làm méo toàn bộ hình học động tác.

### 3.2. Phương Pháp Mới: 12 Góc Khớp Sinh Cơ Học & Khắc Phục Lệch Miền Webcam
* **Được định nghĩa tại**: [`src/rehab_mtl/features/angles.py`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/features/angles.py) (tích hợp trong commit `f36b737`).
* **Cơ chế**: Biến đổi 25 khớp tọa độ không gian thành **12 góc giải phẫu chính của cơ thể người**.
* **Ưu điểm vượt trội khi sử dụng với Webcam**:
  1. **Bất biến tịnh tiến (Translation Invariance)**: Đứng ở bất kỳ vị trí nào trong khung hình webcam, góc gập cẳng tay hay góc gối vẫn giữ nguyên.
  2. **Bất biến tỉ lệ (Scale Invariance)**: Dù là người lớn hay trẻ nhỏ, khoảng cách tay dài hay ngắn, góc mở khớp luôn nằm trong miền chuẩn hóa $[0, \pi]$ radian. Mọi tỉ lệ khuếch đại $s$ của camera đều tự động bị triệt tiêu:
     $$\frac{(s \cdot \vec{u}) \cdot (s \cdot \vec{v})}{\|s \cdot \vec{u}\| \cdot \|s \cdot \vec{v}\|} = \frac{s^2 (\vec{u} \cdot \vec{v})}{s^2 (\|\vec{u}\| \|\vec{v}\|)} = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
  3. **Miễn nhiễm với sự khác biệt đơn vị**: Không còn quan tâm Kinect đo bằng mét hay MediaPipe đo bằng pixel/chuẩn hóa.
  4. **Khả năng giải thích lâm sàng (Clinical Explainability)**: Các bài tập vật lý trị liệu đánh giá dựa trên biên độ cử động (Range of Motion - ROM), ví dụ góc khuỷu tay phải đạt $90^\circ - 120^\circ$, lưng phải thẳng (Spine flexion $\approx 180^\circ$).

---

## 4. Công Thức Toán Học & Danh Sách 12 Góc Lâm Sàng

### 4.1. Công thức tính góc 3D
Hàm [`calculate_angle_3d(a, b, c)`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/features/angles.py#L55-L75) tính góc tại đỉnh $B$ tạo bởi 3 khớp $A$, $B$, $C$:

$$\vec{ba} = A - B, \quad \vec{bc} = C - B$$
$$\text{cos\_theta} = \frac{\vec{ba} \cdot \vec{bc}}{\|\vec{ba}\|_2 \cdot \|\vec{bc}\|_2}$$
$$\theta = \arccos\big(\max(-1.0, \min(1.0, \text{cos\_theta}))\big) \quad (\text{đơn vị: radian})$$

*Xử lý suy biến*: Nếu một trong hai đoạn xương có độ dài $<\!10^{-6}$ (hai khớp trùng nhau), hàm trả về an toàn $0.0$, tránh lỗi chia cho $0$ hoặc giá trị `NaN`.

### 4.2. Danh sách 12 góc giải phẫu
Được trích xuất từ 25 khớp Kinect thông qua hàm [`extract_angles_from_kinect_frame(frame)`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/features/angles.py#L77-L114):

```
       [3: Head]
           |
       [2: Neck]
           |
   4---[20: SpineShoulder]---8
  /        |                  \
 5         |                   9
 |     [1: SpineMid]           |
 6         |                  10
 7         |                  11
       [0: SpineBase]
      /              \
    12                16
    |                  |
    13                17
    |                  |
    14                18
    15                19
```

1. **`left_elbow`**: Góc khuỷu tay trái: $\angle(\text{ShoulderLeft}(4) - \text{ElbowLeft}(5) - \text{WristLeft}(6))$.
2. **`right_elbow`**: Góc khuỷu tay phải: $\angle(\text{ShoulderRight}(8) - \text{ElbowRight}(9) - \text{WristRight}(10))$.
3. **`left_shoulder`**: Góc nách/khớp vai trái: $\angle(\text{SpineShoulder}(20) - \text{ShoulderLeft}(4) - \text{ElbowLeft}(5))$.
4. **`right_shoulder`**: Góc nách/khớp vai phải: $\angle(\text{SpineShoulder}(20) - \text{ShoulderRight}(8) - \text{ElbowRight}(9))$.
5. **`left_hip`**: Góc khớp háng trái: $\angle(\text{SpineBase}(0) - \text{HipLeft}(12) - \text{KneeLeft}(13))$.
6. **`right_hip`**: Góc khớp háng phải: $\angle(\text{SpineBase}(0) - \text{HipRight}(16) - \text{KneeRight}(17))$.
7. **`left_knee`**: Góc khớp gối trái: $\angle(\text{HipLeft}(12) - \text{KneeLeft}(13) - \text{AnkleLeft}(14))$.
8. **`right_knee`**: Góc khớp gối phải: $\angle(\text{HipRight}(16) - \text{KneeRight}(17) - \text{AnkleRight}(18))$.
9. **`left_ankle`**: Góc cổ chân trái: $\angle(\text{KneeLeft}(13) - \text{AnkleLeft}(14) - \text{FootLeft}(15))$.
10. **`right_ankle`**: Góc cổ chân phải: $\angle(\text{KneeRight}(17) - \text{AnkleRight}(18) - \text{FootRight}(19))$.
11. **`spine_flexion`**: Góc gập/duỗi cột sống: $\angle(\text{SpineBase}(0) - \text{SpineMid}(1) - \text{SpineShoulder}(20))$.
12. **`trunk_tilt`**: Góc nghiêng thân mình: $\angle(\text{SpineShoulder}(20) - \text{SpineBase}(0) - \text{HipLeft}(12))$.

---

## 5. Quy Trình Chuyển Đổi MediaPipe Sang Kinect v2 (25 Khớp)

Mô-đun [`src/rehab_mtl/inference/mediapipe_to_kinect.py`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/inference/mediapipe_to_kinect.py) thực hiện nhiệm vụ cầu nối giữa MediaPipe Pose và cấu trúc 25 khớp Kinect v2.

### 5.1. Đảo Trục Hệ Tọa Độ (Axis Flipping)
Quy ước hệ trục của 2 hệ thống ngược chiều nhau:
- MediaPipe: Trục $X$ hướng sang trái màn hình, trục $Y$ hướng xuống đất.
- Kinect v2: Trục $X$ hướng sang phải, trục $Y$ hướng lên trần nhà.
- **Biến đổi**:
  $$\text{Landmarks}_{\text{kinect\_coord}} = \text{Landmarks}_{\text{mediapipe}} \odot \begin{bmatrix} -1.0 \\ -1.0 \\ 1.0 \end{bmatrix}$$

### 5.2. Bù Góc Nghiêng Camera (Pitch Rotation Compensation)
Camera laptop/webcam thường để thấp và ngửa lên một góc (mặc định $8^\circ$). Ma trận quay quanh trục $X$ bù trừ độ dốc này:
$$R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{bmatrix}, \quad P_{\text{corrected}} = P \cdot R_x(\alpha)^T$$

### 5.3. Nội Suy Các Khớp Ảo Cột Sống (Synthetic Spine Joints)
MediaPipe không phát hiện trực tiếp cột sống. Các khớp này được tính toán dựa trên tỉ lệ nhân trắc học đo đạc từ cơ thể người:
1. **`SpineBase` (Khớp 0)**:
   $$\text{SpineBase} = \frac{\text{HipLeft}(23) + \text{HipRight}(24)}{2} + [0.0, 0.0, 0.03]$$
2. **`SpineShoulder` (Khớp 20)**:
   $$\text{SpineShoulder} = \frac{\text{ShoulderLeft}(11) + \text{ShoulderRight}(12)}{2} + [0.0, 0.015, 0.01]$$
3. **`SpineMid` (Khớp 1)**: Nằm ở vị trí $57\%$ chiều dài thân từ hông lên vai:
   $$\text{SpineMid} = \text{SpineBase} + 0.57 \times (\text{SpineShoulder} - \text{SpineBase})$$
4. **`Neck` (Khớp 2)**: Nằm ở vị trí $31\%$ từ gốc vai lên đầu:
   $$\text{Head} = \frac{\text{LeftEar}(7) + \text{RightEar}(8)}{2}$$
   $$\text{Neck} = \text{SpineShoulder} + 0.31 \times (\text{Head} - \text{SpineShoulder})$$

---

## 6. Đồng Bộ Hóa Thời Gian & Tần Số Lấy Mẫu (Temporal Resampling)

Để mạng nơ-ron GRU/SharedBackbone xử lý nhất quán một chuỗi vận động, hệ thống áp dụng kỹ thuật co/giãn thời gian về **độ dài cố định $T=90$ frames** (~3 giây ở tốc độ 30 FPS).

### Thuật toán nội suy tuyến tính:
Hàm [`resample_sequence_to_fixed_length(sequence, target_length=90)`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/features/angles.py#L116-L152):
1. Nhận chuỗi góc có độ dài gốc $T_{\text{orig}}$ (ví dụ 45 frames hoặc 150 frames).
2. Xây dựng trục thời gian chuẩn hóa:
   $$x_{\text{orig}} = \text{linspace}(0.0, 1.0, T_{\text{orig}})$$
   $$x_{\text{target}} = \text{linspace}(0.0, 1.0, 90)$$
3. Sử dụng `scipy.interpolate.interp1d(x_orig, seq, axis=0, kind="linear")` để lấy mẫu chính xác tại 90 mốc thời gian đều nhau.
4. Trả về mảng numpy hình dạng cố định `[90, 12]` kiểu `float32`.

---

## 7. Luồng Tích Hợp Dự Đoán Tại `predictor.py` & `assess_exercise.py`

Khi gọi hàm dự đoán thời gian thực tại [`src/rehab_mtl/inference/predictor.py`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/inference/predictor.py):

```python
# Tự động phát hiện kích thước input từ checkpoint (12 góc hoặc 75 tọa độ)
model = load_model("artifacts/kimore_baseline/best_model.pt")

# Nếu model.input_dim == 12:
# Tự động nhận chuỗi 25 khớp [T, 25, 3] từ recorder,
# tính 12 góc và nội suy về đúng [1, 90, 12] trước khi feed-forward qua mạng!
outputs = _forward(model, keraal_sequence)
```

Quá trình đánh giá hoàn chỉnh (Pass/Fail, loại lỗi, điểm số 0-100) được đóng gói trong hàm `assess()` tại [`src/rehab_mtl/inference/assessment.py`](file:///d:/DATN/physiotherapy_multi_task_learning/src/rehab_mtl/inference/assessment.py):
- **Phân loại bài tập**: Chọn lớp có xác suất cao nhất trong head `exercise`.
- **Phát hiện lỗi sai**: Đọc head `global_evaluation`, `error_type`, `severity`, `body_part`.
- **Tính điểm tổng hợp**:
  $$\text{Final Score} = (1 - w_{\text{quality}}) \times \text{Score}_{\text{classification}} + w_{\text{quality}} \times \text{Score}_{\text{quality}}$$

---

## 8. Hướng Dẫn Triển Khai Thực Tế & Code Mẫu Thời Gian Thực

### Cách 1: Chạy trực tiếp công cụ có sẵn trong repository
Chạy lệnh đánh giá bài tập qua webcam với giao diện tương tác (phím `SPACE` bắt đầu/kết thúc 1 lần tập, phím `q` thoát):

```bash
# Cài đặt môi trường nếu chưa có
pip install -e ".[inference]"

# Chạy đánh giá bài tập CTK qua webcam
python scripts/assess_exercise.py --source 0 --exercise CTK
```

### Cách 2: Triển khai Cửa Sổ Trượt (Sliding Window) chạy liên tục không cần nhấn phím
Dưới đây là mã nguồn Python độc lập cho phép nhận diện liên tục theo thời gian thực:

```python
"""webcam_streaming_demo.py: Dự đoán chuyển động thời gian thực bằng Sliding Window."""

import collections
import cv2
import mediapipe as mp
import numpy as np
import torch

from rehab_mtl.features.angles import (
    extract_angles_from_kinect_frame,
    resample_sequence_to_fixed_length,
)
from rehab_mtl.inference.mediapipe_to_kinect import (
    REQUIRED_LANDMARKS,
    mediapipe_world_to_kinect,
)
from rehab_mtl.inference.predictor import load_model


def main():
    model_path = "artifacts/combined_mtl_best/best_multitask_model.pt"
    print(f"Đang tải model: {model_path}...")
    model = load_model(model_path)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    cap = cv2.VideoCapture(0)
    buffer = collections.deque(maxlen=90)  # Lưu trữ tối đa 90 frames góc khớp
    stride_counter = 0

    status_text = "Đang tích lũy chuỗi cử động..."
    quality_score = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_world_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_world_landmarks.landmark
            visibility = np.array([lm.visibility for lm in landmarks])

            # Chỉ xử lý khi các khớp chính không bị che khuất
            if (visibility[list(REQUIRED_LANDMARKS)] >= 0.5).all():
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                # 1. Chuyển đổi sang 25 khớp Kinect v2
                kinect_joints = mediapipe_world_to_kinect(pts)
                # 2. Trích xuất 12 góc sinh cơ học [12]
                angles = extract_angles_from_kinect_frame(kinect_joints)
                buffer.append(angles)

                # 3. Kích hoạt dự đoán mỗi 6 frames khi đã có ít nhất 45 frames
                stride_counter += 1
                if len(buffer) >= 45 and stride_counter % 6 == 0:
                    raw_seq = np.array(buffer, dtype=np.float32)
                    # 4. Nội suy cố định 90 frames [90, 12]
                    fixed_seq = resample_sequence_to_fixed_length(raw_seq, target_length=90)
                    tensor_in = torch.from_numpy(fixed_seq).unsqueeze(0)
                    lengths = torch.tensor([90], dtype=torch.long)

                    with torch.no_grad():
                        outputs = model(tensor_in, lengths)
                        if "quality" in outputs:
                            quality_score = float(outputs["quality"][0].item()) * 100.0
                        if "exercise" in outputs:
                            pred_idx = int(torch.argmax(outputs["exercise"][0]))
                            status_text = f"Bai tap: {pred_idx} | Chat luong: {quality_score:.1f}/100"

        # Hiển thị HUD thông tin
        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Rehab MTL Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 9. Checklist Môi Trường & Thiết Lập Phòng Tập Cho Người Bệnh

> [!IMPORTANT]
> **Các khuyến nghị vàng để mô hình hoạt động ổn định nhất**:
> 1. **Vị trí đứng của người tập**: Cách webcam **1.8 mét đến 2.5 mét**. Camera phải nhìn thấy trọn vẹn từ đỉnh đầu tới bàn chân (Full body in frame).
> 2. **Ánh sáng**: Đủ sáng, chiếu sáng từ phía trước mặt. Tuyệt đối tránh đứng ngược sáng (đứng trước cửa sổ hoặc bóng đèn mạnh).
> 3. **Trang phục**: Quần áo gọn gàng, tránh mặc váy dài hoặc áo khoác quá rộng làm MediaPipe nhận diện sai khớp gối và khuỷu tay.
> 4. **Góc đặt webcam**: Đặt camera ở độ cao khoảng $0.9\text{m} - 1.2\text{m}$ (ngang tầm ngực/bụng). Nếu để laptop trên bàn thấp, tham số `--camera-pitch 8.0` sẽ tự động bù góc nghiêng.
> 5. **Môi trường Python trên Windows**: Luôn tạo virtualenv ở thư mục không chứa ký tự tiếng Việt có dấu (ví dụ: `C:\envs\rehab_mtl`) để tránh lỗi nội bộ khi MediaPipe nạp model nhị phân.
