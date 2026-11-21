# Tài Liệu Đặc Tả Kỹ Thuật Model AI (Input / Output & Chế Độ Chạy)

Tài liệu này mô tả ngắn gọn, đúng trọng tâm về cơ chế vận hành, định dạng dữ liệu đầu vào (Input) và kết quả đầu ra (Output JSON) của model đánh giá bài tập vật lý trị liệu để tích hợp với hệ thống Backend/Frontend.

---

## 1. Kiểu chạy Model: Real-time hay Không Real-time?

Model hoạt động theo cơ chế **Chấm theo từng lần tập (Attempt / Clip-based)** và hỗ trợ **cả 2 chế độ**:

1. **Chế độ Tương tác trực tiếp qua Webcam (Near Real-time per Attempt)**:
   - Trong lúc người dùng tập trước camera, hệ thống trích xuất khung xương theo thời gian thực và **tự động phát hiện biên độ 1 lần tập (1 rep)** (hoặc bấm phím `SPACE` thủ công): *Giữ yên tư thế chuẩn bị (~1 giây) $\rightarrow$ Thực hiện động tác $\rightarrow$ Trở về tư thế ban đầu (~0.8 giây)*.
   - **Ngay sau khi kết thúc 1 lần tập**, model chấm điểm tức thì và trả kết quả lên màn hình.
2. **Chế độ Không Real-time (Offline Video Clip)**:
   - Người tập quay xong gửi file video clip lên để chấm. Toàn bộ video được tính là **1 lần tập** (thời lượng hợp lệ từ **3 giây đến 60 giây**).

> **Lưu ý kiến trúc**: Model **không** chấm điểm rời rạc trên từng frame đơn lẻ, mà cần gom đủ **chuỗi chuyển động của 1 lần tập** rồi mới đưa vào mạng Neural để đánh giá tổng thể.

---

## 2. Đầu vào (Input) & Định dạng

Hệ thống linh hoạt nhận **Video / Luồng Frame ảnh** hoặc **Chuỗi tọa độ khớp (Skeleton)**:

### A. Nếu truyền Video hoặc Frame ảnh
- **Video**: Nhận các định dạng chuẩn OpenCV đọc được (`.mp4`, `.webm`, `.avi`, `.mov`...) hoặc chỉ số webcam (`0`, `1`...).
- **Kích thước ảnh & FPS**:
  - **Không cố định kích thước** (hỗ trợ mọi độ phân giải chuẩn như `640x480`, `1280x720`, `1920x1080`... vì MediaPipe tự chuẩn hóa tọa độ 3D theo mét).
  - **FPS**: Mặc định xử lý ở **30 FPS** (hoặc theo FPS gốc của video/camera). Toàn bộ chuỗi sau đó sẽ được nội suy thời gian về độ dài cố định nên không bị phụ thuộc FPS đầu vào.
  - **Yêu cầu chất lượng**: Thời lượng lần tập $\ge 3.0$ giây, thấy rõ các khớp chính ở ít nhất **70% số frame**, không mất dấu khung xương liên tục quá **0.5 giây**.

### B. Nếu truyền Tọa độ khớp (Skeleton JSON / Array)
- **Không dùng COCO 17 điểm**.
- **Chuẩn sử dụng**:
  1. **Đầu vào từ Camera / Client**: **MediaPipe Pose (BlazePose) 33 điểm 3D** (`pose_world_landmarks`, tọa độ thực `x, y, z` tính bằng mét + độ rõ `visibility`). Cần thấy rõ 14 khớp chính (2 vai, 2 khuỷu tay, 2 cổ tay, 2 hông, 2 gối, 2 cổ chân, 2 mũi chân với `visibility >= 0.5`).
  2. **Bước chuyển đổi trung gian**: Hệ thống ánh xạ từ **33 điểm MediaPipe $\rightarrow$ 25 khớp chuẩn Kinect v2** (`[T, 25, 3]`).
  3. **Đầu vào thực tế của mạng Neural (Model v11)**: Từ chuỗi $T$ frames của 25 khớp Kinect, hệ thống trích xuất **12 góc khớp giải phẫu 3D** (khuỷu tay, vai, háng, gối, cổ chân trái/phải, độ gập cột sống, độ nghiêng thân mình) và nội suy tuyến tính co/giãn về **chuỗi cố định $N = 90$ frames** (Tensor kích thước `[90, 12]`). *(Hỗ trợ tương thích ngược với model cũ dùng tọa độ `[T, 75]`).*

---

## 3. Đầu ra (Output) & Định dạng JSON

### A. Tên / Mã bài tập (`exercise`)
Hiện tại model hỗ trợ danh sách **3 bài tập** phục hồi chức năng chuẩn tập dữ liệu KERAAL:

| Mã bài tập (`code`) | Tên bài tập hiển thị (`exercise_name`) | Mô tả động tác |
| :--- | :--- | :--- |
| **`CTK`** | `CTK - Che mặt (Cache-tête)` | Gập hai tay đưa lên che mặt / chạm đầu |
| **`ELK`** | `ELK - Kéo giãn bên (Étirement latéral)` | Nghiêng thân mình sang bên kết hợp nâng tay kéo giãn |
| **`RTK`** | `RTK - Xoay thân (Rotation du tronc)` | Giơ tay và xoay vặn thân trên sang hai bên |

*(Nếu người dùng chọn trước bài tập `expected_exercise` nhưng tập sang bài khác, model sẽ trả về trạng thái `"CHƯA CHẤM"` kèm cảnh báo).*

### B. Trạng thái Đúng/Sai, Điểm số & Số Rep
- **Trạng thái**:
  - `passed`: `true` (Đúng), `false` (Sai), hoặc `null` (Chưa chấm được do tập sai bài đã chọn).
  - `verdict`: `"ĐÚNG"`, `"SAI"`, hoặc `"CHƯA CHẤM"`.
- **Điểm số (`score`)**: Tính trên **thang 0 – 100** (số thực làm tròn 1 chữ số thập phân, ví dụ `85.4`), kèm xếp loại `grade`:
  - `Tốt` ($\ge 80$), `Khá` ($65 - 79.9$), `Đạt` ($50 - 64.9$), `Chưa đạt` ($35 - 49.9$), `Kém` ($< 35$).
  - Điểm được tổng hợp từ **Độ đúng phân loại** (`correctness`, thang 0–100) và **Chất lượng chuyển động KiMoRe** (`quality`, thang 0–100).
- **Số rep đúng / Tổng số rep**:
  - Mỗi lần gọi hàm `assess()` chấm cho **1 lần tập (1 rep)**.
  - Khi chạy theo phiên (nhiều lần tập liên tiếp), hệ thống lưu danh sách các lần tập vào mảng JSON và tổng kết:
    - **Tổng số rep**: Tổng số phần tử trong danh sách `attempts`.
    - **Số rep chấm hợp lệ**: Số lần tập không bị `skipped` và `passed != null`.
    - **Số rep đúng**: Số lần tập có `passed == true`.

### C. Chi tiết lỗi sai (`error`)
- **Trả về ở dạng**: Object JSON gồm **mã lỗi**, **text mô tả tiếng Việt**, **bộ phận cơ thể bị sai**, **mức độ nghiêm trọng** và **độ tin cậy (confidence)**:
  - **Nguyên nhân chung (`reason`)**: `"Incorrect"` (Sai kỹ thuật), `"Incomplete"` (Chưa hoàn thành hết biên độ), `"Motionless"` (Bất động).
  - **Mã lỗi (`error_type`)**: `"Error1"`, `"Error2"`, `"Error3"`, `"Other"`.
    - *Ví dụ bài `ELK`*: `Error1` (Tay đối diện không giữ dọc theo thân), `Error2` (Thân chưa nghiêng đủ), `Error3` (Tay phía trên chưa gập đúng).
    - *Ví dụ bài `RTK`*: `Error1` (Hai tay nâng chưa đủ cao), `Error2` (Mức xoay thân chưa đủ), `Error3` (Thân bị nghiêng sang bên).
  - **Vị trí khớp / Bộ phận sai (`body_part`)**: **CÓ** — Chỉ rõ vùng cơ thể sai gồm: `"Spine"` (cột sống / thân), `"BothArms"` (cả hai tay), `"RightArm"` (tay phải), `"LeftArm"` (tay trái).
  - **Mức độ lỗi (`severity`)**: `"SmallError"` (lỗi nhẹ) hoặc `"LargeError"` (lỗi nặng).
  - **Thời điểm / Giây bị sai**: **KHÔNG** — Model đánh giá trên toàn bộ chu kỳ của 1 lần tập (sequence-level), chỉ trả về tổng thời lượng lần tập (`duration_seconds`), không cắt timestamp giây cụ thể xảy ra lỗi trong rep đó.

---

## 4. Ví dụ JSON Output Chuẩn

Dưới đây là cấu trúc JSON thực tế trả về cho các lần tập (khi xuất qua `--report` hoặc trả về từ hàm đánh giá):

```json
[
  {
    "duration_seconds": 5.42,
    "visible_ratio": 0.965,
    "assessment": {
      "exercise": "ELK",
      "exercise_name": "ELK - Kéo giãn bên (Étirement latéral)",
      "exercise_confidence": 0.9412,
      "verdict": "SAI",
      "passed": false,
      "score": 46.8,
      "grade": "Chưa đạt",
      "correctness": 24.5,
      "quality": 61.2,
      "pass_score": 40.0,
      "min_error_confidence": 0.4,
      "scoring_mode": "penalty",
      "head_predictions": {
        "exercise": { "label": "ELK", "confidence": 0.9412 },
        "global_evaluation": { "label": "Incorrect", "confidence": 0.7821 },
        "exercise_error_class": { "label": "Error2", "confidence": 0.7105 },
        "error_type": { "label": "Error2", "confidence": 0.6895 },
        "severity": { "label": "LargeError", "confidence": 0.812 },
        "body_part": { "label": "Spine", "confidence": 0.884 }
      },
      "warnings": [],
      "error": {
        "reason": "Incorrect",
        "error_type": "Error2",
        "error_type_confidence": 0.7,
        "description": "Thân chưa nghiêng đủ",
        "body_part": "Spine",
        "body_part_confidence": 0.884,
        "severity": "LargeError",
        "severity_confidence": 0.812,
        "message": "Sai động tác: Thân chưa nghiêng đủ (70%) - bộ phận: cột sống / thân (88%) - mức độ: nặng (81%)"
      }
    }
  },
  {
    "duration_seconds": 4.85,
    "visible_ratio": 0.982,
    "assessment": {
      "exercise": "RTK",
      "exercise_name": "RTK - Xoay thân (Rotation du tronc)",
      "exercise_confidence": 0.975,
      "verdict": "ĐÚNG",
      "passed": true,
      "score": 92.4,
      "grade": "Tốt",
      "correctness": 91.0,
      "quality": 95.6,
      "pass_score": 40.0,
      "min_error_confidence": 0.4,
      "scoring_mode": "penalty",
      "head_predictions": {
        "exercise": { "label": "RTK", "confidence": 0.975 },
        "global_evaluation": { "label": "Correct", "confidence": 0.912 },
        "exercise_error_class": { "label": "Correct", "confidence": 0.895 },
        "error_type": { "label": "none", "confidence": 0.923 },
        "severity": { "label": "SmallError", "confidence": 0.65 },
        "body_part": { "label": "Spine", "confidence": 0.54 }
      },
      "warnings": [],
      "error": null
    }
  }
]
```
