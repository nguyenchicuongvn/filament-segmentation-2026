# Nhật ký tiến độ

Mỗi phiên làm việc thêm một mục ở **đầu** phần "Nhật ký" (mới nhất trên cùng) và cập nhật bảng
"Trạng thái". Chi tiết từng thí nghiệm nằm ở [EXPERIMENTS.md](EXPERIMENTS.md), kế hoạch ở
[PLAN.md](PLAN.md).

## Trạng thái (cập nhật 2026-10-24)

| | |
|---|---|
| Submission tốt nhất | **Public LB 0.31**: U-Net ResNet34 @768, 8x TTA (val PQ 0.383) |
| Hạng | 767 / 1,153 đội (snapshot 2026-10-24) |
| LB tham chiếu | #1 0.69 · 97 đội ≥ 0.50 · trung vị 0.34 · 494 đội > 0.35 |
| Submissions đã dùng | 3 (tối đa 5/ngày) |
| Deadline | Kaggle: **2026-11-15 06:00 UTC** (13:00 VN). Report + code (Google form): Nov 15 11:59 pm CDT |
| Repo | https://github.com/nguyenchicuongvn/filament-segmentation-2026 (**private** đến khi cuộc thi đóng) |
| GPU | Kaggle GPU quota tuần này đã hết → train GPU trên Colab (`colab/train_unet.ipynb`); Mac M5 (MPS) cho run nhỏ |
| Đang chờ | Run `unet1024_r34` trên Colab (người dùng chạy notebook) |

## Lịch sử submission

| Ngày (UTC) | File | Mô tả | val PQ | Public LB |
|---|---|---|---|---|
| 2026-10-23 03:47 | `submission_unet768.csv` | U-Net R34 @768, TTA, post t0.4 a50 g4 | 0.383 | **0.31** |
| 2026-10-22 20:58 | `submission_unet512.csv` | U-Net R34 @512, TTA, post t0.4 a25 g4 | 0.354 | 0.29 |
| 2026-10-22 16:48 | `submission.csv` | Classical baseline v1 | 0.242* | 0.21 |

\* Đo lại trên split chia theo ảnh (sau khi sửa lỗi rò rỉ, xem 2026-10-22).

Val luôn cao hơn LB khoảng 0.03–0.07, nhưng thứ tự giữa các model giữ nguyên, nên val vẫn dùng
được để chọn model (xem [DATA_AND_METRIC.md](DATA_AND_METRIC.md#khoảng-cách-val--lb)).

## Nhật ký

### 2026-10-24
- Viết bộ docs theo dõi: `docs/PROGRESS.md`, `EXPERIMENTS.md`, `DATA_AND_METRIC.md`, `PLAN.md`,
  `docs/logs/`, `CLAUDE.md`.

### 2026-10-23
- U-Net @768 trên Mac (MPS, ~2.5 h): val PQ 0.362 thô → **0.383** sau tune post-processing.
  Submit → **LB 0.31**.
- Ensemble trung bình map 512 + 768: val 0.380, thấp hơn 768 dùng một mình → không nộp.
- Kaggle GPU: 2 slot bị kernel cuộc thi khác chiếm, sau đó quota tuần hết → chuyển sang Colab.
  Viết `colab/train_unet.ipynb` (resume theo epoch trên Drive, TTA maps, upload private dataset
  Kaggle). Đã chạy thử các cell local; chưa chạy thật trên Colab.

### 2026-10-22
- Tạo project, repo GitHub private, license MIT, venv Python 3.12 (`uv`).
- Đọc luật + overview → [PRIZE_RULES.md](../PRIZE_RULES.md). Giải chia theo rubric (70% định lượng
  / 30% định tính), không chỉ LB.
- EDA: 707 ảnh train / 1,154 bản gán nhãn (nhiều annotator), 180 ảnh test, 2048².
- Viết PQ khớp notebook self-evaluation của host (`src/filseg/metric.py`) + unit test.
- Classical baseline (flatten limb → hysteresis threshold → morphology → gom mảnh), random search
  ~140 vòng: val 0.235 → **LB 0.21**.
- **Phát hiện và sửa lỗi**: split cũ chia theo bản gán nhãn nên cùng một ảnh rơi vào cả train và
  val; cache ảnh bị ghi đè giữa các annotator. Đổi sang chia theo ảnh.
- U-Net ResNet34 @512 trên Mac (MPS, ~1.5 h): val 0.354 (đã tune) → **LB 0.29**.
