# Nhật ký tiến độ

Mỗi phiên làm việc thêm một mục ở **đầu** phần "Nhật ký" (mới nhất trên cùng) và cập nhật bảng
"Trạng thái". Chi tiết từng thí nghiệm nằm ở [EXPERIMENTS.md](EXPERIMENTS.md), kế hoạch ở
[PLAN.md](PLAN.md).

## Trạng thái (cập nhật 2026-10-24)

| | |
|---|---|
| Submission tốt nhất | **Public LB 0.34**: U-Net R34 @1024 + lọc FP (val 0.394), và ensemble 768 + 1024 + lọc FP (val 0.399) |
| Hạng | ~767 / 1,153 với 0.31 (snapshot sáng 2026-10-24, trước khi nộp 0.33) |
| LB tham chiếu | #1 0.69 nhưng bị nhiễm (rò rỉ nhãn, payload 0.55 × 76 đội); đội trung thực tốt nhất ~0.40; trung vị 0.34 |
| Submissions đã dùng | 8 (tối đa 5/ngày, reset 00:00 UTC) |
| Deadline | Kaggle: **2026-11-15 06:00 UTC** (13:00 VN). Report + code (Google form): Nov 15 11:59 pm CDT |
| Repo | https://github.com/nguyenchicuongvn/filament-segmentation-2026 (**private** đến khi cuộc thi đóng) |
| GPU | Kaggle GPU quota tuần này đã hết → train GPU trên Colab (`colab/train_unet.ipynb`); Mac M5 (MPS) cho run nhỏ |
| Đang chờ | Người dùng chạy hàng đợi Colab thứ hai (R1–R3, 11 run); watcher trên Mac đang xử lý các run E8–E10 cũ và chờ các run fold (log: `outputs/colab_watch.log`) |

## Lịch sử submission

| Ngày (UTC) | File | Mô tả | val PQ | Public LB |
|---|---|---|---|---|
| 2026-10-24 23:53 | `submission_ens_768_1024r34_nofilt.csv` | trung bình map E5 + E7, không lọc FP | 0.394 | 0.33 |
| 2026-10-24 23:52 | `submission_ens_768_1024r34.csv` | trung bình map E5 + E7 + lọc FP | 0.399 | **0.34** |
| 2026-10-24 23:47 | `submission_unet1024_r34.csv` | E7, tune nhanh 12 tổ hợp + lọc FP | 0.396 | 0.33 |
| 2026-10-24 23:40 | `submission_unet1024_r34.csv` | E7 (watcher, tune đầy đủ) + lọc FP | 0.394 | **0.34** |
| 2026-10-24 08:37 | `submission_unet768_filt.csv` | E5 + lọc FP (ngưỡng 0.225) | 0.389 | **0.33** |
| 2026-10-23 03:47 | `submission_unet768.csv` | U-Net R34 @768, TTA, post t0.4 a50 g4 | 0.383 | **0.31** |
| 2026-10-22 20:58 | `submission_unet512.csv` | U-Net R34 @512, TTA, post t0.4 a25 g4 | 0.354 | 0.29 |
| 2026-10-22 16:48 | `submission.csv` | Classical baseline v1 | 0.242* | 0.21 |

\* Đo lại trên split chia theo ảnh (sau khi sửa lỗi rò rỉ, xem 2026-10-22).

Val luôn cao hơn LB khoảng 0.03–0.07, nhưng thứ tự giữa các model giữ nguyên, nên val vẫn dùng
được để chọn model (xem [DATA_AND_METRIC.md](DATA_AND_METRIC.md#khoảng-cách-val--lb)).

## Nhật ký

### 2026-10-25
- Dựng R1–R3 (5-fold, target mềm, crop @2048, map OOF) thành hàng đợi Colab thứ hai gồm 11 run;
  watcher biết ghép fold. R4 (watershed + lấp lỗ) không giúp; R5: đặc trưng vị trí trên đĩa +0.002,
  regressor IoU kém hơn classifier.
- **Research** ([RESEARCH.md](RESEARCH.md)): top LB bị nhiễm do rò rỉ nhãn test (MAGFiLO 1.0 công
  khai) và payload 0.55 (76 đội). Đội trung thực tốt nhất đạt OOF 0.45–0.46 / public 0.39–0.40; hai
  chuyên gia chấm nhau chỉ được 0.34. Lỗi chính của mình là sai biên (gần trúng), không phải tách/gộp.
  Viết lại plan (R1–R9). Host yêu cầu mới: weights phải nằm trong repo.
- Session Colab L4 chạy xong E7–E10 trong ~4.7 h (nhanh hơn ước tính ~3 lần: 1 min/epoch với R34
  @1024). Val PQ thô: EfficientNet-B4 **0.381** > R34+group 0.364 ≈ R34 60 epoch 0.364 > R34 0.357 >
  ConvNeXt-Tiny 0.342.
- Watcher không submit được run nào: (1) tải nguyên dataset bị 404 → sửa sang tải từng file; (2) run
  `unet1024_r34` upload lỗi vì trùng slug với một notebook → `kaggle_upload.py` giờ báo lỗi, và run
  này cần được upload lại hoặc infer lại từ checkpoint trên Drive.
- Lần 2 watcher bị Kaggle rate-limit (429) khi tải 16 file song song → giảm còn 4 luồng, chờ lùi khi
  bị 429; upload mới đóng map thành `probs_*.tar`.
- E7: tải checkpoint từ Drive, infer TTA trên Mac (`LOCAL` marker), watcher submit → **LB 0.34**.
  Trước giờ reset (00:00 UTC) nộp thêm: E7 tune nhanh (0.33), ensemble 768 + 1024 có lọc (**0.34**)
  và không lọc (0.33). Lọc FP vẫn giúp ~+0.01 trên LB. Config `post_unet1024_r34.json` /
  `fpfilter_unet1024_r34.pkl` hiện là bản tune nhanh (bản của watcher bị ghi đè).

### 2026-10-24
- Viết bộ docs theo dõi: `docs/PROGRESS.md`, `EXPERIMENTS.md`, `DATA_AND_METRIC.md`, `PLAN.md`,
  `docs/logs/`, `CLAUDE.md`.
- Dựng các thí nghiệm còn lại của plan:
  - Watershed (E5w): 0.384, không hơn cc → tách instance cần học, không phải luật.
  - Lọc FP theo instance (E5f): val 0.383 → 0.389, **LB 0.31 → 0.33**.
  - Kênh group cho U-Net (`--targets group`, post `--mode group`), encoder EfficientNet-B4 /
    ConvNeXt-Tiny, run 60 epoch, run full-data: tất cả nằm trong hàng đợi `EXPERIMENTS` của notebook
    Colab (E8–E11). Đã smoke-test toàn bộ đường đi trên Mac (256 px, 2 epoch).
- Notebook chạy cả E7–E10 trong một session; `scripts/watch_colab.py` tự submit từng run ngay khi
  có kết quả, và submit ensemble nếu val tốt hơn. Đã test offline (giải nén, tune 2 mode, filter, CSV).

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
