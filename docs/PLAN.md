# Kế hoạch

Mục tiêu: lọt top-3 theo rubric của ban tổ chức (70% định lượng, 30% định tính; xem
[PRIZE_RULES.md](../PRIZE_RULES.md)). Muốn cạnh tranh thì cần PQ khoảng ≥ 0.5 trên LB, đồng thời
code và report phải tốt.

## Mốc thời gian

| Ngày | Việc |
|---|---|
| 2026-11-15 06:00 UTC (13:00 VN) | Hạn submission Kaggle; chọn 2 final submissions |
| 2026-11-15 11:59 pm CDT | Hạn Google form: report 4 trang (PDF) + link repo public |
| 2026-11-15, ngay khi đóng | Chuyển repo GitHub sang **public** (`gh repo edit ... --visibility public`) |
| 2026-11-30 | Công bố người thắng |

Nên chốt model trước ~2026-11-10 để còn thời gian viết report và dọn code.

## Mục tiêu thực tế (sau research 2026-10-25, xem [RESEARCH.md](RESEARCH.md))

LB public bị nhiễm (rò rỉ nhãn test qua MAGFiLO 1.0, payload 0.55). Các đội trung thực tốt nhất có
công bố đạt **OOF ~0.45–0.46 / public ~0.40**; hai chuyên gia chấm lẫn nhau chỉ được 0.34. Mục tiêu:
**OOF ≥ 0.45, public ~0.40**, kèm code tái lập được và report tốt (30% điểm). Mình đang ở 0.399
holdout / 0.34 public.

## Kế hoạch sau tự phản biện lần 2 (2026-10-29)

Xem [EXPERIMENTS.md § Tự phản biện lần 2](EXPERIMENTS.md#tự-phản-biện-lần-2-2026-10-29-mình-sai-ở-đâu-khoảng-cách-thật-nằm-ở-đâu).
Hậu xử lý đã cạn. Khoảng cách ~0.04 so với các đội trung thực đầu bảng đến từ ba chỗ: model nộp chỉ
train trên 70% dữ liệu, không có ensemble fold, và thiếu loss/thành viên có ý thức về instance.

| # | Việc | Bằng chứng | Ước tính | Chi phí |
|---|---|---|---|---|
| N0 | **Mở lại đường GPU**: sửa bug cache (xong), bundle mới, hàng đợi thứ ba (6 run, ~7–8 h L4); hỏi trước khi dùng Kaggle GPU | 4 ngày không có run nào | – | thấp |
| N1 | **5 fold EffNet-B4 @1024 + target mềm**: tune trên OOF 707 ảnh, nộp trung bình 5 fold | công thức E9a đã biết; các đội top đều nộp ensemble fold | **+0.01–0.02 LB**, rủi ro thấp | 5 × ~1 h L4 |
| N2 | So **@1024 cả ảnh với crop @2048** trên fold 0, rồi mới chạy fold @2048 | theainerd thắng với tile 512 native; BSergey thấy native kém 0.07 | ? | 2 run |
| N3 | Ablation fold 0, chỉ giữ cái nào có CI > 0: **EMA**, **blob loss**, **BiCC**, input đã hiệu chỉnh limb (3 kênh) | umerkang66 blob +0.009 (CI); theainerd BiCC +0.003–0.022 trên 4/5 fold | +0.01–0.02 | mỗi cái 1 run |
| N4 | **Thành viên detector**: Mask R-CNN (torchvision, BSD; đã chạy thử trên MPS) @1024, rồi ghép với U-Net bằng score-weighted mask vote | biconcavelens: Mask R-CNN 0.429 / YOLO11m 0.437 so với U-Net 0.396; vote +0.007–0.012 | +0.02–0.04 (trần cao nhất) | lớn: code mới, train ~1 ngày |
| N5 | Metric rubric (Dice, phân bố IoU/Dice, 1:n / n:1) trong `analyze_errors.py`, đưa vào report | 70% điểm định lượng | điểm rubric | thấp |
| N6 | Model cuối + weights trong repo + notebook (R9), chốt trước 2026-11-10 | yêu cầu host | bắt buộc | trung bình |

**Quy tắc submit mới**: chỉ nộp khi model thay đổi có CI > 0 trên OOF/fold. Không nộp biến thể hậu
xử lý nữa. Giữ 2 bài cuối: một bài OOF tốt nhất, một bài ít rủi ro nhất (ensemble 5 fold @1024).

YOLO11-seg (Ultralytics) mạnh nhất trong số detector nhưng có license AGPL-3.0. AGPL là license OSI
nên được luật cho phép, nhưng repo mình dùng MIT. Chỉ dùng nếu người dùng đồng ý; nếu không thì dùng
Mask R-CNN của torchvision.

## Việc tiếp theo (theo thứ tự, kế hoạch cũ 2026-10-25)

| # | Việc | Lý do / bằng chứng | Trạng thái |
|---|---|---|---|
| R1 | **5-fold CV theo ảnh**, chấm OOF PQ trên cả 707 ảnh; tune post/lọc FP trên OOF | holdout 213 ảnh quá nhiễu (SD ~0.014); mọi đội top dùng OOF | dựng xong, chờ Colab |
| R2 | **Target mềm đa annotator** (tỉ lệ annotator vẽ pixel), lấy mẫu theo số bản gán nhãn | umerkang66, +0.01–0.03 | dựng xong, chờ Colab |
| R3 | **Độ phân giải gần gốc**: train crop 1024 từ ảnh 2048, infer full 2048 + TTA 8 | mọi đội top; 768 → 1024 đã +0.01 | dựng xong, chờ Colab |
| R4 | **Tách instance**: seed + gộp seed gần + watershed + lấp lỗ | umerkang66 +0.008; GT không có lỗ | **xong**: không giúp (0.3943 → 0.3944) |
| R5 | **Bộ lọc → regressor IoU kỳ vọng**; thêm đặc trưng vị trí so với kinh tuyến trung tâm | umerkang66, theainerd; paper MAGFiLO | **xong**: đặc trưng vị trí +0.002 (giữ); regressor kém hơn (bỏ) |
| R6 | Ensemble encoder: EffNet-B4 (tốt nhất ở E9) + R34 + EffV2-S / RegNetY, theo fold | Dariush, umerkang66 | sau R1–R3 |
| R7 | Head phụ skeleton/core + blob loss | +0.009 có CI | sau R6 |
| R8 | (Nếu còn thời gian) detector + crop refiner làm thành viên ensemble | biconcavelens 0.40 → 0.461 | tuỳ thời gian |
| R4b | **Hysteresis chặt** (đỉnh > 0.95–0.99) + chỉnh mask theo kích thước | tự phản biện 2026-10-25: +0.007 có CI, test có vẻ thưa nhãn | **xong**: dùng mặc định (watcher tune cả cc và hyst) |
| R9 | Model cuối: công thức thắng, train 5 fold (hoặc full), **weights lưu trong repo (Git LFS)**, notebook load weights | yêu cầu mới của host (24/10) | cuối cùng |

Không làm: dùng MAGFiLO 1.0 công khai, mask chồng nhau, tune theo LB, clDice/CLAHE/pseudo-label/
Mask2Former/SAM (đã có người thử và kém).

Phần đã xong trước đó: E5w watershed thuần (không giúp), E5f lọc FP (+0.01 LB), E7–E10 trên Colab
(EffNet-B4 tốt nhất, ConvNeXt kém, 60 epoch giúp ít).

## Hồ sơ nhận giải (bắt buộc)

- [x] License MIT
- [x] `requirements.txt` pin version
- [x] README: cài đặt, môi trường, thời gian chạy
- [ ] Jupyter notebook chạy toàn bộ pipeline, **load weights có sẵn trong repo** (tạm hoãn, làm khi chốt model)
- [ ] Weights trong repo (Git LFS, mỗi file 80–120 MB > giới hạn 100 MB của GitHub)
- [ ] Technical report 4 trang theo template Overleaf của host (viết khi có kết quả tốt)
- [ ] Repo public ngay khi cuộc thi đóng, giữ public đến khi công bố
- [ ] Nộp Google form trước Nov 15 11:59 pm CDT
- [ ] Hiểu và giải thích được mọi thành phần (luật cho phép dùng AI nhưng yêu cầu điều này)
