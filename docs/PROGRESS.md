# Nhật ký tiến độ

Mỗi phiên làm việc thêm một mục ở **đầu** phần "Nhật ký" (mới nhất trên cùng) và cập nhật bảng
"Trạng thái". Chi tiết từng thí nghiệm nằm ở [EXPERIMENTS.md](EXPERIMENTS.md), kế hoạch ở
[PLAN.md](PLAN.md).

## Trạng thái (cập nhật 2026-11-04)

| | |
|---|---|
| Submission tốt nhất | **Public LB 0.37**: 5 fold EffNet-B4 @1024 target mềm (N1) + hysteresis + lọc FP (OOF 0.425, 707 ảnh), 04/11 |
| Hạng | 483 / 1,260 với 0.36 (LB 2026-10-29 08:16 UTC) |
| LB tham chiếu | #1 0.70 nhưng bị nhiễm (rò rỉ nhãn, payload 0.55 × 82 đội); tầng trung thực cao nhất 0.42 (1 đội), 0.41 (11), 0.40 (46); xem [RESEARCH S16](RESEARCH.md#3-leaderboard-và-khoảng-cách-val--lb) |
| Submissions đã dùng | 27 (tối đa 5/ngày, reset 00:00 UTC) |
| Deadline | Kaggle: **2026-11-15 06:00 UTC** (13:00 VN). Report + code (Google form): Nov 15 11:59 pm CDT |
| Repo | https://github.com/nguyenchicuongvn/filament-segmentation-2026 (**private** đến khi cuộc thi đóng) |
| GPU | Kaggle GPU (T4 ×2, `scripts/kaggle_gpu.py`), ~20 h tuần này; đã dùng ~8 h cho 4 kernel, thêm ~6 h cho kernel 5–6 đang chạy. Colab chưa dùng |
| Đang chờ | Nộp final 1 ứng viên (ensemble 10 model N1 + N2); đo thời gian inference trên Kaggle cho notebook final |

## Theo dõi kế hoạch

Thẻ thí nghiệm và tiêu chí ở [PLAN.md § 4](PLAN.md#4-thẻ-thí-nghiệm). Cập nhật bảng này mỗi khi một
việc đổi trạng thái.

| Việc | Ở đâu | Trạng thái | Hạn / cổng | Kết quả |
|---|---|---|---|---|
| N0: sửa bug cache, bundle Colab mới | Mac | ✅ 29/10 | – | `dl.py` ghi atomic; bundle 2.2 MB |
| N1: 5 fold EffNet-B4 @1024 + target mềm | Kaggle kernel 1, 3, 4 | ✅ 04/11 | 31/10 | OOF hyst 0.4208, + lọc FP 0.4249; **LB 0.37** |
| N2: crop @2048 so với @1024, fold 0 | Kaggle kernel 2 | ❌ G1 04/11 | G1 30/10 | −0.0018 [−0.011, +0.007] đơn lẻ; nhưng **N1 + N2 trung bình +0.0133 [+0.007, +0.020]** → train fold 1–4 làm thành viên ensemble (kernel 5–6) |
| P1: `compare.py` scale grid theo độ phân giải | Mac | ✅ 29/10 | trước G1 | diện tích × r², bán kính × r; grid cc thêm ngưỡng 0.2 cho map target mềm. Chạy lại trên E9a ra đúng số cũ (cc 0.4015, hyst 0.4083, Δ +0.0068 [+0.003, +0.011]); mất 8.4 phút cho 213 ảnh @1024 |
| N3: gói v2 (EMA + blob loss + loss thành phần giả) | nhánh `exp/train-v2` → Colab Q4 | ❌ G2 04/11: −0.050 (f0), −0.080 (f1), FN tăng mạnh. Bỏ; trung bình N1 + v2 f0 vẫn +0.011 | G2 01–02/11 | |
| N4: Mask R-CNN ghép với U-Net | nhánh `exp/maskrcnn` → Colab | 📝 code + 5 test xong; thử @512 trên Mac **âm** (ghép −0.006) | G3 05/11 (go/no-go) | chỉ chạy fold 0 @1024 nếu còn giờ GPU |
| N5: metric rubric (torchmetrics Dice, phân bố, 1:n / n:1) | nhánh `exp/rubric-metrics` | ✅ 01/11 | 08/11 | E9a: Dice 0.668, 1:n 4.9%, n:1 3.7%; chạy lại trên model final |
| N6: final, weights fp16, notebook, report | Mac | 📝 | đóng băng 09/11; nộp 15/11 | |
| Người dùng: số giờ Colab | – | ✅ 29/10 | – | ~25 h L4, đủ cho kế hoạch đầy đủ |
| Người dùng: GPU Kaggle | – | ✅ 03/11 | – | được dùng, ~20 h tuần này, đẩy khi có slot trống |
| Kaggle GPU: 4 kernel (N1 ×5, v2 ×2, N2 ×1) | nhánh `exp/kaggle-gpu`, daemon | ✅ 04/11 04:07 | G1/G2 ~05/11 | 8 run xong, không lỗi; kernel đầu lỗi tìm code (Kaggle giải nén zip), đã sửa |
| Kaggle GPU: kernel 5–6 (N2 f1–f4) | daemon | ✅ 04/11 10:42 | ~11:00 04/11 | 4 run xong, ~6 h GPU; tổng đã dùng ~14 h / 20 h |
| ENS: N1 + N2 OOF 5 fold (G4) | Mac | ✅ qua 04/11 | trước 09/11 | **+0.0105 [+0.0072, +0.0140]**, 707 ảnh, dương 5/5 fold → final 1 |

## Lịch sử submission

| Ngày (UTC) | File | Mô tả | val PQ | Public LB |
|---|---|---|---|---|
| 2026-11-04 02:47 | `submission_effb4_s1024.csv` | N1 5 fold + hysteresis + lọc FP | 0.425 (OOF, in-sample) | **0.37** |
| 2026-11-03 23:19 | `submission_effb4_c2048_f0.csv` | N2 fold 0 + lọc FP | 0.434 (in-sample) | 0.36 |
| 2026-11-03 22:05 | `submission_effb4_v2_s1024_f0.csv` | v2 fold 0 + lọc FP | 0.368 (in-sample) | 0.32 |
| 2026-11-03 20:58 | `submission_effb4_s1024_f0.csv` | N1 fold 0 + lọc FP | 0.427 (in-sample) | 0.36 |
| 2026-10-27 23:38 | `submission_ens6_all_hyst_small07.csv` | ensemble 6 + hysteresis + co mask nhỏ | – | 0.35 |
| 2026-10-27 23:38 | `submission_ens6_all_hyst.csv` | ensemble 6 + hysteresis chặt | 0.409 (cross-tune) | 0.35 |
| 2026-10-27 23:24 | `submission_unet1024_effb4_hyst_strict150.csv` | EffNet-B4 + hysteresis, min_area 150 | – | 0.35 |
| 2026-10-27 23:24 | `submission_unet1024_effb4_hyst_small07.csv` | EffNet-B4 + hysteresis + co mask nhỏ (p > 0.7) | 0.409 (cross-tune) | 0.36 |
| 2026-10-25 18:2x | `submission_unet1024_effb4_hyst.csv` | E9a + hysteresis chặt (đỉnh > 0.99), không lọc | 0.408 (cross-tune) | 0.36 |
| 2026-10-25 04:04 | `submission_unet1024_r34_e60.csv` | E10 R34 60 epoch + lọc FP | 0.402 | 0.34 |
| 2026-10-25 03:40 | `submission_unet1024_cnxt.csv` | E9b ConvNeXt-Tiny + lọc FP | 0.392 | 0.33 |
| ~2026-10-25 | `submission_unet1024_effb4.csv` | E9a EfficientNet-B4 + lọc FP | 0.408 | **0.36** |
| ~2026-10-25 | `submission_unet1024_r34_grp.csv` | E8 R34 + kênh group + lọc FP | 0.399 | 0.34 |
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

### 2026-11-04
- 4 kernel Kaggle chạy xong lúc 04:07 (8 run, mọi exit code 0, ~8 h GPU). Kernel đầu tiên lỗi ở bước
  tìm code vì Kaggle giải nén zip của dataset; đã sửa (commit d8381a0), không tốn giờ GPU.
- **N1 5 fold**: OOF 707 ảnh hyst 0.4208, + lọc FP 0.4249; **LB 0.37** (tốt nhất mới, trước 0.36).
- **G1 (N2 @2048) không qua**: −0.0018 [−0.0114, +0.0074]. N2 bắt thêm 33 filament nhưng thêm 116 FP.
- **G2 (v2) bỏ**: −0.050 (f0), −0.080 (f1). Model quá dè dặt, FN tăng ~+230 mỗi fold.
- **Đo miễn phí ensemble** (fold 0, cùng 142 ảnh): N1 + N2 **+0.0133 [+0.0066, +0.0203]**, N1 + v2
  +0.0109 [+0.0053, +0.0170]. Trung bình 2 model giúp thật, kể cả khi model thứ hai yếu hơn. Trước
  đây E-ens6 (−0.001) là ensemble các model holdout yếu hơn hẳn E9a, nên số này là bằng chứng mới.
- Quyết định: dùng ~6 h GPU còn lại cho N2 fold 1–4 (kernel 5–6, đẩy lúc 07:45). Khi xong thì đo
  OOF N1 + N2 trên 707 ảnh (tiêu chí Δ ≥ +0.010, CI > 0). Câu hỏi còn mở: trên test, 5 fold N1 đã
  được trung bình, nên phần lợi do "trung bình nhiều model" có thể đã nằm sẵn trong LB 0.37; OOF
  không tách được phần này. Phần riêng của N2 là tính đa dạng về độ phân giải.
- Watcher tự nộp 4 bài (0.36, 0.32, 0.36, 0.37).
- Kiểm tra thêm: N1 + v2 trên fold 1 chỉ +0.0005 [−0.005, +0.005] (fold 0 là +0.011), và thêm v2 vào
  N1 + N2 −0.002. Lợi ích ensemble đổi nhiều theo fold, nên kỳ vọng cho G4 hạ xuống khoảng +0.005.
- Kernel 5–6 xong lúc 10:42 (N2 fold 1–4). **G4 qua**: OOF 707 ảnh N1 0.4179, N2 0.4129, trung bình
  N1 + N2 **0.4284, +0.0105 [+0.0072, +0.0140]**, dương cả 5 fold (fold 1 yếu nhất, +0.003). Lợi
  ích đến từ giảm FP (−222) chứ không phải tăng recall. Kỳ vọng +0.005 ở trên là quá bi quan.
- Final 1 = 10 model (N1 ×5 + N2 ×5, trọng số 1:1 giữa hai họ), hysteresis tune trên OOF ensemble +
  lọc FP. Final 2 = N1 5 fold (LB 0.37). Vẫn cần lưu ý: OOF không đo được phần lợi do trung bình
  5 fold ở test.

### 2026-11-03
- Người dùng cho phép dùng GPU Kaggle (~20 h tuần này, đẩy khi có slot trống). Viết
  `scripts/kaggle_gpu.py` (nhánh `exp/kaggle-gpu`): code thành dataset riêng tư; mỗi kernel T4 ×2
  train 2 run song song; daemon đẩy theo hàng đợi và tự fetch. Lần push đầu bị từ chối: 2 phiên GPU
  đang bị kernel của project khác chiếm (`casmi26-fs-class3-panel`, `umud-orient-gpu-v1`). Daemon
  đang chờ slot.
- Quét toàn bộ không gian lựa chọn bằng hai agent (input/target; model/train/inference):
  [RESEARCH § 11](RESEARCH.md#11-quét-toàn-bộ-không-gian-lựa-chọn-0311-input-tiền-xử-lý-target-model-train-inference).
  Phát hiện chính: **model mất 0.070 PQ ở ảnh seeing sắc nét, người thì không** (F36); làm mờ input
  −0.096 (F37). Vì vậy N2 được đưa lên kernel thứ 2. Phía input, tiền xử lý và augmentation không còn
  đòn bẩy (F39–F41). ConvNeXt thiếu skip stride 2 (F42). Grid hysteresis cũ cắt mất filament trên
  map target mềm, nên đã đăng ký trước grid `hyst_soft` (F43).

### 2026-11-01
- Merge `research/new-solution` và `exp/train-v2` vào main (19 test pass), đóng lại bundle Colab.
  Q3 chưa chạy (người dùng xác nhận). Hàng đợi mới, 8 run, ~10.5 h L4: ba run fold 0 (N1, N2,
  v2) trước, rồi v2 fold 1 và N1 fold 1–4. Watcher khởi động lại với danh sách mới. GPU Kaggle chỉ
  dùng khi Colab lỗi (người dùng quyết định).
- N4 (nhánh `exp/maskrcnn`): code Mask R-CNN, detector một mình và ghép với U-Net, spec
  `tune:det` / `tune:fuse` trong `compare.py`, Colab và watcher hỗ trợ run `mrcnn_*`. Thử @512,
  8 epoch trên Mac (split 70% của E9a): detector một mình 0.250, ghép −0.006 [−0.012, +0.000]. Âm,
  nhưng detector @512 yếu; chỉ thử @1024 fold 0 nếu còn giờ GPU.
- N5 (nhánh `exp/rubric-metrics`): `scripts/rubric_metrics.py` (torchmetrics Dice, PQ/SQ/RQ, phân bố
  Dice/IoU, 1:n / n:1). E9a: Dice 0.668, 1:n 4.9%, n:1 3.7%.

### 2026-10-30 (workflow nhiều agent)
- Theo yêu cầu "dùng multi agent để tìm solution mới": 3 agent điều tra (quyết định instance, cách
  train, pháp y lỗi) → judge → 2 build trên nhánh riêng → 2 reviewer. Không dùng GPU. Kết quả ở
  [RESEARCH § 10](RESEARCH.md#10-workflow-nhiều-agent-3010-tìm-ý-tưởng-cải-thiện), báo cáo nguyên
  văn ở `docs/logs/wf30/`.
- Kết quả chính: tầng chọn instance đã cạn (H6 −0.004 dù oracle +0.125; bộ lọc đồng thuận −0.0004).
  Phần mất lớn nhất là bất đồng giữa annotator (FP khớp người khác 29%, FN chỉ một người vẽ 20%).
  Phần model gỡ được: đường biên, cách chia instance, ~120 FP giả.
- Nhánh: `exp/agree-filter` (âm, code harness dùng lại được), `exp/train-v2` (N3', đã sửa theo
  review, chờ GPU), `research/new-solution` (docs + script + log). Chưa merge vào main.

### 2026-10-30 (W-agree, CPU)
- Thử đặc trưng đồng thuận giữa 5 model holdout cho bộ lọc instance, đăng ký trước, cross-tune cả
  grid + bộ lọc + ngưỡng (`compare.py ... tune:hyst,filter:base+agree --agree-maps ...`).
  Đối chứng tái lập đúng 0.4083. Kết quả **âm**: 0.4079, Δ −0.0004 [−0.003, +0.002]; so với bộ lọc
  17 đặc trưng Δ −0.0001, P>0 0.47. AUC OOF +0.010 nhưng PQ không đổi. Bỏ, không nộp
  (CSV ở `outputs/wf_cpu/`, mask đã kiểm tra không chồng nhau). Ghi thêm vào sổ cái: H6 (cây quyết
  định) −0.004 [−0.011, +0.003] → bác bỏ; lỗi vị trí/limb/sunspot < 1%. Log
  `docs/logs/wf_agree_filter.txt`.

### 2026-10-29 (chiều)
- Viết lại docs theo yêu cầu "research tỉ mỉ, có dẫn chứng, plan rõ ràng, theo dõi tiến độ":
  - [RESEARCH.md](RESEARCH.md): mỗi nhận định có mã (`S*` nguồn ngoài, `F*` số đo của mình,
    `H*` giả thuyết), link và mức tin cậy; các số liệu được đối chiếu lại với nguồn.
  - [PLAN.md](PLAN.md): ràng buộc thời gian/GPU, quy trình thí nghiệm (độ nhạy một fold ±0.012,
    đăng ký tiêu chí trước), thẻ thí nghiệm N1–N6, cổng quyết định G1–G3, câu hỏi cho người dùng.
  - [EXPERIMENTS.md § Sổ cái](EXPERIMENTS.md#sổ-cái): mọi thay đổi đã thử kèm Δ, CI, kết luận.
- Người dùng chạy Q3 trên Colab với bundle mới; watcher local đang chờ.

### 2026-10-29
- **Tự phản biện lần 2**, ghi ở [EXPERIMENTS.md](EXPERIMENTS.md). Script mới: `scripts/analyze_errors.py`,
  `scripts/res_ceiling.py`.
- Model đơn của mình (0.41) gần ngang model đơn của các đội đầu (~0.42 mỗi fold). Khoảng cách nằm ở
  ba chỗ: model nộp chỉ train trên 70% dữ liệu, không có ensemble fold, và thiếu loss/detector có ý
  thức về instance. Đội đầu có FP/TP 0.42 so với 0.57 của mình.
- Khoảng cách val → LB chủ yếu do lệch annotator: PQ theo annotator dao động 0.29–0.55; nhóm 01/04
  chỉ được ~0.38, các nhóm khác 0.43–0.46. Host không công bố ai gán nhãn test.
- **Bug**: cache `.npz` không ghi atomic, nên một run bị ngắt để lại file cụt và mọi run sau đó chết
  ngay (`EOFError`). Có thể đây là lý do hàng đợi Colab không ra kết quả. Đã sửa (ghi atomic, file
  hỏng thì build lại).
- Hàng đợi Colab thứ ba: fold 0 của hai công thức EffNet-B4 (@1024 và crop @2048), rồi 4 fold còn
  lại của @1024.
- Research mới (theainerd, umerkang66, biconcavelens, ranjeet258, rubric của host):
  [RESEARCH.md § 6](RESEARCH.md#6-cập-nhật-2026-10-29). Kế hoạch mới N0–N6 ở [PLAN.md](PLAN.md).

### 2026-10-27
- Nộp 4 biến thể hậu xử lý / ensemble trên các map 1024 có sẵn: không biến thể nào vượt 0.36
  (EffNet-B4 + hysteresis + co mask nhỏ 0.36; giữ chặt hơn 0.35; ensemble 6 model 0.35 với cả 2 cách).
  Khớp với kết quả tự phản biện: hậu xử lý đã cạn, ensemble model yếu không giúp. Giả thuyết "nhãn
  test thưa → giữ chặt hơn" không được LB xác nhận (0.35, trong mức nhiễu).
- **Hàng đợi Colab thứ hai không ra kết quả nào sau ~3 ngày** (không có dataset `filseg-*_f*` nào trên
  Kaggle). Đây là điểm nghẽn: muốn tăng thêm phải có map tốt hơn (5 fold, độ phân giải gốc).

### 2026-10-25
- **Tự phản biện** (`scripts/compare.py`, bootstrap ghép cặp + cross-tune): EffNet-B4 > R34 là thật;
  ensemble các model holdout không giúp; nhãn test có vẻ thưa hơn, nên giữ instance chặt có lợi.
  Ý tưởng mới đã kiểm chứng: hysteresis chặt +0.007 [+0.002, +0.011], chỉnh mask theo kích thước
  +0.005, nhưng hai cái không cộng dồn và tương đương lọc FP. Submit EffNet-B4 + hysteresis →
  **LB 0.36** (như dự báo). Watcher giờ tune cả hysteresis cho các run fold.
- Hàng đợi Colab thứ hai: sau nhiều giờ vẫn chưa có run fold nào trên Kaggle, cần kiểm tra notebook.
- Watcher submit xong E8–E10: **EfficientNet-B4 LB 0.36** (tốt nhất), R34+group 0.34, R34 60 epoch
  0.34, ConvNeXt 0.33. Người dùng đã chạy hàng đợi Colab thứ hai (11 run fold) trên L4.
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
