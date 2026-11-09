# Nhật ký tiến độ

Mỗi phiên làm việc thêm một mục ở **đầu** phần "Nhật ký" (mới nhất trên cùng) và cập nhật bảng
"Trạng thái". Chi tiết từng thí nghiệm nằm ở [EXPERIMENTS.md](EXPERIMENTS.md), kế hoạch ở
[PLAN.md](PLAN.md).

## Trạng thái (cập nhật 2026-11-09)

| | |
|---|---|
| Submission tốt nhất | **Public LB 0.38**: final 1 mới = N1 ×5 + N2 ×5 + N9' ×5 (EffNetV2-S crop 512 @2048) + hysteresis + gộp theo hướng + lọc FP (OOF 0.4372 in-sample, 707 ảnh), qua gate gộp 09/11. N9' 5 fold đơn cũng 0.38 |
| Hạng | 483 / 1,260 với 0.36 (LB 2026-10-29 08:16 UTC) |
| LB tham chiếu | #1 0.70 nhưng bị nhiễm (rò rỉ nhãn, payload 0.55 × 82 đội); tầng trung thực cao nhất 0.42 (1 đội), 0.41 (11), 0.40 (46); xem [RESEARCH S16](RESEARCH.md#3-leaderboard-và-khoảng-cách-val--lb) |
| Submissions đã dùng | 51 (tối đa 5/ngày, reset 00:00 UTC) |
| Deadline | Kaggle: **2026-11-15 06:00 UTC** (13:00 VN). Report + code (Google form): Nov 15 11:59 pm CDT |
| Repo | https://github.com/nguyenchicuongvn/filament-segmentation-2026 (**private** đến khi cuộc thi đóng) |
| GPU | Kaggle GPU (T4 ×2, `scripts/kaggle_gpu.py`), ~20 h tuần này; đã dùng ~8 h cho 4 kernel, thêm ~6 h cho kernel 5–6 đang chạy. Colab chưa dùng |
| Đang chờ | Không có run nào đang chạy. N11 và N12 xong 09/11, cả hai không qua gate 707 → final 1 giữ N1 + N2 + N9'. Tái lập xong (kernel v2: final 1 LB 0.38, final 2 trùng byte) và weights fp32 đã vào `weights/`. Còn lại trước đóng băng 12/11: report |

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
| N7: BCE thuần (bỏ Dice), fold 0+1 | Kaggle kernel 7 | ❌ 04/11 22:28 | – | đơn lẻ −0.0002 / −0.0018; thêm vào N1 + N2: +0.0060 [+0.001, +0.011] (f0), +0.0038 [−0.001, +0.009] (f1), chưa đạt tiêu chí. Map hiệu chỉnh tốt nhưng bộ lọc vẫn không giúp. Tổng GPU ~16.3 h / 20 h |
| N7: BCE fold 2–4 (người dùng duyệt 05/11) | Kaggle kernel 8–9 (`bce-f2f3`, `bce-f4`) | ✅ về 09:26 05/11 | – | ~4.5 h giờ phiên GPU, tổng ~20.8 h. Gate trên 707 ảnh OOF đang chạy (`docs/logs/kgpu04/n7ens_oof.txt`) |
| N8: N1 seed 2, 5 fold (người dùng duyệt 05/11: "GPU hết thì qua Colab") | Colab, `colab/train_unet.ipynb` | 📝 bundle đóng gói 05/11, chờ người dùng chạy | cần về trước 08/11 | ~6.3 h L4. Gate: ensemble có N8 hơn không có, CI > 0 trên 707 OOF (thẻ N8) |

## Lịch sử submission

| Ngày (UTC) | File | Mô tả | val PQ | Public LB |
|---|---|---|---|---|
| 2026-11-09 23:50 | `submission_ens_n1n2n9p_orient_nofilt.csv` | loại 3 (ablation lọc FP trên final 1 cho bảng report; người dùng yêu cầu dùng 2 lượt còn lại trước 00:00 UTC): final 1 bỏ lọc FP | OOF 0.4354 so với 0.4372 (Δ −0.0018, dưới độ phân giải LB) → dự báo LB 0.38 như final 1. Đúng dự báo | **0.38** |
| 2026-11-09 23:50 | `submission_ens_n1n2_orient_nofilt.csv` | loại 3 (ablation lọc FP trên final 2): N1 + N2 bỏ lọc FP | OOF 0.4313 so với 0.4327 (Δ −0.0014) → dự báo LB 0.37 như final 2. Đúng dự báo | **0.37** |
| 2026-11-09 ~18:30 | `submission_n9p_orient_nofilt.csv` | loại 3 (ablation lọc FP cho report): N9' 5 fold đơn như dòng 08/11 (LB 0.38) nhưng **bỏ lọc FP**; lọc FP trên N9' cho +0.013 OOF, mức duy nhất trong các bộ lọc đủ lớn để LB có thể thấy | OOF 0.4192 (không lọc) so với 0.4322 (lọc, in-sample) → Δ dự báo −0.013 → LB 0.36–0.37. Thực tế **bằng bản lọc**: +0.013 OOF của bộ lọc (fit in-sample) không thấy trên LB | **0.38** |
| 2026-11-09 ~19:10 | `submission_r34_c512_orient.csv` | loại 3 (LB thành viên khác kiến trúc cho report): N12 ResNet34 5 fold đơn, hậu xử lý tune trên 707 OOF, không lọc FP. Gate pool 707 không qua (−0.0004) | OOF dự báo ~0.40 (từng fold 0.395–0.419) → LB 0.35–0.36. Thực tế OOF **0.3955** (t 0.5 seed 0.99, FP 5092 so với N9' ~3700). Đúng dự báo | **0.36** |
| 2026-11-09 ~20:00 | kernel `filseg-final-infer` v2: `submission_final1.csv` | loại 1 (tái lập): final 1 mới N1 + N2 + N9' (15 checkpoint) chạy lại từ đầu trên Kaggle. Đạt nếu trùng từng byte với `submission_ens_n1n2n9p_orient.csv`; không thì map lệch ≤ 1 mức uint8 trên < 0.1% pixel và LB 0.38 | như final 1 (0.4372) → LB 0.38. Thực tế: map N1, N2 và N1 + N2 trùng 180/180; map N9' khác (map nộp 08/11 tạo trên Colab L4, kernel trên Kaggle T4 torch 2.11.0+cu130), 23/987 RLE khác. Không trùng byte nhưng LB bằng → **đạt** phần LB; phần lệch pixel không đo được (kernel xoá map). Từ nay final 1 = CSV do kernel tạo | **0.38** |
| 2026-11-09 (không nộp) | kernel `filseg-final-infer` v2: `submission_final2.csv` | loại 1 (tái lập): final 2 = N1 + N2 + lọc FP (final 1 cũ). Đạt nếu trùng từng byte với `submission_ens_n1n2_orient.csv` | như final 1 cũ (0.4327) → LB 0.37. **Trùng từng byte** với `submission_ens_n1n2_orient.csv` (đã có LB 0.37) nên không tốn lượt nộp lại | 0.37 (trùng byte) |
| 2026-11-08 ~23:30 | `submission_ens_n1n2n9p_orient.csv` | loại 3 (số cho report, ghi trước khi biết gate 707): pool N1 ×5 + N2 ×5 + N9' ×5 (1:1:1), hysteresis + gộp theo hướng (r 6, L16), lọc FP fit trên pool OOF. Gate từng fold: pool hơn N1 + N2 ở f0 +0.0068, f1 −0.0008, f2 +0.0072, f3 +0.0007, f4 +0.0049 | Δ OOF dự báo +0.004 so với final 1 → LB 0.37–0.38. Thực tế: lọc FP fit trên pool OOF 0.4354 → **0.4372** (in-sample, t 0.5 seed 0.99) | **0.38** |
| 2026-11-08 ~23:30 | `submission_n9p_orient.csv` | loại 3 (LB thành viên cho report): N9' (EffNetV2-S, crop 512 @2048) 5 fold trung bình, hysteresis + gộp theo hướng, lọc FP riêng | OOF dự báo ~0.42 (≈ N1) → LB 0.36–0.37. Thực tế: không lọc 0.4192 (t 0.4 seed 0.99), lọc FP → **0.4322** (+0.013, in-sample) | **0.38** |
| 2026-11-08 07:09 | `submission_ens_n1n2_cc_orient.csv` | loại 3 (ablation cho report): final 1 bỏ hysteresis (mode cc, cùng t 0.4, area 200, gộp theo hướng), không lọc FP; so với dòng t 0.4 không lọc (0.37) | OOF −0.0248 [−0.0290, −0.0204] so với hysteresis không lọc (0.4313 → 0.4065) → LB 0.34–0.36 | 0.36 |
| 2026-11-08 08:34 | kernel `filseg-final-infer`: `submission_final1.csv` | loại 1 (tái lập): final 1 chạy lại từ đầu trên Kaggle bằng `notebooks/final_inference.ipynb` (10 checkpoint, TTA 8, ensemble, hậu xử lý, lọc FP). Đạt nếu CSV trùng từng byte với `submission_ens_n1n2_orient.csv`; nếu không trùng thì map lệch tối đa 1 mức uint8 trên < 0.1% pixel và LB 0.37 | như final 1 (0.4327) → LB 0.37 | 0.37 |
| 2026-11-08 08:34 | kernel `filseg-final-infer`: `submission_final2.csv` | loại 1 (tái lập): final 2 từ cùng kernel; cùng tiêu chí, so với `submission_n1up.csv` | như final 2 (0.4229) → LB 0.37 | 0.37 |
| 2026-11-05 09:10 | `submission_ens_n1n2bce.csv` | **15 model** N1 + N2 + BCE (1:1:1) + hysteresis (t 0.4, seed 0.9, area 200, r 12) + lọc FP. Giả thuyết: 15 so với 10 model trên test | 0.4325 (OOF, in-sample); cross-tune +0.0018 so với N1 + N2, CI chạm 0 → không vào final | 0.37 |
| 2026-11-05 09:00 | `submission_ens_n1n2_r0.csv` | ablation: N1 + N2, hysteresis t 0.4, **không gộp mảnh** (r 0), không lọc FP. Giả thuyết: gộp r 12 đáng +0.0026 OOF, LB không thấy | – | 0.37 |
| 2026-11-07 00:02 | `submission_effb4_s1024_f1.csv` | loại 3 (số cho report): N1 fold 1 đơn, hysteresis + lọc FP cấu hình fold 0 | OOF fold 1 ~0.41–0.43 → LB 0.35–0.37 (fold 0 đã 0.36) | 0.36 |
| 2026-11-07 00:03 | `submission_effb4_s1024_f2.csv` | loại 3 (số cho report): N1 fold 2 đơn, hysteresis + lọc FP cấu hình fold 0 | OOF fold 2 ~0.41–0.43 → LB 0.35–0.37 (fold 0 đã 0.36) | 0.36 |
| 2026-11-07 00:03 | `submission_effb4_s1024_f3.csv` | loại 3 (số cho report): N1 fold 3 đơn, hysteresis + lọc FP cấu hình fold 0 | OOF fold 3 ~0.41–0.43 → LB 0.35–0.37 (fold 0 đã 0.36) | 0.36 |
| 2026-11-07 00:03 | `submission_effb4_s1024_f4.csv` | loại 3 (số cho report): N1 fold 4 đơn, hysteresis + lọc FP cấu hình fold 0 | OOF fold 4 ~0.41–0.43 → LB 0.35–0.37 (fold 0 đã 0.36) | 0.36 |
| 2026-11-07 00:04 | `submission_effb4_c2048.csv` | loại 3: N2 5 fold đơn (crop 1024 @2048), cấu hình fold 0 | OOF 0.4129 (N1 −0.005) → LB 0.36–0.37 | 0.37 |
| 2026-11-06 02:07 | `submission_ens_n1n2_orient.csv` | **final 1 mới**: N1 + N2 + hysteresis + gộp theo hướng (L16, r6) + lọc FP mới | OOF +0.0021 cross-tune (0.4327 sau lọc) | **0.37** |
| 2026-11-06 01:49 | `submission_n1up.csv` | **final 2**: N1 5 fold, map phóng lên 2048 trước hậu xử lý, không lọc FP | OOF 0.4229 (+0.0050 so với N1 @1024) | **0.37** |
| 2026-11-06 01:49 | `submission_ens_n1n2_drop15.csv` | thang suy giảm: final 1 bỏ 15% instance điểm lọc thấp nhất | Δ OOF +0.005 (in-sample) → LB 0.37 | **0.35** |
| 2026-11-06 01:50 | `submission_ens_n1n2_drop30.csv` | thang suy giảm 30% | Δ OOF −0.019 → LB ~0.35 | **0.33** |
| 2026-11-06 01:50 | `submission_ens_n1n2_drop50.csv` | thang suy giảm 50% | Δ OOF −0.085 → LB ~0.30 | **0.27** |
| 2026-11-05 08:35 | `submission_ens_n1n2_t0.5.csv` | thăm dò P/R (PLAN § 5b): N1 + N2, hysteresis t 0.5, **không** lọc FP. Giả thuyết: nếu test thưa annotator hơn train thì extent hẹp (0.5) ≥ rộng (0.3) | – (t 0.4 OOF cross-tune 0.4284) | 0.37 |
| 2026-11-05 08:35 | `submission_ens_n1n2_t0.4.csv` | như trên, t 0.4 (final 1 bỏ lọc FP; đo riêng phần lọc FP so với dòng 04/11) | – | 0.37 |
| 2026-11-05 08:35 | `submission_ens_n1n2_t0.3.csv` | như trên, t 0.3 | – | 0.37 |
| 2026-11-04 09:27 | `submission_ens_n1n2.csv` | **final 1 ứng viên**: N1 ×5 @1024 + N2 ×5 @2048 (1:1) + hysteresis (t 0.4, seed 0.99, area 200, r 12) + lọc FP | 0.431 (OOF, in-sample); cross-tune +0.0105 so với N1 | 0.37 |
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

### 2026-11-10

- **Weights vào repo** (người dùng chọn fp32, giữ trùng byte): 15 checkpoint + `SHA256SUMS` + `weights/README.md` trong `weights/`, đẩy lên GitHub theo 3 commit (mỗi family ~400 MB). GitHub chỉ cảnh báo file > 50 MB. Clone sạch `--depth 1` khớp 15/15 hash.

### 2026-11-09

- **Gate gộp 707 ảnh** (đăng ký trước 05/11): (a) N1 + N2 + BCE + N9' hơn N1 + N2 +0.0065 [+0.0037, +0.0092]; (b) BCE biên +0.0009, P 0.77 < 0.8 → bỏ; kiểm lại N1 + N2 + N9' +0.0056 [+0.0034, +0.0077], P 1.00; (c) 5/5 fold. **Final 1 = N1 + N2 + N9'** (`configs/fpfilter_ens_n1n2n9p_orient.pkl`), LB 0.38. Final 2 tạm giữ bước trước đó (N1 + N2, LB 0.37), xét lại khi N11/N12 về. Việc tái lập: thêm 5 checkpoint N9' vào `filseg-final-weights` và `kaggle/make_infer_kernel.py`.
- Nộp 23:40 UTC 08/11 (2 lượt cuối ngày): pool N1 + N2 + N9' **0.38**, N9' đơn **0.38**. Lọc FP trên N9' đơn được +0.013 OOF.
- Đăng ký trước khi chạy: trọng số N9' 1:1:2 chỉ dùng nếu CI 95% > 0 so với 1:1:1 (707 OOF, cross-tune); ngưỡng hysteresis 0.55–0.6 chỉ dùng nếu OOF in-sample tăng ≥ 0.001 so với t 0.5. Kết quả: 1:1:2 −0.0007 [−0.0025, +0.0010]; t 0.55 −0.0003, 0.6 −0.0017 → **không qua cả hai**, final 1 giữ nguyên.
- N11 fold 0 (N9' seed 2): đơn 0.4119 (N9' 0.4200); N9' + N11 +0.0046 so với N9'; nhưng pool 4 family chỉ +0.0002 [−0.0031, +0.0033] so với final 1 → biên lợi từ seed thứ hai nhỏ khi đã có N1 + N2. Fold 1 đang so; gate 707 khi đủ 5 fold.
- N11 fold 1: đơn 0.4046 (N9' 0.3963); pool 4 family +0.0022 [−0.0022, +0.0064]; seed-avg N1 + N2 + (N9'+N11)/2 +0.0071 [+0.0022, +0.0120].
- Đăng ký trước fold 2–4 và gate 707 (09/11 07:25 UTC): phương án chính vẫn theo thẻ PLAN, N11 là family thứ tư trọng số bằng nhau, gate § 5b. Seed-avg (N9' + N11 thành một family 10 model) đã nhìn số ở 2 fold nên chỉ là phương án phụ: dùng thay phương án chính chỉ khi trên 707 ảnh nó qua (a) so với N1 + N2 + N9' **và** hơn phương án chính với CI 95% > 0. Còn không thì báo là kết quả khám phá.
- **Gate N11 707 ảnh: không qua.** 4 family −0.0026 [−0.0047, −0.0005], seed-avg −0.0024 [−0.0042, −0.0006] so với N1 + N2 + N9'. Seed thứ hai của cùng công thức không giúp pool; N11 bỏ. N12 (R34) 5 fold đã về, đang so từng fold.
- **Gate N12 (R34, công thức N9'): không qua.** (c) qua 5/5 fold; trung bình pool từng fold +0.0016 nhưng (a) trên 707 ảnh −0.0004 [−0.0021, +0.0013], P 0.33. Bài học: thêm family thứ tư vào N1 + N2 + N9' không còn giúp; trung bình từng fold (mỗi fold tune riêng) lạc quan hơn gate 707 (một bộ hậu xử lý chung). **Final 1 giữ N1 + N2 + N9'.** Ổ đĩa đầy giữa lúc dựng ensemble cho (b); đã xoá ensemble dẫn xuất của N11, w112 và (b) (tạo lại được).
- **Tái lập v2** (người dùng: "làm tái lập ngay"): `filseg-final-weights` thêm 5 checkpoint N9' (15 file, `SHA256SUMS`; hash N1/N2 không đổi). `kaggle/make_infer_kernel.py` dựng final 1 = N1 + N2 + N9' và final 2 = N1 + N2 (final 1 cũ). Kernel T4 ×2: map 90.4 phút. Final 2 trùng từng byte; final 1 lệch vì map N9' gốc làm trên Colab, kernel CSV đạt LB **0.38** → chọn CSV kernel làm final 1 (tái lập được từ notebook).
- Nộp 09/11: N9' không lọc FP **0.38** (bằng bản lọc), N12 đơn **0.36**, final 1 từ kernel **0.38**. Lượt 4–5 (23:47 UTC, người dùng yêu cầu dùng hết): final 1 không lọc FP **0.38**, final 2 không lọc FP **0.37**, nên bộ lọc FP không đổi LB ở cả hai final.
- Người dùng: "tiếp tục improve". Hàng đợi Colab mới N11 (`effv2s_c512_s2_f*`) → N12 (`r34_c512_f*`), chạy từ 00:02 UTC. Từ nay làm thẳng trên `main` trong thư mục project (không dùng worktree).

### 2026-11-08

- **Notebook end-to-end** `notebooks/final_inference.ipynb` (dựng bằng `kaggle/make_infer_kernel.py`):
  10 checkpoint → map TTA 8 → ensemble → hậu xử lý + lọc FP → cả hai final, chỉ dùng ảnh H-alpha test.
  Chạy trên kernel Kaggle `filseg-final-infer` (2 × T4, torch 2.11, smp 0.5.0, weights qua dataset riêng
  `filseg-final-weights`): map xong trong 75.5 phút. **Cả 1800 map (10 model × 180 ảnh) và 4 map ensemble
  trùng từng byte** với map đã dùng để nộp (map tạo trên Colab), **hai CSV trùng từng byte** với
  `submission_ens_n1n2_orient.csv` và `submission_n1up.csv` → tiêu chí tái lập đạt ở mức mạnh nhất.
  Log và hash: `docs/logs/final/`. Nộp hai CSV của kernel (loại 1).
- **Ablation mode cc** (loại 3): bỏ hysteresis, OOF −0.0248 [−0.0290, −0.0204] (0.4313 → 0.4065, FP 2284 → 4022);
  LB **0.36** so với 0.37, khớp dự đoán 0.34–0.36. Log `docs/logs/research06/ablation_cc_oof.txt`.
- **Colab**: Drive đổi sang `MyDrive/kaggle/filament-segmentation-2026/`; hàng đợi chỉ N9' (5 fold), bs tự
  giảm trên T4; thiếu zip trên Drive thì notebook lấy code từ dataset `filseg-code`. N9' fold 0 bắt đầu
  08:26 UTC. Nhánh `exp/kaggle-gpu` đã merge vào `main` (người dùng đồng ý).
- Watcher Colab cũ (8 run đã về hết) đã dừng.
- **N9' fold 0 qua gate** (đăng ký trước 06/11): đơn 0.4200 so với N1 0.4158 (ngưỡng 0.4163); N1 + N2 + N9' hơn N1 + N2 +0.0068 [+0.0028, +0.0111] (ngưỡng +0.005). Upload 1.3 GB từ Colab lỗi 400 lúc tạo dataset; upload riêng map val (568 MB) qua bản version thì được. Dataset fold 1–4 tạo sẵn từ máy local để Colab chỉ thêm version. Fold 1–4 chạy tiếp (~6 h).
- **N9' fold 1 không giúp**: đơn 0.3963 so với N1 0.4090 (−0.0126 [−0.0248, −0.0011]; nhiều FP: 861 so với 517); N1 + N2 + N9' so với N1 + N2 −0.0008 [−0.0063, +0.0049]. Gộp fold 0–1 chênh pool ~+0.003, gate gộp 707 ảnh (CI > 0) giờ khó qua. Map val fold 0 bản Colab tải lại trùng từng byte. Fold 2–4 vẫn chạy (xong ~16:30 UTC).

### 2026-11-07

- **Nộp 5 bài số cho report** (hết lượt ngày, tự động lúc 00:02–00:04 UTC, không lỗi mạng): N1 fold 1/2/3/4
  đơn đều **0.36** (fold 0 trước đây cũng 0.36); N2 5 fold đơn **0.37**. Cả năm khớp dự đoán đăng ký trước.
  Kết luận cho report: một fold N1 đơn ≈ 0.36 ổn định trên cả 5 fold; ensemble 5 fold hoặc N2 lên 0.37;
  final 1 (N1 + N2) cũng 0.37, nên trên LB 2 chữ số không tách được ensemble với N2 đơn (chênh OOF ~0.01–0.02
  nằm dưới độ phân giải ~0.02). Chọn final vẫn theo OOF.
- Chờ người dùng tải bundle `build/colab/filseg-colab.zip` lên Drive và chạy N9' fold 0 (gate PLAN § 5b).

### 2026-11-06
- **Nộp 5 bài** (hết lượt ngày). Bài final 1 lúc 00:02 UTC lỗi mạng (không phân giải được
  api.kaggle.com), nộp lại lúc 02:07. Final 1 + gộp theo hướng **0.37**, final 2 N1 UP2048 **0.37**:
  cả hai bằng bài cũ, đúng dự báo (Δ OOF +0.002/+0.005 dưới độ phân giải LB).
- **Thang suy giảm** (bỏ 15/30/50% instance điểm lọc thấp nhất): LB **0.35 / 0.33 / 0.27** so với 0.37,
  OOF +0.005 / −0.019 / −0.085. Từ 30% trở đi LB giảm song song với OOF (độ dốc ≈ 1: −0.04 → −0.10 so
  với −0.019 → −0.085, lệch ~−0.02). Riêng 15% OOF dương mà LB **−0.02**: bậc +0.005 là in-sample (bộ
  lọc fit trên chính OOF). *(Sửa 06/11, F59: chấm bằng điểm lọc cross-fit thì OOF là −0.008 / −0.035 /
  −0.093, khớp LB trong nhiễu + làm tròn; câu cũ "trên test instance điểm thấp đúng nhiều hơn" rút lại.)*
  Giữ ngưỡng lọc hiện tại. LB phân giải được bước ~0.02, không
  phân giải được 0.005 — đúng mô hình làm tròn ở `kaggle_endgame.md`.

### 2026-11-05
- Kernel BCE fold 2–4 về 09:26 (~4.5 h phiên GPU, tổng ~20.8 h). **Gate N7 không qua**: trên 707 ảnh
  OOF, N1 + N2 + BCE +0.0018 [−0.0008, +0.0045] so với N1 + N2 (0.4302 so với 0.4284). Final 1 giữ
  10 model; bài 15 model nộp một lần làm điểm LB.
- **Nghiên cứu đa agent** (RESEARCH § 13, `docs/logs/research05/`): phân rã lỗi OOF (biên 36%, FN 38%,
  FP 26% của 1 − PQ; inter-annotator PQ 0.34 so với model 0.43); spine/bbox/area gần như không có thông
  tin mới so với mask; paper trong ngành không có cách tách instance nào hơn heuristic, Hao & Li 2026
  đo trên MAGFiLO thấy loss skeleton/component không giúp; phương pháp vật mảnh (Omnipose, mutex
  watershed, Skeleton Recall) đều cần GPU và dự kiến < ngưỡng phát hiện. Hậu xử lý rẻ: chỉnh biên theo
  cường độ **âm** (−0.001 đến −0.05), ngưỡng theo kích thước, gộp theo hướng, làm mượt ≈ 0; gộp mảnh
  bằng GBM +0.0036 [−0.0001, +0.0071] (v2 đang chạy).
- **PLAN § 5b**: lịch 05–15/11, ngày nào nộp 5 bài gì, mỗi bài một giả thuyết; LB vẫn không quyết định.
- **Nộp 5 bài** (hết lượt ngày): 15 model, N1 + N2 ngưỡng 0.3/0.4/0.5 không lọc FP, không gộp mảnh:
  **tất cả 0.37**. Bỏ lọc FP không đổi LB; LB 2 chữ số không phân biệt hậu xử lý.
- **Gộp mảnh GBM không qua gate** (v1 +0.0036, CI chạm 0, 3/5 fold; v2 +0.0026). Giữ bán kính 12.
- **TTA theo tỉ lệ** (N1 f0 ở 1152): −0.0031; trung bình 1024 + 1152: −0.0002 → bỏ. Mọi map đã có
  TTA 8 hướng từ trước, nên mục "TTA" trong lịch 06/11 được thay bằng UP2048 và trọng số ensemble.
- **Trọng số N1:N2 trên 707 OOF**: 2:1 +0.0004 [−0.0014, +0.0022], 1:2 −0.0018 → giữ 1:1; bỏ hai bài
  thăm dò 2:1/1:2 khỏi lịch 06/11 (OOF đã trả lời, LB 2 chữ số không thấy được).
- **UP2048 cho final 2 qua F48**: N1 5 fold phóng map lên 2048 trước hậu xử lý: **+0.0050 [+0.0029,
  +0.0069]** trên 707 OOF (0.4179 → 0.4229). Final 2 = `submission_n1up.csv` (cấu hình hysteresis
  trùng final 1, chưa lọc FP). CSV cho 06/11 dựng sẵn: `n1up`, `n1n2_a120`, `n1n2_a300`.
- **Workflow nghiên cứu 06** (4 sweep; critic hai lần lỗi vì giới hạn model, phần phản biện làm trong
  phiên chính; `docs/logs/research06/`): tiền xử lý H-alpha là ngõ cụt; thêm seed (N8) dự kiến trượt
  gate; thành viên khác encoder (N9, EffNetV2-S) và độ phân giải 1536 (N10) xếp **trước** N8 trên Colab
  (bundle đóng gói lại); gate gộp đăng ký trước; lượt nộp chỉ cho tái lập / thang suy giảm / số report.
- **Gộp theo hướng qua F48**: +0.0021 [+0.0005, +0.0037] trên 707 OOF; sau lọc FP 0.4312 → 0.4327.
  Final 1 mới = `submission_ens_n1n2_orient.csv` (`PostParams.group_line`, có test).
- **Thang suy giảm** trên OOF: bỏ 15/30/50% → +0.005 / −0.019 / −0.085; 3 CSV test cho 06/11.
- **N8** (N1 seed 2, 5 fold) chuẩn bị cho Colab vì GPU Kaggle đã hết ("GPU hết thì qua Colab", người
  dùng 05/11): cờ `--seed`, queue trong notebook, bundle đóng gói.

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
- Nộp ensemble 10 model: **LB 0.37**, bằng N1 (LB chỉ hiển thị 2 chữ số, và khoảng ±0.01 là nhiễu theo
  `docs/logs/lb_noise_effb4_hyst.txt`). Không đổi quyết định: chọn theo OOF.
- **Rà soát lại các bước** (RESEARCH § 12, F45–F50): submission không có bug; khoảng thiếu nằm ở OOF
  (0.428 so với 0.45–0.46 của các đội 0.40). Hai lỗi phương pháp: map bị loss Dice làm bão hoà, và
  phóng nhãn nearest (hậu xử lý ở 2048 cho +0.0024 [+0.0003, +0.0045]).
- **N7 (BCE thuần, fold 0+1)**, kernel 7 xong 22:28 (~2.3 h GPU). Map hiệu chỉnh tốt hẳn (p ≈ tỉ lệ
  annotator), nhưng PQ đơn lẻ bằng N1 (−0.0002 / −0.0018). Làm thành viên thứ ba của ensemble: +0.0060
  (f0, CI > 0), +0.0038 (f1, CI chạm 0), nên chưa đạt tiêu chí đăng ký. Gộp 2 fold (ngoài đăng ký) cho
  +0.0041 [+0.0004, +0.0077]. Bộ lọc instance trên map BCE: +0.0004 / −0.0013, bỏ.
- Kết luận: bão hoà map là có thật nhưng **không phải** nguyên nhân giới hạn PQ. Train BCE fold 2–4
  (~3.5 h GPU, vượt ngân sách ~1 h) chỉ hứa hẹn khoảng +0.004 OOF; cần người dùng quyết.

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
