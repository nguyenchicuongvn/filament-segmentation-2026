# Research: LB, đối thủ, tài liệu (2026-10-25)

Tổng hợp từ notebook/discussion public của cuộc thi, repo GitHub của các đội khác, paper MAGFiLO
và các nghiên cứu liên quan. Phần **Sự thật** có nguồn; phần **Đề xuất** là suy luận của mình.

## 1. Leaderboard bị nhiễm, mục tiêu thật là ~0.40 public

**Sự thật**
- **Rò rỉ nhãn test**: 139/180 ảnh test (77%) trùng byte với ảnh trong bộ MAGFiLO 1.0 công khai
  (kèm nhãn), theo một số nguồn là cả 180. Một người thử thay dự đoán bằng nhãn MAGFiLO thì điểm tăng
  0.73 → 0.82 (metric cũ). Discussion
  [728062](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062).
- **Payload**: notebook "Solar Filament Unet Segmentation | 0.55+" không có model, chỉ giải nén một
  submission cứng (`CHAMPION_PAYLOAD`). LB có **76 đội đúng 0.55**, và hầu như trống trong khoảng
  0.42–0.54.
- **Mask trùng lặp**: metric đếm mọi cặp IoU > 0.5, nên nộp mỗi mask 2 lần sẽ được điểm cao hơn
  (OOF 0.4385 → 0.4831, umerkang66). Host nói không chấp nhận mask chồng nhau
  ([735377](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735377)).
- **Mức đồng thuận giữa hai chuyên gia**: một annotator chấm theo annotator khác chỉ được **PQ 0.340**
  (SQ 0.634, RQ 0.536); filament < 200 px chỉ được 0.137
  ([notebook Dariush Afshar](https://www.kaggle.com/code/dariushafshar/filament-pq-two-experts-agree-at-0-34)).
- **Nhiễu LB**: SD ≈ 0.014 trên nửa 90 ảnh, tức ±0.03 (95%). Trên LB ngày 12/10: trung vị 0.34,
  p75 0.37, p90 0.40.
- **Các đội trung thực tốt nhất có công bố**:

| Đội / repo | OOF PQ | Public LB |
|---|---|---|
| [biconcavelens](https://github.com/biconcavelens/filament-segmentation-2026) | 0.461 | 0.40 |
| [theainerd](https://github.com/theainerd/filament_segmentation) | 0.452 | 0.40 |
| [Dariush Afshar](https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39) | 0.450 | 0.39 |
| [umerkang66](https://github.com/umerkang66/solar-filament-segmentation/blob/main/REPORT.md) | 0.448 | 0.39 |
| [zhang0894](https://github.com/zhang0894/Solar-Filament-Segmentation-Challenge-2026-by-Anon-Tokyo) | 0.443 | 0.38 |
| **Mình** (ensemble 768 + 1024 + lọc FP) | 0.399 (holdout) | 0.34 |

- Mọi đội đều thấy LB thấp hơn val/OOF khoảng 0.05–0.06. umerkang66 đã loại trừ các nguyên nhân do
  ngẫu nhiên, độ dày mask và dịch ảnh; họ nghiêng về giải thích là lệch giữa các annotator.
- **Host (Remarks, 24/10)**: chấm điểm không dựa vào LB; LB sẽ được dọn sau deadline; host nhận ra
  được các "shortcut"; **weights phải nằm trong repo và được notebook load**, nếu không sẽ không được
  xét giải; không khuyến khích report do AI viết mà thiếu kiểm soát
  ([724503](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503)).

**Kết luận**: khoảng cách thật giữa mình và nhóm trung thực đầu bảng là **0.34 → ~0.40 public**
(0.40 → ~0.46 OOF), không phải 0.34 → 0.69.

## 2. Mình đang mất PQ ở đâu (ensemble 768 + 1024, val, không lọc FP)

PQ 0.394 · SQ 0.664 · TP 1,363 · FP 861 · FN 1,003.

| Loại lỗi | Số lượng | Trần PQ nếu sửa hết |
|---|---|---|
| FN gần trúng (0.3 < IoU ≤ 0.5): sai biên | 370 | 0.483 |
| FN không có dự đoán nào chạm: phần lớn < 800 px (313) | 461 | 0.480 |
| FP (một nửa là gần trúng 0.3–0.5, tức cùng lỗi sai biên) | 861 | 0.485 (bỏ hết FP) |
| Nâng SQ từ 0.664 lên 0.70 | – | 0.416 |

Lỗi biên (gần trúng) chiếm phần lớn: cùng một filament nhưng mask của mình lệch so với cách annotator
vẽ. Kết quả này khớp với umerkang66: 56% PQ mất do biên, 15% do bỏ sót, chỉ ~9% do tách/gộp.

## 3. Paper MAGFiLO (Sci. Data 2024) — cách gán nhãn

[PMC11437148](https://pmc.ncbi.nlm.nih.gov/articles/PMC11437148/)
- Vẽ bằng brush/eraser (V7 Darwin), lưu dạng polygon. Mask phải bao **mọi barb nổi bật**, là **một
  mảnh liền, không lỗ**, ôm sát thân filament.
- Tối đa 3 annotator làm độc lập. Các mảnh có chirality mâu thuẫn được tách thành filament riêng, nên
  annotator có thể khác nhau về chuyện gộp hay tách.
- **Bỏ qua**: filament quá nhỏ hoặc mờ, cụm filament nhỏ nằm sát nhau, và filament cách kinh tuyến
  trung tâm hơn ~±70°.
- Không có baseline ML.

## 4. Kỹ thuật đã được chứng minh (xếp theo tác động dự kiến)

| # | Kỹ thuật | Bằng chứng | Ước tính cho mình |
|---|---|---|---|
| A | **Target mềm đa annotator**: pixel = tỉ lệ annotator của ảnh vẽ pixel đó; lấy mẫu ảnh theo số bản gán nhãn | umerkang66 | +0.01–0.03 |
| B | **Độ phân giải gần gốc** (1536–2048, train trên crop, infer full ảnh) + TTA 8 hướng | mọi đội top dùng 2048 | +0.01–0.03 |
| C | **K-fold OOF + ensemble** nhiều encoder (R34, EffNet-B4/EffV2-S, RegNetY) | Dariush 10 model; umer, theainerd | +0.02–0.03, chọn tham số ổn định hơn |
| D | **Tách instance**: seed (p > t_seed) → gộp seed gần nhau ~16 px → watershed tới p > t_mask → **lấp lỗ** (GT không có lỗ) | umerkang66: gộp seed +0.008 | +0.005–0.01 |
| E | **Chọn instance theo "luật PQ/2"**: regressor dự đoán IoU kỳ vọng của mỗi ứng viên, giữ nếu > PQ/2 (~0.2); thêm đặc trưng khoảng cách tới rìa | umerkang66, theainerd (GBM, τ 0.25) | +0.005 (nâng cấp bộ lọc FP hiện tại) |
| F | Head phụ skeleton/core + blob loss | umerkang66: +0.009 (CI dương) | +0.01 |
| G | Thành viên khác kiểu: detector (Mask R-CNN / YOLO-seg) + U-Net tinh chỉnh crop, ensemble có hiệu chỉnh | biconcavelens: 0.40 → 0.461 OOF | +0.02–0.05, tốn công |

**Đã có người thử và thất bại**: clDice, CLAHE, pseudo-label trên test (0.417 → 0.393), pretrain
SimSiam trên ảnh GONG không nhãn, Mask2Former (≤ 0.345), SAM với box prompt (0.316), closing/dilate
1 px cho từng instance, chỉ train trên một annotator.

## 5. Không được làm

- Dùng bộ MAGFiLO 1.0 công khai dưới bất kỳ hình thức nào (kể cả làm ảnh pretrain không nhãn), vì nó
  chứa ảnh test.
- Nộp mask chồng nhau hoặc trùng lặp. `make_submission.py` cần assert mọi mask rời nhau.
- Tune theo LB public (nhiễu ±0.03, và LB lệch so với val).
