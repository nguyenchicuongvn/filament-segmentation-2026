# Research 07/11: lens `competition_structure`. Giải được quyết định bởi cái gì, và mình nên dồn sức vào đâu

Phạm vi: không train, không đụng repo (ngoài file này). Số mới đo ngày 06/11 bằng `.venv/bin/python -I`.
Script và output nằm ở
`/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/competition_structure/`
(`lb.csv`, `noise.py`, `shake.py`, `human.py`, `morph.py`). LB tải bằng
`kaggle competitions leaderboard --show --csv` (chỉ đọc, 1,404 đội, 06/11). Per-image stats lấy từ
`outputs/cmp/cache/*.pkl` (cột: ΣIoU, TP, FP, FN; combo tốt nhất in-sample): N1 0.4193, N2 0.4144,
ENS (N1 + N2) 0.4294, BCE15 0.4314, N1UP 0.4233.

## 0. Kết luận ngắn

1. **Đường PQ tới giải gần như bằng 0.** Bỏ hết tầng rò rỉ (≥ 0.49, 116 đội), mình 0.37 vẫn đứng
   sau 294 đội trung thực 0.38–0.42; mô phỏng shake-up cho hạng private trung vị **~364**, P(top 3
   theo PQ) = 0.0000. Ngay cả khi chỉ so với 8 đội trung thực có repo/report công khai, P(top 3 theo
   PQ) chỉ 0.05–0.9 %. Khoảng cách ~0.03 so với các đội 0.40 là thật: nhiễu ghép cặp trên 90 ảnh chỉ
   **0.005–0.007** (đo được), nên shake-up không xoá được.
2. **Nhưng giải không trao theo LB.** Top 3 do ban giám khảo chấm theo rubric (S19): 70 % định
   lượng (PQ, Dice, phân bố Dice/IoU, 1:n / n:1, **hiệu năng end-to-end**), 30 % định tính (mô tả
   pipeline, hình mask trên ảnh H-alpha, chất lượng code). Chỉ đội nộp đủ report 4 trang + repo
   public + notebook tái lập mới được xét. Đội leak và đội fork notebook công khai (0.40 của
   tushhaaarrrr và 0.39 của Dariush đều là notebook công khai) gần như chắc chắn **không** có hồ sơ
   này. Số đội cạnh tranh giải thực tế có lẽ chỉ 5–15. Mình đang thua họ về PQ, nhưng chỉ ~0.03,
   tức ~7–9 % điểm thành phần PQ.
3. **Chọn final không phải đòn bẩy.** E[max(private₁, private₂)] chỉ hơn final 1 **+0.0002** (với
   final 2 = N1UP) đến **+0.0028** (với BCE15). Muốn "variance-seeking" kéo lại 0.03 thì cần SD
   ghép cặp giữa hai final ~0.075, tức là gấp 10 lần mức hai model khác hẳn nhau. Bất khả.
4. Vì vậy, **hướng "đặc biệt" duy nhất trong lens này là tối đa hoá điểm rubric, không phải PQ**.
   Có ba thứ mình đo được và đối thủ gần như chắc chắn không có: (a) so sánh model với chuyên gia
   **trên cùng ảnh**, có CI (+0.090 [+0.077, +0.103] PQ); (b) mask **đúng giao thức gán nhãn
   MAGFiLO** (một mảnh): hiện 5.1 % instance trên test của mình bị nhiều mảnh, trong khi GT chỉ có
   0.4 %; (c) đường Pareto hiệu năng–PQ lấy thẳng từ OOF (OOF vốn là ensemble **2 model**).

## 1. Cấu trúc cuộc thi: ai thực sự là đối thủ

### 1.1 LB 06/11 (1,404 đội; 29/10 là 1,260)

| Tầng | Số đội | Ghi chú |
|---|---|---|
| ≥ 0.49 | 116 | rò rỉ (0.55 × 92 đội = payload S2; 0.56–0.70 = nhãn MAGFiLO đã chỉnh). Ảnh private cũng nằm trong MAGFiLO 1.0 (S1: cả 180 tên file), nên tầng này **cũng cao trên private**. Host dọn sau deadline (S6) |
| 0.42 | 1 | |
| 0.41 | 18 | 29/10 chỉ có 11; tầng trung thực vẫn đang đi lên |
| 0.40 | 55 | có `_.tushhaaarrrr` (notebook công khai 0.40); nhiều đội ở đây có lẽ nộp lại notebook đó |
| 0.39 | 129 | notebook Dariush 0.39 công khai |
| 0.38 | 91 | |
| **0.37** | 134 | **mình** (hạng LB thô ~411–544) |

Notebook công khai xếp theo điểm (`kaggle kernels list --sort-by scoreDescending`): các notebook
trung thực cao nhất là tushhaaarrrr 0.40 và Dariush 0.39. Phần lớn tầng 0.39–0.40 vì vậy là bản fork.

### 1.2 Mô phỏng shake-up (`shake.py`)

Mô hình: điểm thật của mỗi đội = điểm public (đều trong khoảng làm tròn ±0.005) + nhiễu riêng
N(0, σ); private = điểm thật + N(0, σ). Phần nhiễu chung giữa các đội (ảnh khó/dễ) triệt tiêu khi
xếp hạng. σ lấy từ SD ghép cặp đo trên OOF:

| Cặp (90 ảnh rút từ 707, 20k lần) | Δ trung bình | SD ghép cặp 90 ảnh | corr |
|---|---|---|---|
| N1 so với N2 (khác độ phân giải, cùng công thức) | −0.0050 | **0.0070** | 0.84 |
| N1 so với ENS | +0.0101 | 0.0049 | 0.93 |
| ENS so với BCE15 | +0.0019 | 0.0041 | 0.95 |
| ENS so với N1UP | −0.0062 | 0.0046 | 0.94 |
| SD một đội trên 90 ảnh | | 0.0122–0.0129 | (khớp F8 0.014) |

Hai đội có pipeline khác hẳn nhau (detector so với U-Net) có lẽ ở mức 0.007–0.010.

| Trường đối thủ | σ riêng | Hạng private của mình (trung vị / p5) | P(top 3 theo PQ) |
|---|---|---|---|
| Mọi đội < 0.43 (bỏ hết leak) | 0.0035 / 0.005 / 0.007 | 361 / 255 · 364 / 220 · 364 / 177 | 0.0000 |
| 8 đội có repo/report (RESEARCH § 6.2) | 0.005 / 0.007 | 8 / – | 0.0005 / 0.0087 |
| 8 đội trên + 5 đội 0.41–0.42 có hồ sơ | 0.005 / 0.007 | 13 | 0.0000 / 0.0001 |

P(mình vượt **một** đội cụ thể trên private), có tính cả làm tròn LB của hai bên:

| Khoảng cách thật | SD ghép cặp 0.007 | 0.010 |
|---|---|---|
| 0.020 | 0.041 | 0.095 |
| 0.025 | 0.015 | 0.051 |
| 0.030 | 0.004 | 0.025 |

Bằng chứng độc lập rằng khoảng cách ~0.02–0.03 là thật: OOF của họ 0.45–0.46 so với của mình 0.433
(cách đo khác nhau, nhưng cùng chiều); tỉ lệ LB/OOF của họ 0.87–0.89, của mình 0.855.

### 1.3 Rủi ro ngưỡng "PQ > 0.35"

Host nói sẽ xem kỹ mọi bài PQ > 0.35 (S8). Public và private là hai nửa của 180 ảnh, nên
SD(private − public) = **0.019** (đo trên ENS). Với public 0.37 đã làm tròn: P(private < 0.35) =
**0.15**, P(< 0.36) = 0.30, P(≥ 0.38) = 0.31. Hai final của mình tương quan 0.93–0.95, nên chọn final
khác đi cũng không giảm rủi ro này. Có lẽ đây không phải ngưỡng cứng, nhưng là thêm một lý do để
report phải tự đứng được, không dựa vào hạng.

## 2. Rubric: chỗ còn có thể thắng

Luật (PRIZE_RULES.md, S6b, S8, S19):

- top 3 do ban giám khảo chấm theo rubric, không theo LB;
- giải chia theo `3000 · s_i² / Σ s_j²`, nên điểm rubric là số liên tục;
- 70 % định lượng gồm PQ, Dice (`torchmetrics`), phân bố Dice, phân bố IoU, phân bố 1:n / n:1 và
  **hiệu năng tính toán end-to-end**; 30 % định tính gồm mô tả pipeline, hình thái mask trên ảnh
  H-alpha và chất lượng code;
- S6b: giám khảo xem mask trên những ví dụ họ coi là quan trọng; tính khả thi (thời gian chạy
  vừa phải, không phức tạp hoá) được điểm cao.

**Tính hoà vốn (giả định của mình, host không công bố cách quy đổi)**: nếu mỗi thành phần định
lượng được chấm theo tỉ lệ so với đội tốt nhất, thì PQ private 0.37 so với 0.40–0.41 cho
0.90–0.925; Dice và các phân bố nhạy kém hơn PQ, khoảng 0.95–0.98. Tổng lại mình thiếu ~3–6 điểm trên
70. Muốn bù, phần định tính phải hơn đội đầu **10–20 %** (3–6 trên 30). Việc này khả thi nếu report của
họ chỉ có bảng PQ (README của theainerd, umerkang66, zhang0894, BSergey2000 hiện đúng như vậy), nhưng
khó nếu gặp biconcavelens (họ có cả paper LaTeX đầy đủ, `paper/main.tex`). Nếu host chấm theo hạng
thay vì tỉ lệ thì khoảng cách định lượng lớn hơn, và cơ hội của mình thấp.

**Hệ quả**: mỗi giờ Mac còn lại đáng giá hơn khi đổ vào các hạng mục rubric mà đối thủ bỏ trống,
chứ không phải vào +0.002 PQ (vốn vô hình cả trên LB lẫn trên điểm rubric).

## 3. Ý tưởng

### I1. Đổi mục tiêu endgame: "rubric-max", không phải "PQ-max" (chiến lược, đo được)

- **Cơ chế**: hạng PQ không đổi được (§ 1.2), nhưng điểm rubric thì còn đổi được. Điểm định tính
  (30 %) và các thành phần định lượng không phải PQ (Dice, phân bố, 1:n / n:1, hiệu năng) là những
  chỗ đối thủ fork và đối thủ chỉ có README không có gì.
- **Việc cụ thể** (Mac, ~1.5 ngày):
  - bảng rubric đủ 6 mục cho final, có CI theo ảnh;
  - histogram Dice/IoU theo từng bản gán nhãn;
  - 1:n / n:1 so với chính mức của người (chấm annotator theo annotator);
  - thời gian và bộ nhớ đo trên kernel Kaggle;
  - một lưới hình mask có chủ đích: filament lớn có barb, filament gần limb, filament đứt, ảnh mờ,
    ca thất bại.
- **Quy tắc GPU**: hai run Colab còn lại chỉ đáng chạy nếu dự báo ΔOOF ≥ +0.01, vì dưới mức đó
  không đổi được hạng PQ lẫn điểm rubric.
- ΔOOF 0. ΔP(giải): lớn nhất trong mọi đòn bẩy còn lại (định tính, không đo được).

### I2. "Vượt chuyên gia trên cùng ảnh": headline của report (đo được)

`human.py`, 296 ảnh có ≥ 2 annotator, chấm mọi cặp có thứ tự annotator A theo B (đúng F6) và model
OOF (ENS, chưa từng thấy nhãn của ảnh) theo từng annotator:

| | PQ | SQ | RQ |
|---|---|---|---|
| Người so với người | 0.340 | 0.634 | 0.536 |
| Model ENS so với từng người | **0.430** | 0.674 | 0.637 |
| Chênh (bootstrap theo ảnh, 2,000 lần) | **+0.090 [+0.077, +0.103]** | +0.040 | +0.101 |

Kết quả ổn định theo số annotator: ảnh 2 người 0.441, ảnh 3 người 0.423, ảnh 1 người 0.429.

- **Cơ chế cho điểm**: giám khảo là nhóm đã tạo MAGFiLO. Câu "model đồng thuận với từng chuyên gia
  nhiều hơn các chuyên gia đồng thuận với nhau, CI 95 % loại trừ 0, ở cả biên (SQ) lẫn phát hiện
  (RQ)" đáng giá hơn mọi bảng PQ. Nó cũng giải thích luôn vì sao LB thấp (trần do bất đồng
  annotator, H5).
- **Mới so với RESEARCH**: F6 và F53 đã có hai con số 0.340 và 0.430. Phần mới là so trên cùng ảnh,
  có CI, tách SQ/RQ, tách theo số annotator. Bước tiếp: mỗi nhóm annotator 01–06 xếp model vào
  "hạng" giữa 37 người.
- **Chú ý trung thực**: hậu xử lý được tune in-sample trên chính 707 ảnh này (lạc quan ~0.002), và
  model học từ "trung bình" các annotator, nên phải nói đây là đồng thuận chứ không phải "đúng hơn".

### I3. Mask đúng giao thức MAGFiLO: một mảnh (mới, đo được)

`morph.py` giải RLE của 964 instance final 1 trên test và 5,052 instance GT (600 bản gán nhãn):

| | Instance nhiều mảnh | Instance có lỗ | Pixel lỗ / diện tích |
|---|---|---|---|
| GT train | **0.4 %** | 11.4 % (lỗ 1–2 px do raster polygon) | 0.0001 |
| Final 1 (`ens_n1n2_orient`) | **5.1 %** | 1.5 % | 0.0007 |
| Final 2 (`n1up`) | 5.9 % | 1.8 % | 0.0006 |

- **Cơ chế**: S14 nói mask MAGFiLO phải là **một mảnh, không lỗ**. Gộp theo bán kính hoặc hướng
  (r 12, L16 r6) tạo ra một instance gồm nhiều mảnh rời nhau: ~50 instance trên test, gấp ~13 lần
  tỉ lệ của GT. Khi giám khảo "xem trực tiếp mask" (S6b), một filament hiện thành nhiều đốm cùng
  màu là lỗi nhìn thấy ngay. Sửa: nối các mảnh của cùng instance bằng một cầu mảnh (đoạn thẳng giữa
  hai điểm gần nhất, rộng bằng bề rộng cục bộ hoặc theo vùng p > t thấp hơn), sau đó lấp lỗ.
- **PQ**: GT vẽ liền qua chỗ đứt, nên cầu nối thêm pixel đúng. IoU của TP tăng nhẹ, số TP/FP không
  đổi. Kỳ vọng ΔOOF **0 đến +0.001**. Lấp lỗ đã đo ≈ 0 (F56 −0.0003). Lợi chính ở điểm định tính
  và ở bảng 1:n / n:1.
- **Mới**: RESEARCH chưa đo tính một mảnh của instance dự đoán; F23 và F56 chỉ đo lấp lỗ.

### I4. Chọn final và Pareto hiệu năng (đo được, chỉnh nhẹ PLAN)

- **Final**: E[max] − E[final 1] = +0.0002 (final 2 = N1UP), +0.0028 (BCE15, gồm +0.0019 trung bình),
  +0.0010 (N2 so với N1). Kết luận: chọn final theo tiêu chí **tái lập và giải thích được**, không
  theo phòng hộ. Final 1 = ENS orient. Final 2 = BCE15 chỉ khi notebook chạy được nó không tốn thêm
  công; nếu không thì giữ N1UP. Cả hai phải chạy được trong notebook, vì Kaggle lấy điểm private cao
  hơn trong hai bài, và giám khảo có thể đòi tái lập đúng bài đó.
- **Hiệu năng là một mục định lượng**: OOF của ENS vốn là ensemble **2 model** (mỗi ảnh chỉ được 1
  model N1 + 1 model N2 không thấy nó dự đoán), tức 0.4294 là của cấu hình rẻ hơn 5 lần cấu hình 10
  model nộp lên test. N1 OOF (1 model) là 0.4193. Trên test: 1 fold 0.36, 5 fold 0.37. Report nên có
  một đường Pareto (1 / 2 / 10 model × TTA 1 / 8, giây mỗi ảnh trên T4) cùng một "chế độ lite"
  trong notebook. Đối thủ dùng 12 nguồn detector (biconcavelens) khó thắng mình ở mục này.
- ΔOOF private kỳ vọng 0 đến +0.003.

## 4. Báo cáo thắng giải cần có gì (checklist rút từ rubric)

1. Bảng 6 mục rubric (PQ, Dice torchmetrics, histogram Dice, histogram IoU, 1:n / n:1, thời gian
   và bộ nhớ end-to-end) cho cả OOF 707 ảnh (có CI) lẫn test (LB public).
2. Dòng "người so với người" ngay trong bảng đó (I2), đặt cạnh model.
3. Lưới hình: 6–8 ảnh, mỗi ảnh có GT của từng annotator và dự đoán; có ca thắng và ca thua; barb;
   filament đứt được nối một mảnh (I3).
4. Ablation chỉ trên OOF, có CI: G4 N1 + N2 +0.0105, hysteresis, lọc FP, gộp theo hướng. Kèm đoạn
   "LB 90 ảnh không phân giải" (research06 § 1.2).
5. Một câu tuyên bố trung thực: không dùng MAGFiLO 1.0 công khai, không dùng frame lân cận, mask rời
   nhau có assert. Host đang dọn leak, nên đây là điểm cộng.
6. Dependencies đều permissive (smp MIT, timm Apache-2.0, pycocotools BSD). Luật đòi code thắng
   giải phải theo license OSI không hạn chế thương mại. Đối thủ dùng ultralytics (AGPL-3.0) có rủi
   ro về license, mình thì không. Chỉ nêu sự thật về license của mình, không bình luận về người khác.

## 5. Những hướng đã kiểm và thấy không đáng làm

| Hướng | Số đo | Kết luận |
|---|---|---|
| Chọn final 2 "liều" để tăng phương sai | E[max] tối đa +0.0028; để bù 0.03 cần SD ghép cặp ~0.075 | chết |
| Chờ host loại leak để lên hạng | loại leak: hạng ~295–428 trong đội trung thực | không giúp |
| Shake-up private | P(vượt một đội 0.40) 0.4–2.5 % | không giúp |
| Mong đội 0.38–0.42 bị loại vì tái lập | phần lớn không nộp hồ sơ, nhưng 5–8 đội có repo vẫn hơn mình về PQ | chỉ thu hẹp trường đối thủ |
