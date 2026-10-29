# Research: bài toán, dữ liệu, đối thủ, giả thuyết (bản 2026-10-29)

Tài liệu này là **cơ sở bằng chứng** cho [PLAN.md](PLAN.md). Quy ước:

| Mã | Nghĩa | Mức tin cậy |
|---|---|---|
| `S*` | Nguồn ngoài: discussion/notebook Kaggle, repo GitHub, paper. Kèm link, ngày đọc, số liệu như trong nguồn | A: đọc trực tiếp, có số. B: chỉ đọc lại từ transcript cũ hoặc nguồn thứ cấp |
| `F*` | Số đo của chính mình, kèm script và log để tái lập | cao (đo trực tiếp), trừ khi ghi chú khác |
| `H*` | Giả thuyết hoặc suy luận: nêu bằng chứng ủng hộ, phản đối, và cách kiểm chứng | – |

Không lấy số từ trí nhớ. Mọi số trong PLAN đều phải trỏ về một mã ở đây.

## 1. Bài toán và metric

| Mã | Nhận định | Bằng chứng |
|---|---|---|
| F1 | 707 ảnh train / 1,154 bản gán nhãn (annotator-image) / 8,199 filament; 411 ảnh có 1 annotator, 145 ảnh có 2, 151 ảnh có 3; 180 ảnh test; ảnh 2048², GONG H-alpha | [DATA_AND_METRIC.md](DATA_AND_METRIC.md), EDA 22/10 |
| F2 | Diện tích filament (px ở 2048²): p10 410, trung vị 1,228, p90 4,685 | như trên |
| F3 | Metric = PQ, khớp IoU > 0.5; **mỗi bản gán nhãn chấm riêng** với toàn bộ dự đoán của ảnh; TP/FP/FN cộng dồn trên cả tập. Ảnh có 3 annotator nặng gấp 3 | `src/filseg/metric.py`, bám theo notebook self-evaluation của host |
| F4 | Hệ quả của F3: một dự đoán chỉ khớp 1 trong 3 annotator thì vẫn là FP với 2 người kia. Không thể đạt PQ cao khi các annotator bất đồng | suy ra từ F3 |
| F5 | 37 annotator; id dạng `<annotator>-<stem>`; hai chữ số đầu của annotator là **nhóm** (01–06). Nhóm 01 chiếm ~45% số bản gán nhãn | `scripts/analyze_errors.py` |
| S14 | Cách gán nhãn MAGFiLO: vẽ bằng brush/eraser trong V7 Darwin, sau khi có bản nháp tự động; mask phải bao mọi barb nổi bật, là **một mảnh, không lỗ**; tối đa 3 annotator độc lập; các mảnh có chirality mâu thuẫn thường thuộc filament khác nhau. **Bỏ qua**: filament quá nhỏ hoặc mờ, cụm filament nhỏ nằm sát nhau, filament cách kinh tuyến trung tâm hơn ±70° | [Paper Sci. Data 11:1031 (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11437148/), các mục Annotation Platform, Definitions. A |

## 2. Người với người: trần thật của bài toán

| Mã | Nhận định | Bằng chứng |
|---|---|---|
| F6 | Chấm một annotator theo annotator khác (mọi ảnh train nhiều annotator): **PQ 0.340**, SQ 0.634, TP 4,292, FP = FN = 3,709 | `docs/logs/critique2_errors_effb4_hyst.txt` (khối "human agreement") |
| F7 | Annotator nhóm 01 được 0.28–0.36 khi đóng vai "dự đoán"; nhóm 02/05 được 0.40–0.44 | như trên |
| S4 | Dariush Afshar tính độc lập ra đúng các số đó: PQ 0.3399 (SQ 0.6337, RQ 0.5364; TP 4,292, FP 3,709, FN 3,709) trên 296 ảnh. Filament < 200 px chỉ được PQ 0.137, nhưng bin này chỉ có 58 phép so | [notebook](https://www.kaggle.com/code/dariushafshar/filament-pq-two-experts-agree-at-0-34), ~13/10. A |

**Ý nghĩa**: model của mình (0.41 trên val) đã **vượt** mức đồng thuận giữa người với người. Phần
còn lại chủ yếu là học "kiểu trung bình" của các annotator, chứ không phải nhận ra filament tốt hơn.

## 3. Leaderboard và khoảng cách val ↔ LB

| Mã | Nhận định | Bằng chứng |
|---|---|---|
| S1 | **Rò rỉ nhãn test**: 139/180 ảnh test (77.2%) trùng từng byte với ảnh trong MAGFiLO 1.0 công khai. Một comment nói cả 180 tên file đều có trong MAGFiLO (kiểm theo metadata). Thay dự đoán bằng nhãn MAGFiLO thì điểm tăng 0.73 → 0.82. Bài đăng trước đợt chấm lại ngày 12/8, nên có lẽ là metric cũ (suy luận của mình) | [discussion 728062](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062), ~cuối 7/2026. A |
| S2 | **Payload**: notebook "Solar Filament Unet Segmentation 0.55+" giải mã `CHAMPION_PAYLOAD` (lzma + base85) thành 1,342 dòng RLE rồi ghi thẳng vào submission. Có train một U-Net nhỏ cho đẹp nhưng không dùng. LB ngày 29/10: **82 đội đúng 0.55** / 1,260 | [notebook lamhuy8904](https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55); LB tải bằng CLI 29/10 08:16 UTC. A |
| S3 | **Mask chồng nhau**: nộp mỗi mask 2 lần làm PQ tăng 0.4385 → 0.4831 (umerkang66). Host: "we do not accept overlapping predictions"; Kaggle đã trả lỗi cho một bài như vậy | [umerkang66 REPORT L62](https://github.com/umerkang66/solar-filament-segmentation/blob/master/REPORT.md?plain=1#L62), [discussion 735377](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735377). A |
| S16 | **LB 29/10** (1,260 đội): tầng rò rỉ 0.70 / 0.64 / 0.62 (4 đội) / 0.60 (4 đội) / 0.55 (82 đội). Tầng trung thực: 0.42 (1), 0.41 (11), 0.40 (46), 0.39 (100), 0.38 (79), 0.37 (111), **0.36 (100, có mình, hạng 483)** | CLI `kaggle competitions leaderboard --download`, 29/10. A |
| F8 | **Nhiễu lấy mẫu của LB**: PQ trên 90 ảnh (cỡ public LB) có SD **0.014** (bootstrap), tức ±0.027 (95%). Trên 180 ảnh SD 0.010 | `docs/logs/lb_noise_effb4_hyst.txt` |
| F9 | Qua 12 submission: Spearman(val, LB) 0.92; LB ≈ 0.84 × val; U-Net ≥ 768 có LB thấp hơn val trung bình 0.059 | [EXPERIMENTS § Tự phản biện](EXPERIMENTS.md#tự-phản-biện-và-kiểm-chứng-2026-10-25--scriptscomparepy) |
| F10 | Model dự đoán trên test và val giống nhau (6.57 so với 6.37 instance mỗi ảnh; foreground 3.25‰ so với 3.18‰), nên khoảng cách LB đến từ phía **nhãn**, không phải từ ảnh | như trên |
| F11 | PQ theo nhóm annotator trên val: 01 **0.383**, 04 **0.373**, 02 0.461, 03 0.455, 05 0.425, 06 0.443. Theo từng annotator: 0.286–0.549 (trung vị 0.411) | `docs/logs/critique2_errors_effb4_hyst.txt` |
| S7 | Hỏi metric xử lý ảnh test nhiều annotator thế nào, host trả lời rằng thông tin này không giúp được đội nào, tức **từ chối trả lời**. Vậy không biết số annotator mỗi ảnh test, và cũng không biết ai gán. Public LB chấm ~50% ảnh test | [discussion 753043](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753043) (~27/10); con số ~50% lấy từ [Overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview). A |
| H5 | **Khoảng cách val → LB (~0.05) chủ yếu do lệch annotator**: nếu test do nhóm "khó" (01/04) gán nhiều hơn thì LB ≈ 0.37–0.38 trước nhiễu | Ủng hộ: F10, F11, F7; umerkang66 thấy PQ theo nhóm annotator 0.40–0.55 (S11d); mọi đội đều lệch 0.05–0.06 (§ 6.2). Phản đối: chưa có. Không kiểm chứng được nếu host không công bố. **Hệ quả**: không tune được cho test; chỉ model tốt hơn ở mọi nhóm mới nâng LB |

## 4. Mình đang mất PQ ở đâu (EffNet-B4 @1024 + hysteresis chặt, val 213 ảnh)

Tổng: PQ 0.410 · SQ 0.668 · TP 1,401 · FP 794 · FN 965 (tune in-sample, lạc quan ~0.003). Nguồn:
`docs/logs/critique2_errors_effb4_hyst.txt`.

| Mã | Nhận định | Số liệu | Ý nghĩa |
|---|---|---|---|
| F12 | Filament nhỏ bị bỏ sót | < 400 px: recall 0.32, **54% không có dự đoán nào chạm** (IoU < 0.1); 400–800 px: recall 0.52, 30% không chạm | chi tiết nhỏ bị mất (độ phân giải hoặc model) |
| F13 | Mask nhỏ dự đoán quá to | diện tích dự đoán / GT: 1.39 (< 400 px), 1.23 (400–800 px), ~1.02 (> 3,200 px) | lỗi biên ở mask nhỏ |
| F14 | Trần độ phân giải (GT → hạ xuống → phóng lên 2048, model hoàn hảo) | IoU trung bình 0.797 @512, **0.894 @1024**, 0.909 @1536; riêng < 800 px: 0.735 / 0.860 / 0.881 | độ phân giải không phải nút thắt chính (SQ hiện tại 0.668), nhưng có tác dụng với filament nhỏ |
| F15 | Filament được nhiều người đồng thuận đã bắt gần hết | ảnh 2 annotator: GT có người khác cũng vẽ → recall **0.83**; GT chỉ một người vẽ → 0.32. Ảnh 3 annotator: 0.83 / 0.60 / 0.27 | phần lớn FN ở ảnh nhiều annotator là bất đồng giữa người với người |
| F16 | Dự đoán không khớp annotator nào | 33% ở ảnh 1 annotator, 27% ở ảnh 2, 17% ở ảnh 3 | FP "thật" tập trung ở ảnh một annotator |
| F17 | Gần trúng (IoU 0.3–0.5) | 17–20% ở filament 1,600–8,000 px+ | lỗi biên ở filament lớn |
| F18 | Tỉ lệ lỗi so với đội đầu | FP/TP **0.57**, FN/TP 0.69; RQ 0.614, SQ 0.668 | theainerd: FP/TP 0.42, FN/TP 0.61, RQ 0.660 (mình suy ra từ TP/FP/FN của họ), SQ 0.685 (S10c) → khoảng cách chủ yếu ở **FP**, tức khả năng phân biệt |
| F19 | Metric rubric | Dice trung bình mỗi bản gán nhãn 0.668; tỉ lệ 1:n 3.9%, n:1 3.4% | umerkang66: 3.0% / 2.9% (S11d) |

## 5. Những gì đã kiểm chứng trên dữ liệu của mình

Chi tiết và CI ở [EXPERIMENTS § Sổ cái](EXPERIMENTS.md#sổ-cái). Tóm tắt các bài học:

| Mã | Bài học | Bằng chứng |
|---|---|---|
| F20 | Encoder quan trọng hơn độ phân giải trong khoảng 768–1024 | EffNet-B4 hơn R34 +0.013 [+0.003, +0.023]; R34 768 → 1024 chỉ +0.008 (không có CI) |
| F21 | Hậu xử lý đã cạn | hysteresis chặt +0.007 [+0.002, +0.011]; chỉnh theo kích thước +0.005; hai cái **không cộng dồn**; tương đương bộ lọc FP; 9 lượt nộp biến thể đều trong 0.35–0.36 |
| F22 | Trộn model yếu với model mạnh không giúp | ensemble 6 model holdout −0.001 [−0.009, +0.007] |
| F23 | Tách instance bằng luật (watershed, gộp seed, lấp lỗ) chỉ đổi lỗi gộp lấy lỗi tách | E5w +0.001, R4 0.0000 |
| F24 | Độ nhạy: CI của Δ ghép cặp trên 213 ảnh rộng ~±0.010, tức ~±0.012 trên một fold (~141 ảnh) | `docs/logs/critique_models.txt` |
| F25 | Train fold trên Mac (MPS) không khả thi: ≥ 1.7 h/epoch, ≥ 50 h một fold | benchmark 29/10 |

## 6. Đối thủ trung thực và luật của host

Mọi dòng dưới đây đã được đối chiếu lại với nguồn gốc ngày 29/10 (A). Ngày của bài trên Kaggle được
quy đổi từ "N ngày trước". Kể từ 27/10, không repo nào trong số này có commit mới.

### 6.1 Luật và tiêu chí chấm (host)

| Mã | Nhận định | Nguồn |
|---|---|---|
| S6 | Ngày 23/10 host rà lại top 250 bài. LB sẽ được dọn **sau** deadline (~27/10 host nhắc lại: không muốn lộ guardrail). **Weights phải nằm trong repo và được notebook load**. Host không khuyến khích dùng AI viết report mà thiếu kiểm soát | [discussion 724503](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503) |
| S6b | Hai tiêu chí chính của ban giám khảo (~26/10): **chất lượng** (xem trực tiếp mask trên một số ví dụ họ coi là quan trọng) và **tính khả thi** (thời gian chạy hợp lý, không phức tạp hoá, quy trình dễ hiểu; model "cài đặt tốt" được điểm cao) | như trên |
| S8 | **Rubric** (Overview > Evaluation, cập nhật 7/8): 70% định lượng gồm Dice trung bình (`torchmetrics.segmentation.DiceScore`), PQ, phân bố Dice, phân bố IoU, phân bố 1:n / n:1; 30% định tính. Host sẽ xem kỹ mọi bài có PQ > 0.35 (bài 747399 viết nhầm thành "3.5"). Về hiệu năng: chỉ loại pipeline "quá chậm hoặc rò bộ nhớ", "không phải yếu tố cạnh tranh". Host **không** nói rõ ensemble lớn hay TTA được chấp nhận; đó là suy luận của mình | [Overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview), [747399](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/747399) (~8/10), [749526](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/749526) (~15/10) |
| S17 | Được dùng frame GONG lân cận (chuyển nhãn sang, hoặc dùng làm ảnh không nhãn), miễn loại các frame trong ±1 h quanh thời điểm ảnh test | [discussion 748045](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/748045) (~11/10) |
| S18 | Câu hỏi mở, host chưa trả lời: barb có thuộc cùng instance không; filament đứt đoạn là một hay nhiều instance; có được dùng metadata/frame lân cận lúc infer không; ~800 MB weights để thẳng trong repo có được không. **Cần theo dõi** | [discussion 753665](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753665) (~28/10) |

**Hệ quả** (suy luận): S6b ưu tiên một pipeline **gọn, giải thích được** hơn một ensemble khổng lồ.
Mỗi thành phần thêm vào phải có lợi đo được (CI > 0). Đây là lý do PLAN đặt tiêu chí chấp nhận chặt.

### 6.2 Các đội trung thực có công bố (S5)

| Đội | OOF PQ | Public | Cách đo OOF | Nguồn |
|---|---|---|---|---|
| tushhaaarrrr | **0.4597** (nested 0.4591) | 0.40 | end-to-end @2048 | [notebook](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40) (~9/2026) |
| biconcavelens | 0.461 | 0.40 | trung bình 5 lần chia calibration, mỗi lần 70 ảnh val; **không phải** OOF k-fold | [README L13](https://github.com/biconcavelens/filament-segmentation-2026/blob/master/README.md?plain=1#L13) |
| theainerd | 0.4519 (CI 0.4426–0.4611) | 0.40 | 5 fold | [README L9](https://github.com/theainerd/filament_segmentation/blob/main/README.md?plain=1#L9) |
| Dariush Afshar | 0.4502 | 0.39 | nested CV trên 707 ảnh; 5 R34 + 5 EffNetV2-S, infer @2048, TTA 8 | [notebook](https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39) (~18/10) |
| umerkang66 | 0.4477 | 0.39 | nested OOF | [REPORT L13](https://github.com/umerkang66/solar-filament-segmentation/blob/master/REPORT.md?plain=1#L13) |
| zhang0894 | 0.4427 | 0.38 | ước lượng cho công thức một fold; tác giả nói **không** phải OOF của ensemble | [README L16](https://github.com/zhang0894/Solar-Filament-Segmentation-Challenge-2026-by-Anon-Tokyo/blob/main/README.md?plain=1#L16) |
| BSergey2000 | 0.432 | 0.36 | CV | [README L24](https://github.com/BSergey2000/filament-segmentation-2026/blob/main/README.md?plain=1#L24) |
| ranjeet258 | – | 0.39 (chỉ model) | – | [README](https://github.com/ranjeet258/solar-filament-consensus-2026/blob/main/README.md) (28/10) |
| **Mình** | 0.408 (holdout 213 ảnh, cross-tune) | 0.36 | holdout 70/30 | [EXPERIMENTS](EXPERIMENTS.md#sổ-cái) |

Khi so cần lưu ý: các con số OOF **không cùng cách đo** (nested hay không, k-fold hay chia
calibration). Chênh lệch dưới ~0.01 giữa các đội không có ý nghĩa. Mọi đội đều có LB thấp hơn OOF
0.05–0.06.

### 6.3 Kỹ thuật và hiệu quả đo được (chỉ ghi số có trong nguồn)

| Mã | Đội | Kỹ thuật | Hiệu quả đo được | Cách đo | Nguồn |
|---|---|---|---|---|---|
| S9a | biconcavelens | model đơn | Mask R-CNN **0.429**, YOLO11m-seg 0.437, RT-DETR-l 0.436, YOLO11l-seg 0.420, U-Net ngữ nghĩa (ConvNeXt-T, crop 1024, 62 epoch, ~3 h) **0.396** | trung bình 5 lần chia | [paper/main.tex L149–160](https://github.com/biconcavelens/filament-segmentation-2026/blob/master/paper/main.tex#L149) |
| S9b | biconcavelens | hiệu chỉnh isotonic điểm detector | bỏ hiệu chỉnh: 0.450 → 0.428 (**−0.021**) | ablation, 4 detector | main.tex, bảng ablation |
| S9c | biconcavelens | bỏ phiếu mask có trọng số điểm | so với 4 detector đã hiệu chỉnh: +0.007 (12 nguồn), +0.012 (thêm model ngữ nghĩa). Hệ thống đầy đủ chỉ có CI 95% > 0 ở **3/5** lần chia | bootstrap | main.tex, bảng ablation |
| S9d | biconcavelens | chuyển nhãn sang frame lân cận | model ngữ nghĩa một mình: 0.404 → 0.372 (kém ở 5/5 lần chia). Trong hệ thống đầy đủ: 0.4594 → 0.4577 (CI chứa 0) | 5 lần chia | [RESULTS.md L96](https://github.com/biconcavelens/filament-segmentation-2026/blob/master/RESULTS.md?plain=1#L96) |
| S9e | biconcavelens | **thất bại**: clDice (0.4252 → 0.4192), CLAHE (0.425 → 0.412–0.419), pseudo-label test (0.417 → 0.393), pretrain SimSiam (0.448 so với 0.449; public 0.38), Mask2Former (≤ 0.345), SAM với box prompt (0.316) | | | main.tex L326–330, RESULTS.md L40 |
| S9f | biconcavelens | tài nguyên | Mask R-CNN train @2048, 6 epoch; YOLO11m @1280, 40 epoch; trên **một GPU Kaggle 16 GB** hoặc RTX 4060 laptop 8 GB; infer Mask R-CNN 1.9–2.2 s/ảnh | | main.tex L77–90, L343 |
| S10a | theainerd | công thức | input 3 kênh (tương phản đã hiệu chỉnh limb, cường độ / median đĩa, bán kính); EffNet-B3 + scSE; target mềm; crop/tile 512 (chồng 128 px, trộn Hann, 4 góc xoay); hysteresis 0.5/0.3; nối mảnh cách nhau < 8 px; rescorer GBM τ 0.25 | | [README L80–93](https://github.com/theainerd/filament_segmentation/blob/main/README.md?plain=1#L80) |
| S10b | theainerd | **BiCC loss** ([arXiv 2609.30223](https://arxiv.org/abs/2609.30223), 24/9/2026) | so với B3, từng fold: +0.0080, +0.0052, +0.0230, +0.0135, **−0.0019**, tức tăng ở 4/5 fold | 5 fold, không có CI | README L99–109 |
| S10c | theainerd | PQ từng fold | B3 0.414–0.438; B3 + BiCC 0.412–0.446; pipeline đầy đủ 0.443–0.461. TP 5,092, FP 2,135, FN 3,107, SQ 0.6846. Mình suy ra: RQ 0.660, FP/TP 0.42, FN/TP 0.61 | | README L99–109 |
| S10d | theainerd | tài nguyên | train 10 model mất ~**17 h trên một A10**; notebook infer ~1 h GPU | | README |
| S11a | umerkang66 | blob loss + head skeleton/core + deep supervision + loss JML1 (thay soft Dice) | **+0.0092, CI 90% [+0.006, +0.012]**. Đây là ablation proxy (ResNet-34 @1024, 5 fold) và là hiệu ứng **duy nhất** họ báo có CI | 5 fold, bootstrap ghép cặp | [REPORT L104](https://github.com/umerkang66/solar-filament-segmentation/blob/master/REPORT.md?plain=1#L104) |
| S11b | umerkang66 | EMA 0.998 | có dùng, không có số tách riêng | | REPORT L107 |
| S11c | umerkang66 | gộp seed trong 16 px | +0.008; không có CI, không nói số fold | | REPORT L125 |
| S11d | umerkang66 | phân tích lỗi | phần PQ bị mất: 56% do biên, 15% do bỏ sót, 9% do tách/gộp. PQ theo nhóm annotator 0.40–0.55. Tỉ lệ 1:n 3.0%, n:1 2.9% | | REPORT L190–206 |
| S11e | umerkang66 | **thất bại**: closing theo instance bán kính 2–6 (−0.003 đến −0.016); co/giãn 1 px cả submission (public 0.37 / 0.35) | | | REPORT L229 |
| S11f | umerkang66 | target mềm | có dùng nhưng **không có ablation**. Con số "+0.01–0.03" trong bản research cũ là ước tính của mình, không phải số đo của họ | | – |
| S12 | ranjeet258 | 3 U-Net (R34, ResNeXt50, EffNet-B4) × 4 fold + EMA + TTA 8 + YOLO11m-seg + LightGBM dự đoán IoU khớp kỳ vọng theo từng annotator | public 0.39 (chỉ model). Đội này cũng có 2 bài 0.55 dựng trên payload | | [README](https://github.com/ranjeet258/solar-filament-consensus-2026/blob/main/README.md) |
| S13 | BSergey2000 | độ phân giải | native 2048 với crop 640 **kém 0.07** so với resize 1024 + crop 768 ("ngữ cảnh quan trọng hơn độ phân giải") | CV | [README L24](https://github.com/BSergey2000/filament-segmentation-2026/blob/main/README.md?plain=1#L24) |
| S15 | tushhaaarrrr | chuẩn hoá vật lý (fit limb + flat field); ensemble dày đặc U-Net R34, U-Net ConvNeXt-T, UNet++ R34 (5 fold, TTA D4); target mềm + head spine; bằng chứng từ RF-DETR; cây thành phần qua 15 ngưỡng; **lớp quyết định học được** | tác giả quy **+0.06 PQ** cho lớp quyết định so với mọi ngưỡng cố định (không có CI) | OOF @2048 | [notebook](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40) |

**Đọc bảng này thế nào** (suy luận của mình):
- Chỉ **S11a** có CI. Mọi số khác là một lần đo hoặc thiếu nhất quán giữa các fold. Kỳ vọng cho
  mình nên lấy khoảng **một nửa** con số công bố.
- Hai bằng chứng về độ phân giải mâu thuẫn nhau (S10a tile 512 native, S13 −0.07), nên phải tự đo
  (N2).
- S15 cho thấy **bước quyết định instance** (chọn ngưỡng theo từng thành phần) có thể rất đáng giá.
  Mình đã có phiên bản đơn giản (bộ lọc GBM, hysteresis), chỉ được +0.004 đến +0.007 (F21). Con số
  +0.06 của họ là so với ngưỡng cố định, không phải so với một bộ lọc như của mình, nên không so
  thẳng được. Đây là ứng viên cho một thẻ sau N3 nếu còn thời gian (H6).

## 7. Giả thuyết cải thiện

Mỗi giả thuyết ứng với một thẻ trong [PLAN § 4](PLAN.md#4-thẻ-thí-nghiệm).

| Mã | Giả thuyết | Ủng hộ | Phản đối / bất định | Thẻ |
|---|---|---|---|---|
| H1 | Ensemble 5 fold train trên toàn bộ dữ liệu tốt hơn 1 model trên 70% | mọi đội ở § 6.2 đều nộp ensemble fold | F22 (nhưng đó là trộn model yếu); lợi ích không đo được trên OOF | N1 |
| H2 | Độ phân giải gốc (crop @2048) cứu filament nhỏ | F12, F14 (< 800 px: 0.860 → 0.881), S10a, S15 (@2048) | S13 (−0.07), F14 (trần tăng ít), F20 | N2 |
| H3 | Gói EMA + blob loss + input hiệu chỉnh limb cho +0.01–0.02 | S11a (+0.009, CI 90% > 0), S10a, S11b, S12, S15 (chuẩn hoá vật lý) | S11a là cả một gói (blob loss + head + JML1), không phải blob loss riêng; EMA và input limb chưa có số tách riêng; độ nhạy F24 | N3 |
| H4 | Detector (Mask R-CNN) giảm FP; ghép với U-Net được +0.02–0.04 | F18 (khoảng cách nằm ở FP), S9a (+0.033 so với U-Net của họ), S9c (bỏ phiếu +0.007–0.012) | U-Net của S9a (0.396) yếu hơn của mình (~0.41); S9c chỉ có CI > 0 ở 3/5 lần chia; tốn code nhiều nhất; đi ngược S6b (phức tạp hoá) | N4 |
| H6 | Lớp quyết định instance học được (ngưỡng theo từng thành phần, cây thành phần nhiều ngưỡng) tốt hơn bộ lọc GBM hiện tại | S15 (+0.06 so với ngưỡng cố định), S11c | F21 (bộ lọc và hysteresis của mình chỉ +0.004–0.007 và không cộng dồn); không có CI | **bác bỏ 30/10**: cây −0.004 [−0.011, +0.003], đồng thuận model −0.0004 [−0.003, +0.002] (EXPERIMENTS § W-agree) |
| H5 | Khoảng cách val → LB do lệch annotator | mục 3 | không kiểm chứng được | không có thẻ: chấp nhận như ràng buộc |

## 8. Không được làm

- Dùng MAGFiLO 1.0 công khai dưới bất kỳ hình thức nào (kể cả làm ảnh pretrain không nhãn), vì nó
  chứa ảnh test (`S1`).
- Nộp mask chồng nhau hoặc trùng lặp (`S3`). Hiện mask rời nhau theo cách dựng: mọi instance được
  mã hoá từ **một** ảnh nhãn (`encode_masks`). Chưa có assert. Khi ghép detector (N4) thì mask có
  thể chồng nhau, nên phải giải chồng lấn và thêm assert trước khi nộp.
- Tune theo LB public (F8, F9, H5).

## 9. Đã sửa so với bản research trước (đối chiếu nguồn 29/10)

| Bản cũ ghi | Sự thật theo nguồn | Ảnh hưởng tới kế hoạch |
|---|---|---|
| 76 đội ở 0.55 | 82 đội (LB 29/10) | không |
| Host remarks ngày 24/10 | ngày 23/10 | không |
| 753043: host không công bố ai gán nhãn test | thread chỉ cho thấy host từ chối nói cách chấm ảnh nhiều annotator; con số "~50% public" lấy từ Overview | H5 vẫn đứng (không biết annotator test), nhưng dẫn nguồn đúng |
| Rubric nằm ở 747399; 749526 chấp nhận ensemble/TTA | rubric ở Overview > Evaluation; 749526 chỉ nói hiệu năng không phải yếu tố cạnh tranh | thêm tiêu chí "không phức tạp hoá" (S6b) vào mục tiêu |
| BiCC +0.003 đến +0.022 ở 4/5 fold | +0.005 đến +0.023 ở 4/5 fold, −0.002 ở fold thứ 5 | không |
| theainerd mỗi fold 0.414–0.446 | B3 0.414–0.438; B3 + BiCC 0.412–0.446; RQ và FP/TP là số mình tự suy ra | không |
| Blob loss +0.009, CI [+0.006, +0.012] | là gói blob + head + deep supervision + JML1, CI **90%**, ablation proxy R34 | N3 ghi rõ đây là gói |
| Hiệu chỉnh isotonic +0.021 | bỏ hiệu chỉnh thì mất 0.021 | N4: hiệu chỉnh là bắt buộc khi ghép |
| Chuyển nhãn sang frame lân cận 0.404 → 0.372 | chỉ đúng với model ngữ nghĩa một mình; trong hệ thống đầy đủ không đổi | không |
| Target mềm +0.01–0.03 (umerkang66) | umerkang66 không có ablation; con số này là ước tính của mình | N1 không hứa hẹn con số này |
| clDice, CLAHE, pseudo-label, SimSiam, Mask2Former, SAM thất bại (umerkang66) | là kết quả của biconcavelens | không |
| "Closing/dilate 1 px theo instance" thất bại | umerkang66: closing bán kính 2–6 kém 0.003–0.016; co/giãn 1 px cả submission public 0.37/0.35 | không |
| "Chỉ train trên một annotator" thất bại | không tìm thấy nguồn → bỏ | không |
| Link REPORT của umerkang66 (`/blob/main/`) | 404; đúng là `/blob/master/` | đã sửa |
| (thiếu) | tushhaaarrrr OOF 0.4597, public 0.40; lớp quyết định học được +0.06 | thêm H6 |
| `make_submission.py` assert mask rời nhau | không có assert; mask rời nhau vì được dựng từ một ảnh nhãn | N4 phải thêm assert |

## 10. Workflow nhiều agent 30/10: tìm ý tưởng cải thiện

**Cách làm**: 3 agent điều tra độc lập, mỗi agent một góc nhìn (lớp quyết định instance, cách train,
pháp y lỗi) → 1 agent "judge" phản biện, chấm điểm và chọn → 2 agent build trên nhánh riêng
(`exp/agree-filter`, `exp/train-v2`) → 2 reviewer độc lập tìm cách bác bỏ. Không dùng GPU. Báo cáo
nguyên văn của từng agent: `docs/logs/wf30/` (`ideation_*.md`, `judge.md`, `builds_reviews.md`).
Script: `scripts/research/wf30/`. Mọi số đo trên E9a holdout (213 ảnh, hysteresis chặt, PQ 0.4103
in-sample / 0.4083 cross-tune).

### 10.1 Phát hiện mới (đo trực tiếp)

| Mã | Nhận định | Số liệu | Nguồn |
|---|---|---|---|
| F26 | Với mask rời nhau, mỗi bản gán nhãn có TP + ½FP + ½FN = ½(n_pred + n_gt). Vì vậy luật giữ instance tối ưu là: **giữ nếu IoU khớp trung bình qua các annotator của ảnh > PQ/2 (~0.21)**. Trên cây thành phần, lời giải tối ưu là antichain trọng số lớn nhất (DP) | đại số, kiểm bằng code | `ideation_decision.md` |
| F27 | **Trần oracle của tầng chọn instance rất cao, nhưng không học được**. Oracle: giữ/bỏ +0.058, chọn trên cây +0.125. Học được: −0.004 [−0.011, +0.003], hai agent độc lập, code khác nhau. Bộ dự đoán IoU chỉ đạt R² 0.26 (cần ~0.85); DP chọn đúng các nút bị ước lượng quá cao | oracle 0.535 so với 0.410 | `errors_select_fast_r4.out`, `decision_learned.log` |
| F28 | **FP gồm gì** (794 bản gán nhãn): khớp GT của **annotator khác** cùng ảnh 28.7% (bỏ thì mất TP của người kia); FP "mồ côi" trong đĩa 21.3%; mảnh vụn nằm trong một GT 18.6%; gần trúng nhưng **mảnh quá** 18.0%; gần trúng nhưng béo quá 6.5%; gộp ≥ 2 GT 5.4%; limb/sunspot/ngoài ±70°: 3 FP (0.4%). Chỉ 120 FP (15%) không chạm GT của bất kỳ ai | PQ nếu bỏ hết từng nhóm: +0.022 / +0.016 / +0.014 / +0.013 / +0.005 / +0.004 | `errors_analyze.out`, `training_summary_effb4.txt` |
| F29 | **FN gồm gì** (965): sót ở ảnh 1 annotator 22.2% (trung vị 608 px); sót filament chỉ một người vẽ ở ảnh nhiều annotator 19.7%; gần trúng mảnh quá 14.5% (trung vị 1,816 px; diện tích dự đoán/GT 0.58); bị gộp vào filament bên cạnh 14.2%; gần trúng béo quá 9.7%; sót dù có người khác cũng vẽ chỉ **6.5%**; bị cắt đôi 6.0% | | `errors_analyze.out` |
| F30 | **Filament nhỏ bị sót phần lớn do khác biệt giữa người với người**: recall của người với người cho GT < 400 px là 0.33 (47% không chạm), của model 0.32; GT nhỏ có người khác cũng vẽ thì model bắt 0.50. Chỉ 37% GT < 400 px ở ảnh nhiều annotator được người thứ hai vẽ. Lưu ý: số của người và của model đo trên hai tập ảnh khác nhau | | `training_summary_effb4.txt` |
| F31 | Lỗi "gần trúng" phần lớn cũng là bất đồng: ở ảnh nhiều annotator, 59% dự đoán gần trúng lại là TP với một annotator khác | | `ideation_errors.md` |
| F32 | Đổi encoder R34 → EffNet-B4 làm giảm FP cô lập (291 → 257) và FP một phần (189 → 157), **không** giảm FP gần trúng (301 → 289) hay FP gộp (66 → 67). Encoder tốt hơn giúp phân biệt, không giúp đường biên hay cách nhóm | | `training_kinds_*.txt` |
| F33 | Trần nếu sửa hết từng nhóm (các nhóm chồng nhau, không cộng được): mọi gần trúng thành TP IoU 0.55 → +0.056 (mảnh quá +0.034, béo quá +0.023); ghép các ca bị cắt đôi +0.021; tách các ca bị gộp +0.028; nâng mọi TP lên IoU ≥ 0.70 → +0.033 | | `errors_analyze.out` |
| F34 | Độ tự tin của map phân biệt kém: AUC giữa dự đoán trúng và không trúng là 0.72; 75% dự đoán ở cả hai nhóm có đỉnh p = 1.0. Trên hysteresis chặt, bộ lọc 17 đặc trưng học trong cross-tune cho −0.0003; thêm đặc trưng đồng thuận giữa 5 model cho −0.0004 [−0.003, +0.002] | | `docs/logs/wf_agree_filter.txt` |
| F35 | Lịch sử train E9a dao động: val PQ ep10 0.352, ep15 0.342 (FP 1,729), ep20 0.375, ep25/30 0.381 | | `unet1024_effb4.history.csv` |

### 10.2 Kết luận rút ra

1. **Tầng hậu xử lý / chọn instance trên map hiện tại đã cạn**, có bằng chứng trực tiếp (F27, F34,
   F21). Mọi cải thiện còn lại phải đến từ **map tốt hơn**.
2. Phần mất lớn nhất **không gỡ được bằng model**: FP khớp annotator khác (28.7%), FN chỉ một người
   vẽ (19.7%) và phần lớn filament nhỏ bị sót (F30). Đây là nhiễu gán nhãn; mức người với người chỉ
   0.34 (F6).
3. Phần model **có thể** gỡ: đường biên ở filament vừa và lớn (mảnh quá, F29), cách chia instance
   (gộp, cắt, mảnh vụn: F28, F29), và ~120 FP giả thật sự. Trần lý thuyết của các nhóm này là
   +0.02 đến +0.05 mỗi nhóm (F33). Nhưng thực tế khó vượt +0.01–0.02, vì bất đồng giữa annotator
   chiếm luôn một phần các nhóm này (F31).
4. **Kỳ vọng trung thực**: không có ý tưởng nào đã kiểm chứng hứa hẹn bước nhảy lớn (≥ +0.03). Con
   đường thực tế lên ~0.43–0.45 OOF là cộng dồn: N1 (5 fold) + N3' (loss instance) + có thể N4
   (detector cho cách chia instance). Mỗi bước chỉ được nhận khi qua tiêu chí ở PLAN.

### 10.3 Bảng ý tưởng và quyết định

| Ý tưởng | Góc | Điểm judge | Quyết định | Lý do / bằng chứng |
|---|---|---|---|---|
| Loss chuẩn hoá theo instance: blob Dice + phạt thành phần dự đoán không chạm GT (kiểu BiCC) + EMA, trên công thức N1 | train | 6 | **Build** → `exp/train-v2`, thẻ N3' | nhắm vào FP mảnh vụn/cô lập (F28) và đường biên; bằng chứng S11a, S10b; review: cần ghi PQ trọng số thô, ngưỡng 0.35, chạy 2 fold vì nhiễu seed |
| Bỏ input hiệu chỉnh limb và mục tiêu recall filament nhỏ | train / lỗi | 6.5 | nhận (chỉ đổi plan) | F28 (lỗi vị trí 0.4%), F30 |
| Lớp quyết định học được (H6) | quyết định | 6 | **bác bỏ** (kết quả âm có giá trị) | F27 |
| Ngừng hậu xử lý, bỏ prior vị trí/sunspot | lỗi | 5.5 | nhận (chỉ đổi plan) | F28, F34 |
| Dùng nhóm lỗi (gần trúng, cách chia) làm chẩn đoán cơ chế cho N3 | lỗi | 5.5 | nhận, đưa vào thẻ N3' | F29, F33; không dùng để quyết định |
| Siết G1 (N2): N2 phải giảm nhóm FN "có người khác vẽ" và "béo quá", không chỉ PQ | lỗi | 5 | nhận, đưa vào thẻ N2 | F30: phần lớn filament nhỏ bị sót là do bất đồng |
| Đặc trưng đồng thuận giữa các model cho bộ lọc | quyết định | 3.5 | **Build** → `exp/agree-filter`: **âm** | −0.0004 [−0.003, +0.002] (F34) |
| Mask R-CNN cho lỗi chia instance (N4) | train / lỗi | 3.5 / 3 | hoãn, giữ cổng G3 | nhóm gộp + cắt chiếm ~25% FN; bằng chứng ngoài chỉ có một số (S9a) và model của họ yếu hơn của mình |
| Chạy lại oracle/cây trên map OOF của N1 | quyết định | 3 | hoãn, chỉ khi CPU rảnh | chẩn đoán cho report |
| Train có điều kiện theo annotator (style head) | train | 1.5 | **loại** | không có bằng chứng; annotator của test không biết (S7) |
