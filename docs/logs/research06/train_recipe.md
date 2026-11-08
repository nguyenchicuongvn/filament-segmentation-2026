# Research 06/11 — `train_recipe`: thay đổi phía train cho U-Net EfficientNet-B4 trước khi đóng băng 09/11

Agent: `train_recipe` (một trong các agent của workflow lập kế hoạch tuần cuối). Phạm vi: chỉ những thay đổi
**phía train** vừa trong ≤ 2 run Colab (mỗi run 5 fold ≈ một ngày Colab) và có thể cho ≥ +0.005 OOF PQ.
Không đề xuất hậu xử lý (RESEARCH § 13 đã cạn). Không train gì trong lúc nghiên cứu; mọi số đo mới ở đây
chỉ là đọc log có sẵn.

## 0. Công thức hiện tại (đọc từ `scripts/train_unet.py`, `src/filseg/dl.py`, `colab/train_unet.ipynb`)

| Khâu | Giá trị hiện tại (N1 = `effb4_s1024_f{k}`) |
|---|---|
| Model | `smp.Unet("efficientnet-b4", encoder_weights="imagenet", in_channels=1)`; 20.2 M tham số; skip ở stride 2 có |
| Input | ảnh 2048 → resize INTER_AREA 1024, một kênh, chia 255, **không** chuẩn hoá ImageNet (F50) |
| Target | mềm: tỉ lệ annotator vẽ pixel (uint8/255), `--targets soft`; `WeightedRandomSampler` theo số annotator, `num_samples` = số bản gán nhãn (~920/epoch) |
| Loss | BCE + 1.0 × soft Dice (theo ảnh) |
| Tối ưu | AdamW lr 1e-3, wd 1e-4; OneCycleLR `pct_start 0.05` (anneal cosine về ~0 ở epoch 30); AMP fp16; bs 6 (L4/A100) hoặc 4 (T4); 30 epoch |
| Augmentation | rot90 ×4, lật ngang, gamma 0.7–1.4, tương phản 0.8–1.2, độ sáng ±0.08. Không scale/elastic/blur/cutmix. RNG augmentation không seed (F44) |
| Checkpoint | val PQ (cc cố định, thresh 0.35) ở epoch 10/20/30 trên chính fold held-out; lấy tốt nhất |
| Tuỳ chọn có sẵn, chưa dùng trong final | `--ema 0.998` (lưu cả `model` = EMA và `model_raw`), `--dice-weight 0` (N7, không qua gate), `--blob-weight/--fpc-weight` (v2, âm), `--seed` (N8), `--crop` (N2) |
| Chi phí thật | Kaggle T4, bs 4: **110–115 phút/fold** (log `outputs/kaggle/filseg-gpu-n1-*`); L4 ~75 phút; A100 ước ~30–40 phút. 5 fold trên Colab T4 ≈ 9.5 h ≈ một phiên/ngày |

Baseline để so: final 1 = N1 + N2 (10 model), OOF 707 ảnh cross-tune `hyst_soft` **0.4284**; thêm BCE (15 model)
+0.0018 [−0.0008, +0.0045] không qua gate.

## 1. Đọc lại bằng chứng trong repo về "thêm thành viên ensemble"

Đây là đòn bẩy phía train duy nhất đã đo được dương, nên trước khi ra ngoài tìm, xem lại nó cho kỹ:

| Thành viên thêm vào N1 | Khác N1 ở đâu | Δ OOF so với nền | CI 95% | Nguồn |
|---|---|---|---|---|
| N2 (crop 1024 @2048) | **độ phân giải input** | +0.0105 (707 ảnh, so với N1) | [+0.0072, +0.0140], 5/5 fold dương | EXPERIMENTS G4 |
| BCE thuần (N7) | **loss** (bỏ Dice), cùng kiến trúc, cùng input | +0.0018 (707 ảnh, so với N1 + N2) | [−0.0008, +0.0045] | EXPERIMENTS gate N7 |
| v2 (blob + fpc + EMA) | loss, model đơn kém 0.05 | +0.0109 (f0) nhưng +0.0005 (f1); thêm vào N1 + N2: −0.0021 | – | EXPERIMENTS ENS-n1v2 |
| N8 seed 2 (đang xếp hàng Colab) | **chỉ seed init + sampler** | chưa có | – | PLAN thẻ N8, kỳ vọng +0.003–0.005 |

Nhận xét: lợi ích ensemble trong repo **tăng theo mức khác biệt** của thành viên (độ phân giải ≫ loss). Một
thành viên chỉ khác seed là mức khác biệt thấp nhất có thể, nên kỳ vọng +0.003–0.005 của thẻ N8 có lẽ **lạc
quan**; ước lượng trung thực hơn là +0.001–0.003, tức nhiều khả năng **không qua gate CI > 0**. Điều này
khớp với lý thuyết ensemble: Wenzel et al. 2020 ("Hyperparameter Ensembles for Robustness and Uncertainty
Quantification", NeurIPS, https://arxiv.org/abs/2006.13570) cho thấy ensemble vừa khác init vừa khác
hyperparameter/kiến trúc ("hyper-deep ensembles") thắng deep ensembles chỉ khác init trên mọi kiến trúc họ
thử (MLP, LeNet, ResNet-20, WRN-28-10); Georgescu et al. 2023 (DiPE, SAC '23, https://arxiv.org/abs/2210.12388)
xây ensemble U-Net từ nhiều backbone (ResNet-34, EfficientNet-B0, …) và chọn các cặp model ít tương quan
(Dice giữa dự đoán thấp) thì vượt cả model đơn tốt nhất lẫn ensemble "top-k". Cả hai paper đều không đo
riêng "backbone khác so với seed khác" ở mức Dice nên không có một con số chuyển thẳng được; bằng chứng
trực tiếp hơn là **trong repo** (bảng trên) và Dariush Afshar (RESEARCH S5: 5 R34 + 5 EffNetV2-S, OOF
0.4502, infer @2048, TTA 8 — ensemble hai encoder khác họ).

Về EMA: ledger ghi v2 "PQ trọng số thô ≈ EMA". Log kernel xác nhận: val PQ (cc cố định, trong lúc train)
của EMA là **0.3138** (f0) / **0.2910** (f1) so với raw 0.317 / 0.292, tức EMA **−0.003 / −0.001**, không
dương. Giải thích hợp lý: OneCycle anneal LR về ~0 ở epoch 30, nên trọng số cuối đã "tự trung bình"; EMA
0.998 (~500 step ≈ 2 epoch) không thêm được gì. Đây là số trên recipe v2 (loss khác), nhưng cơ chế LR giống
N1. Kết luận: **không kỳ vọng EMA cho lợi đo được**; chỉ thêm nếu miễn phí.

## 2. Những gì bên ngoài nói (chỉ ghi số có trong nguồn)

### 2.1 Kaggle: các cuộc thi vật mảnh / instance

| Cuộc thi | Kỹ thuật phía train của đội top | Số | Chuyển sang bài này? |
|---|---|---|---|
| **SenNet + HOA 2024** (mạch máu 3D, Surface Dice) — hạng 1 Clevert, [discussion 475522](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475522) (đọc qua tổng hợp [zenn.dev/yume_neko](https://zenn.dev/yume_neko/articles/5bc998cfd8cd05); trang Kaggle không tải được nội dung) | augmentation "3D rotate" cắt lát theo trục xoay ngẫu nhiên: private **0.682 → 0.835**; loss mô phỏng metric Surface Dice: **+0.148**; input 2.5D | **Không**: cả hai đòn bẩy đều đặc thù 3D và metric mặt (surface). PQ khớp IoU > 0.5 không phải metric biên; lỗi biên của mình phần lớn là bất đồng annotator (F53, F56 âm với mọi cách chỉnh biên). Loss "mô phỏng metric" của mình là blob/fpc (v2) đã **âm** |
| SenNet hạng 3/5 ([475074](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475074), [475288](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475288)) | pseudo-label để "làm dày" nhãn thưa, lặp nhiều vòng; tác giả zenn tự thử **một vòng** pseudo-label: lợi "rất nhỏ" | Dữ liệu của mình không thưa nhãn (mask đủ trên 707 ảnh). Xem § 2.3 |
| HuBMAP 2021 (kidney) — hạng 4 dùng pseudo-label test; Robin Smits báo pseudo-label "chỉ cho hiệu quả rất hạn chế" ([repo](https://github.com/RobinSmits/KaggleHuBMAP)) | – | Như trên |
| Sartorius 2021, HuBMAP 2023, Vesuvius 2023 | Không tải được write-up hạng 1 (trang Kaggle render phía client; trang albumentations chỉ liệt kê link: Sartorius hạng 6 [297986](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/discussion/297986)). Kiến thức chung: detector + pseudo-label LIVECell (Sartorius), 2.5D/3D CNN + crop lớn (Vesuvius) | Không có số kiểm chứng được trong phiên này → **không dùng làm bằng chứng** |

### 2.2 Paper: multi-rater, soft label

| Nguồn | Kết quả | Ý nghĩa cho mình |
|---|---|---|
| Lemay et al. 2022, MELBA 2022:031, https://arxiv.org/abs/2202.07550 | Dice (%) MS lesion: STAPLE 54.9, **trung bình 54.3**, **lấy mẫu ngẫu nhiên rater 50.6** (conventional); SoftSeg: 55.0 / **56.0** / 53.7. SCGM gray matter: 86.6 / 83.6 / 84.2 (conventional). Kết luận paper: "no equivocal conclusion" về cách hợp nhãn tốt nhất; SoftSeg hiệu chỉnh tốt hơn (ECE 2.9% so với 16.2%) mà không đổi Dice | Target **trung bình** (đang dùng) không thua lấy mẫu ngẫu nhiên; trên tập khó (MS lesion, giống bài mình: vật nhỏ, bất đồng cao) trung bình **hơn** ngẫu nhiên 3.7 điểm Dice. Đổi sang lấy mẫu annotator: **không có cơ sở** |
| Wang et al. 2023, "Dice Semimetric Losses", https://arxiv.org/abs/2303.16296 | QUBIQ (7 task, 3–7 rater), U-Net R50, 5 fold: Dice majority 76.80 / **random 76.30** / **trung bình 77.31** / trung bình có trọng số 79.85 / label smoothing 79.15 | Cùng kết luận: soft trung bình ≥ random. "Trung bình có trọng số" (+2.5) là trọng số theo chất lượng rater — bài mình **không biết** annotator của test (S7) và PQ chấm từng annotator như nhau, nên không áp được |
| F46/F51 trong repo | Dice làm map bão hoà nhưng BCE thuần (hiệu chỉnh tốt) **không** tăng PQ (−0.0002 / −0.0018) | Hiệu chỉnh không phải nút thắt; hysteresis seed tune được đã bù. Không đổi loss nữa |

### 2.3 Pseudo-label / self-training trên 180 ảnh test không nhãn

- Bằng chứng: biconcavelens thử pseudo-label test: **0.417 → 0.393** (S9e, −0.024); HuBMAP 2021 "rất hạn
  chế" (Robin Smits); zenn SenNet: một vòng pseudo-label "lợi rất nhỏ". Lý do cấu trúc: model đã **vượt**
  inter-annotator (0.43 so với 0.34, F53), nên pseudo-label của chính nó chỉ dạy lại "kiểu trung bình" đã
  có; 180 ảnh so với 707 ảnh train (×1,154 bản gán nhãn) là quá ít để đổi gì; F40: test khớp train ở mọi
  trạm, không có domain shift cần thích nghi.
- **Rủi ro luật, hai chiều**: (a) *Ủng hộ*: luật nói "inference uses only test H-alpha images" — self-training
  là bước train, lúc infer vẫn chỉ dùng ảnh test; host cho phép dùng frame GONG lân cận làm ảnh không nhãn
  (S17), nên ảnh không nhãn không bị cấm về nguyên tắc. (b) *Phản đối*: host nói rõ "lúc infer chỉ dùng ảnh
  test (không metadata, không frame lân cận, không MAGFiLO)" (S19) và đang rà **thủ công** mọi bài > 0.35
  (S8); một model mà weights đã "nhìn" ảnh test lúc train rất khó giải thích là không dùng gì ngoài ảnh test,
  và rubric 30% định tính coi trọng quy trình dễ hiểu (S6b). Hơn nữa 139/180 ảnh test có trong MAGFiLO 1.0
  (S1) nên bất kỳ thao tác nào trên ảnh test đều bị ngờ là rò nhãn.
- **Kết luận: ngõ cụt.** Kỳ vọng ≤ 0 theo bằng chứng, rủi ro luật và rủi ro báo cáo cao, tốn một ngày Colab.

### 2.4 Loss topology / skeleton, lịch dài, augmentation nặng, encoder

- clDice / Skeleton Recall / loss theo component: Hao & Li 2026 đo 8 phương pháp × 3 seed trên MAGFiLO
  @1024: **không có lợi ổn định** (S22); biconcavelens clDice **0.4252 → 0.4192** (S9e); JML1 +0.0003 (S11).
  Ngõ cụt, đã có 3 nguồn độc lập cùng chiều.
- Lịch dài / cosine restart: E10 (R34 60 epoch) **−0.005 [−0.014, +0.005]**; F35: PQ phẳng từ epoch 25.
  Không có nguồn ngoài nào trên bài này báo lợi từ epoch dài. Ngõ cụt.
- Augmentation nặng (scale, elastic, cutmix, blur): biconcavelens copy-paste **−0.015**; CLAHE −0.006 đến
  −0.013; F40 không có lệch train/test; F37 model cần chi tiết mịn nên blur đi ngược chiều; elastic làm méo
  đúng đường biên mà metric chấm. Đòn bẩy "3D rotate" của SenNet không có tương đương 2D ngoài rot90/lật đã
  dùng. Không có bằng chứng dương nào → không chi GPU.
- Encoder: trong repo, R34 → EffNet-B4 **+0.013 [+0.003, +0.023]** (E9a) là bước nhảy lớn nhất từng đo;
  ConvNeXt −0.020 do thiếu skip stride 2 (F42). Kiểm tra trên `.venv` (smp 0.5.0, timm 1.0.30): các encoder
  sau build được với `in_channels=1` và **có** skip stride 2: `efficientnet-b5` (31.2 M), `tu-tf_efficientnetv2_s`
  / `tu-tf_efficientnetv2_s.in21k_ft_in1k` (22.1 M, `out_channels [1, 24, 48, 64, 160, 256]`), `tu-regnety_040`
  (24.7 M), `tu-resnest50d` (34.4 M). EfficientNetV2-S là encoder Dariush dùng cạnh R34 (OOF 0.4502).

## 3. Đề xuất (xếp theo lợi kỳ vọng trên mỗi giờ Colab)

### Ý tưởng 1 — Thành viên thứ 4 **khác encoder** (U-Net EfficientNetV2-S, in21k) thay cho/trước N8 seed 2

- **Thay đổi cụ thể**: trong `colab/train_unet.ipynb` thêm
  `effb4v2s_s1024_f{k} = '--res 1024 --bs 6 --encoder tu-tf_efficientnetv2_s.in21k_ft_in1k --targets soft --epochs 30 --val-every 10 --fold k --ema 0.998'`
  (bs 4 trên T4), 5 fold; `RUN_NAMES` trỏ vào 5 run này **trước** `effb4_s1024_s2_f*`. Không đổi gì khác
  trong recipe (lr, loss, augmentation, target). Map OOF về Mac, infer 8 hướng @1024 như N1, rồi
  `compare.py` cross-tune `hyst_soft` trên 707 ảnh: {N1, N2, V2S} so với {N1, N2}.
- **Nguồn**: Wenzel 2020 (hyper-deep > deep ensembles); DiPE 2023 (ensemble đa backbone); Dariush
  (R34 + EffNetV2-S OOF 0.4502); **trong repo**: thành viên khác độ phân giải +0.0105 so với thành viên
  khác loss +0.0018 (§ 1).
- **Lý do chuyển được**: cùng dữ liệu, cùng metric, cùng pipeline; chỉ thay encoder — đúng kiểu thay đổi
  đã từng cho +0.013 (E9a). Diversity lớn hơn seed (kiến trúc stage đầu Fused-MBConv, pretrain in21k).
- **Đã thử chưa**: chưa. RESEARCH § 11.2 liệt kê "EfficientNetV2-S / B5: dự phòng, chỉ khi còn giờ sau
  rollout" — bây giờ chính là lúc đó.
- **Chi phí**: ~9.5 h Colab T4 (5 × 110–115 phút) hoặc ~3 h A100 — **một ngày Colab**. Mac: ~25 phút infer
  5 fold + ~30 phút compare.
- **Ngày**: bắt đầu 06/11 → map về 07/11 (A100: cùng ngày) → gate 07–08/11 → kịp đóng băng 09/11.
- **ΔOOF kỳ vọng**: **+0.002 … +0.006** trên 0.4284 (dưới ngưỡng +0.005 của gate "tốn GPU", nên chỉ áp
  gate thành viên như thẻ N8).
- **Rủi ro**: V2-S có thể kém B4 đơn lẻ ở input 1 kênh (gate thành viên vẫn cho vào nếu ensemble tăng,
  như N2 đã chứng minh); bs 4 trên T4 16 GB (hoạt hoá V2-S @1024 tương đương B4, dự kiến vừa — chạy smoke
  `--epochs 1` trước); timm tải weights từ HF hub trong Colab (cần mạng); ensemble 15 model làm notebook
  infer dài hơn (~+50%).
- **Gate (đăng ký trước)**: CI 95% > 0 của Δ PQ ghép cặp trên 707 ảnh OOF, cross-tune `hyst_soft`, ensemble
  {N1, N2, V2S} so với {N1, N2}; kiểm tra hợp lý: PQ fold của V2S trong ±0.02 so với N1 cùng fold (rộng hơn
  ±0.01 của N8 vì khác encoder). Không đạt → final 1 giữ 10 model.

### Ý tưởng 2 — Bật `--ema 0.998` trong **mọi** run Colab còn lại (0 giờ GPU thêm) để có ablation EMA miễn phí

- **Thay đổi**: thêm `--ema 0.998` vào các run của ý tưởng 1 và N8. `train_unet.py` khi có EMA lưu cả
  `model` (EMA) và `model_raw`, và log `raw_val_pq` mỗi lần val; chỉ cần infer thêm `model_raw` của 1–2 fold
  trên Mac (~5 phút/fold) để có Δ EMA − raw với CI.
- **Nguồn**: Izmailov et al. 2018 (SWA, UAI; CIFAR-100 +0.6 đến +2.0 điểm acc so với SGD, bảng
  torchcontrib); umerkang66 (EMA 0.998, S11b) và ranjeet258 (S12) dùng nhưng không có số tách riêng.
- **Trong repo**: v2 EMA −0.003 / −0.001 so với raw (val cc trong lúc train) → kỳ vọng ≈ 0.
- **ΔOOF kỳ vọng**: **+0.000 … +0.003**; giá trị chính là **câu trả lời cho report** ("EMA không giúp với
  OneCycle anneal về 0") với chi phí 0.
- **Rủi ro**: xáo trộn seed/encoder với EMA trong một run — không ảnh hưởng gate thành viên (gate so ensemble
  có/không thành viên); chỉ ảnh hưởng nếu muốn tách riêng "seed" — không cần.
- **Gate**: không có gate chấp nhận riêng; chỉ ghi Δ EMA − raw với CI trên 2 fold vào EXPERIMENTS. Nếu
  Δ < 0 với CI < 0 thì dùng `model_raw` làm thành viên (checkpoint vẫn chọn theo EMA, lệch nhẹ).

### Ý tưởng 3 — N8 seed 2 như đã duyệt, chạy **sau** ý tưởng 1 nếu còn ngày Colab thứ hai (07/11)

- **Thay đổi**: giữ nguyên thẻ N8 (`--seed 2`), thêm `--ema 0.998` (ý tưởng 2). Ngày Colab 07/11, map về
  08/11, gate 08/11.
- **Bằng chứng**: chỉ có F51 (+0.004 của BCE trên 2 fold, nhưng gate 707 ảnh chỉ +0.0018). Seed là mức
  khác biệt thấp nhất (§ 1).
- **ΔOOF kỳ vọng**: **+0.000 … +0.003**; nhiều khả năng không qua CI > 0. Nếu ý tưởng 1 qua gate, N8 phải
  được gate **trên nền có V2S** (ensemble 4 họ so với 3 họ), kỳ vọng còn thấp hơn.
- **Rủi ro**: tốn một ngày Colab cho kết quả gần như chắc chắn dưới ngưỡng phát hiện. Nếu chỉ có **một** ngày
  Colab chắc chắn, dùng cho ý tưởng 1.
- **Gate**: như thẻ N8.

Không đề xuất ý tưởng thứ 4 tốn GPU: mọi hướng khác (§ 4) đều có bằng chứng âm hoặc không có bằng chứng, và
ngân sách thật chỉ còn 2 ngày Colab có ích (06–07/11; 08/11 là ngày gate, 09/11 đóng băng).

## 4. Ngõ cụt (có bằng chứng, không chi GPU)

| Hướng | Vì sao bỏ |
|---|---|
| Pseudo-label / self-training trên ảnh test | biconcavelens −0.024 (S9e); HuBMAP 2021 "rất hạn chế"; model đã vượt inter-annotator nên pseudo-label không thêm thông tin; rủi ro luật S19 và rà soát thủ công S8; 139/180 ảnh test nằm trong MAGFiLO 1.0 (S1) |
| clDice / Skeleton Recall / loss component / BiCC-fpc | Hao & Li 2026 (3 seed, MAGFiLO): không ổn định; biconcavelens clDice −0.006; v2 fpc −0.05/−0.08 trong repo |
| Loss mô phỏng metric biên (Surface Dice, boundary loss kiểu SenNet) | metric PQ khớp IoU > 0.5, không phải metric mặt; lỗi biên là bất đồng annotator (F53); mọi chỉnh biên âm (F56) |
| Lấy mẫu ngẫu nhiên annotator thay target trung bình | Lemay 2022: random 50.6 so với trung bình 54.3 (MS lesion); Wang 2023 QUBIQ: 76.30 so với 77.31 |
| Trọng số theo chất lượng annotator / head theo annotator | không biết annotator test (S7); judge 30/10 loại (1.5 điểm) |
| Lịch dài / cosine restart / đổi LR | E10 −0.005; F35 phẳng từ epoch 25 |
| Augmentation nặng (scale, elastic, cutmix, blur, CLAHE) | copy-paste −0.015, CLAHE −0.006…−0.013 (S9e); F37, F40 |
| Pretrain tự giám sát trên GONG không nhãn | SimSiam 0.448 so với 0.449 (S9e), tốn ≥ 1 ngày Colab |
| ConvNeXt / SegFormer / Swin / FPN / DeepLab | thiếu skip stride 2 (F42), E9b −0.020 |
| Chuẩn hoá input kiểu ImageNet, bs lớn hơn cho BatchNorm (F50) | chưa có bằng chứng; đổi trong run mới sẽ xáo trộn với encoder; để report ghi là "chưa kiểm" |
| Distillation ensemble → một model (final 2 gọn) | không tăng OOF của final 1; chỉ có giá trị "đơn giản hoá" (S6b) nhưng tốn một ngày Colab và không còn ngày để gate |
| SWA theo checkpoint cuối | cần đổi lịch LR và lưu checkpoint từng epoch; EMA (ý tưởng 2) đã bao phủ với chi phí 0 |

## 5. Lịch đề nghị cho phần train (khớp PLAN § 5b)

| Ngày | Colab | Mac |
|---|---|---|
| 06/11 | Smoke `--epochs 1 --bs 4` V2-S trên T4 (10 phút), rồi 5 fold V2-S + EMA (ý tưởng 1 + 2) | sửa notebook/queue, đóng gói bundle lại; sẵn script gate |
| 07/11 | Nếu còn ngày Colab: N8 seed 2 + EMA (ý tưởng 3) | map V2-S về: infer 8 hướng, gate {N1,N2,V2S} so với {N1,N2} trên 707 OOF; ablation EMA − raw fold 0+1 |
| 08/11 | – | N8 về: gate trên nền đã có/không có V2S; chạy rubric N5 cho ứng viên |
| 09/11 | **Đóng băng** | chọn final 1/2 theo OOF |

Kỳ vọng trung thực cho toàn bộ phần train còn lại: **+0.002 … +0.008 OOF** (0.430–0.436), và LB nhiều
khả năng vẫn 0.37–0.38.

## 6. Nguồn đã đọc trong phiên

- Lemay et al. 2022, MELBA 2022:031 — https://www.melba-journal.org/papers/2022:031.html, https://arxiv.org/abs/2202.07550
- Wang et al. 2023, Dice Semimetric Losses — https://arxiv.org/abs/2303.16296 (HTML v4)
- Wenzel et al. 2020, Hyperparameter Ensembles — https://arxiv.org/abs/2006.13570
- Georgescu et al. 2023, DiPE — https://arxiv.org/abs/2210.12388
- Izmailov et al. 2018, SWA; bảng số ở https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
- SenNet + HOA 2024: danh sách write-up https://albumentations.ai/adoption/competitions/kaggle/blood-vessel-segmentation/ ; tổng hợp https://zenn.dev/yume_neko/articles/5bc998cfd8cd05 ; hạng 1 https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475522
- HuBMAP 2021: https://github.com/RobinSmits/KaggleHuBMAP ; https://hubmapconsortium.github.io/ccf/pages/kaggle.html
- Sartorius 2021: https://albumentations.ai/adoption/competitions/kaggle/sartorius-cell-instance-segmentation/
- Trong repo: RESEARCH § 5, 6.3, 11–13; EXPERIMENTS sổ cái; log `outputs/kaggle/filseg-gpu-*/*.log`
