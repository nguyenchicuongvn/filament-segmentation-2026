# research06 / ensemble_diversity — vắt thêm gì từ 10 (sắp 15–20) model fold, và một seed mới đáng bao nhiêu

Ngày 05/11/2026. Phạm vi: không train lại từ đầu; chỉ cách kết hợp các checkpoint đã có trong
`outputs/colab/`, cách đo, và giá trị thật của N8 (seed 2, cùng công thức). Mọi số của mình là OOF
707 ảnh, cross-tune `hyst_soft`, bootstrap ghép cặp 2 000 lần (`scripts/compare.py`); không phải LB.

## 0. Kết luận ngắn

1. **N8 (seed 2, cùng công thức N1) đáng rất ít**: dự báo **+0.0005…+0.002 OOF**, gần như **0 trên
   test/LB**, và với gate hiện tại (CI > 0 trên 707 ảnh, half-width ≈ 0.0027) **nhiều khả năng không
   qua**, giống BCE (+0.0018 [−0.0008, +0.0045]). Lý do (§ 2): lợi của thành viên thứ k tỉ lệ
   1/(k−1) − 1/k, và trên test N1 đã là trung bình 5 fold nên "seed thứ 2" thực ra là bước 5 → 10 model
   cùng công thức, nhỏ hơn bước 1 → 2 mà OOF đo được ~5 lần. Contrails 1st place đo 2-fold so với
   5-fold cùng công thức: lệch 0.0001 private.
2. **Việc rẻ duy nhất còn đáng làm trong phạm vi này** là *cách gate*: gate **gộp** N8 + BCE như một
   khối (4 công thức, 20 model) so với N1+N2 trên 707 OOF, đăng ký trước ngay 06/11 (§ 3.1). Hai thành
   viên "miễn phí" mỗi cái +0.002 cộng lại mới đủ lớn để CI tách khỏi 0; nhận một thành viên đã train
   sẵn có kỳ vọng Δ ≥ 0 không có rủi ro test. Kỳ vọng +0.002…+0.004 OOF, +0.00…+0.01 LB (không phân biệt
   được).
3. **Chết, có bằng chứng** (§ 4): model soup/weight averaging giữa fold (không thể kiểm chứng OOF —
   mọi model fold khác đều đã nhìn ảnh val) và giữa seed (decoder khởi tạo khác → rào cản permutation);
   SWA/checkpoint averaging (không có checkpoint trung gian nào trên đĩa, `.pt` = epoch 30 ở cả 5 fold,
   lr OneCycle về 4e-9; EMA ≈ raw trong chính run v2 của mình); trung bình logit (map uint8 bão hoà,
   N1 có 72 % thành phần đỉnh ≥ 0.99 sẽ lấn N2); tối ưu trọng số thành viên (đang chạy, kỳ vọng
   |Δ| < 0.002 — "forecast combination puzzle"); train lại toàn bộ dữ liệu (không gate được, lợi
   80 % → 100 % dữ liệu nhỏ theo bằng chứng Contrails).
4. **Nếu bundle N8 chưa chạy**: cùng 6.3 h L4 đó, một thành viên ở **độ phân giải thứ ba** (ví dụ
   N2-recipe crop 1024 từ ảnh resize 1536) có kỳ vọng cao hơn seed 2 (§ 3.2), vì trong dữ liệu của mình
   đa dạng độ phân giải (+0.0105) ≫ đa dạng loss (+0.0018) > đa dạng seed (dự báo ≤ +0.002). Đây là
   quyết định của người dùng; rủi ro là công thức mới chưa thử.

## 1. Hiện trạng trong repo (đã kiểm tra)

| Mục | Thực tế |
|---|---|
| Checkpoint có trên đĩa | `outputs/colab/<run>_f<k>/<run>_f<k>.pt` cho N1 (`effb4_s1024`), N2 (`effb4_c2048`), BCE (`effb4_bce_s1024`), v2 (f0, f1): mỗi file 81.7 MB, **chỉ một checkpoint/fold**. Không có `.last.pt`, không có checkpoint trung gian |
| `.pt` là epoch nào | `train_unet.py` lưu "best" theo val PQ mỗi `--val-every 5`; history của cả 5 fold N1 tăng đơn điệu 10 → 20 → 30 (f0 0.3664/0.3863/0.3918, f1 0.3587/0.3726/0.3766, f2 0.3479/0.3857/0.3908, f3 0.3675/0.3803/0.3872, f4 0.3635/0.3791/0.3813) nên **best = epoch 30 = last**. OneCycleLR, lr epoch 30 = 4.06e-9 |
| EMA | N1/N2/BCE không có EMA (`--ema 0`); chỉ v2 có 0.998, và ở v2 PQ raw ≈ EMA (0.317/0.292, EXPERIMENTS G2) |
| Map | `probs_val/*.npy` uint8 (0–255) ở độ phân giải dự đoán, **đã trung bình 8 hướng TTA trong không gian xác suất** (`infer_probs.py` dòng 70). `ensemble_probs.py` trung bình có trọng số trong không gian xác suất, resize lên độ phân giải lớn nhất, làm tròn uint8 |
| OOF đã đo | N1 0.4179; N2 0.4129 (−0.0050 [−0.0098, −0.0003]); N1+N2 **0.4284** (+0.0105 [+0.0072, +0.0140], dương 5/5 fold); N1+N2+BCE 0.4302 (+0.0018 [−0.0008, +0.0045], P>0 0.91) |
| Đang chạy | `docs/logs/research05/ens_weights_oof.txt` (N1:N2 = 2:1 và 1:2), `up2048_oof.txt` (chưa ra số lúc viết) |
| Seed | `torch.manual_seed(args.seed)`, mặc định 0 → **5 fold N1 dùng cùng khởi tạo decoder**, khác dữ liệu; N8 (`--seed 2`) khởi tạo decoder khác |
| Chi phí | compare.py cross-tune: 8.4 phút/213 ảnh @1024 (PROGRESS P1) → ~30 phút/707 ảnh @1024, lâu hơn ở 2048; infer Mac 1 model TTA 8: ~7 phút/142 ảnh @1152 |
| N8 | bundle đóng gói 05/11, chưa chạy; ~75 phút/fold L4 (6.3 h), 2.3 h/fold T4; cần về trước 08/11 |

## 2. Một seed mới đáng bao nhiêu — ước lượng từ chính dữ liệu của mình và từ tài liệu

### 2.1 Mô hình lợi ích giảm dần

Với M thành viên có lỗi tương quan ρ, phương sai của trung bình là σ²(ρ + (1−ρ)/M); lợi khi thêm
thành viên thứ M+1 tỉ lệ (1−ρ)(1/M − 1/(M+1)): bước 1 → 2 = 0.50, 2 → 3 = 0.17, 3 → 4 = 0.083,
5 → 10 = 0.10 (cho cả 5 thành viên). Dạng này đúng với mọi số đo ensemble đã công bố:

- Deep Ensembles (Lakshminarayanan 2017, [arXiv 1612.01474](https://arxiv.org/abs/1612.01474), Fig. 4
  ImageNet): top-1 error M=1 22.17 → M=2 20.46 → M=5 19.10 → M=10 18.68. Tức 1 → 2: −1.70 điểm;
  2 → 5: −0.45/thành viên; 5 → 10: **−0.09/thành viên**.
- Lobacheva 2020 "On power laws in deep ensembles" ([arXiv 2007.08483](https://arxiv.org/abs/2007.08483)):
  NLL theo số mạng là luật luỹ thừa c + b·n^a với a ≈ −1; b (toàn bộ lợi có thể có) **giảm khi mạng
  lớn hơn** — EfficientNet-B4 U-Net là mạng lớn trên 566 ảnh train.
- Deep ensemble phân đoạn (SegDepthFormer, [arXiv 2405.17097](https://arxiv.org/pdf/2405.17097)): mIoU
  74.35 với 2 thành viên → 76.25 với **20** thành viên, chất lượng dừng tăng rõ sau ~10–12.
- Contrails 1st place (Jun Koda, [github.com/junkoda/kaggle_contrails_solution](https://github.com/junkoda/kaggle_contrails_solution)):
  cùng công thức, 2-fold (private 0.72432) so với 5-fold (0.72420) — "only differ by random fluctuation".
  Đây là bằng chứng gần nhất với câu hỏi "thêm model cùng công thức trên test": từ 2 lên 5 model/kiến
  trúc, **0.0001**.

### 2.2 Áp vào số của mình

- Bước 1 → 2 với thành viên **khác độ phân giải** (N2): +0.0105. Theo tỉ lệ 0.17/0.50, thành viên thứ
  3 cùng mức đa dạng đó cho tối đa ≈ +0.0035; BCE (khác loss, cùng độ phân giải, ρ cao hơn) đo được
  **+0.0018** — khớp.
- N8 là thành viên thứ 4 trên OOF nếu BCE vào (0.083/0.17 × +0.0018 ≈ +0.0009), hoặc thứ 3 nếu BCE
  không vào (≤ +0.0018, nhưng ρ(N8, N1) > ρ(BCE, N1) vì cùng loss, cùng res, cùng dữ liệu, chỉ khác
  khởi tạo và thứ tự sampler). **Dự báo OOF: +0.0005…+0.002**, dưới half-width CI 0.0027 → gate đơn lẻ
  nhiều khả năng không qua.
- **Trên test, lợi còn nhỏ hơn**: ở OOF mỗi ảnh chỉ được 1 model N1 (fold của nó), nên "thêm N8" là bước
  1 → 2 trong nội bộ công thức N1 (hệ số 0.50). Trên test, N1 đã là trung bình 5 fold; thêm N8 là bước
  5 → 10 (hệ số 0.10). Vậy **OOF đánh giá cao lợi test của một seed cùng công thức khoảng 5 lần**;
  đa dạng kiểu N2 (khác độ phân giải) không bị phóng đại như vậy vì cả hai công thức đều 5 fold trên
  test. Dự báo test cho N8: ≈ +0.0002…+0.0005, không nhìn thấy ở LB 2 chữ số.
- Thứ hạng đa dạng đo được trong repo: **độ phân giải (+0.0105) ≫ loss (+0.0018) > seed (dự báo
  ≤ +0.002)**; thành viên yếu hẳn (v2, −0.05 đơn lẻ) không ổn định: +0.011 fold 0 nhưng +0.0005 fold 1.

### 2.3 Thời gian inference của notebook

Thêm 5 model @1024 TTA 8 cho ~180 ảnh test ≈ +15 phút T4; 20 model (N1, N2, BCE, N8) ≈ 2–2.5 h (N2 ở
2048 nặng gấp ~4). Vẫn trong giới hạn kernel Kaggle 9 h, nhưng cần ghi vào kiểm tra tái lập 11/11.

## 3. Đề xuất (ít, có gate cụ thể)

### 3.1 Gate gộp cho thành viên "miễn phí": N1+N2 **+BCE+N8** so với N1+N2 — đăng ký trước 06/11

- **Nguồn**: lý thuyết 1/M ở § 2.1 (Lakshminarayanan 2017; Lobacheva 2020) + số của mình: hai thành
  viên mỗi cái ≈ +0.002, CI half-width 0.0027.
- **Việc làm**: khi N8 về (08/11), `ensemble_probs.py` 4 thư mục theo fold (1:1:1:1), rồi
  `compare.py base=n1n2_oof,tune:hyst_soft pool=n1n2bce_s2_oof,tune:hyst_soft` trên 707 ảnh. Thêm hai
  spec phụ để tách đóng góp: `n1n2bce_oof` (đã có, +0.0018) và `n1n2s2_oof`. Mac ~1.5 h (tạo map + 3
  cross-tune ở 2048; chạy `--jobs 6`).
- **Vì sao hợp lệ**: thành viên đã train, chi phí thêm là inference; theo lý thuyết, thêm một model
  ngang chất lượng vào trung bình có kỳ vọng Δ ≥ 0, nên **sai nhận** gần như không có giá trên test,
  còn **sai bỏ** mất ~+0.002–0.004 OOF. Gate đơn lẻ CI > 0 đang tối ưu cho rủi ro sai nhận (hợp với
  hậu xử lý, nơi Δ có thể âm) chứ không cho tình huống này.
- **Gate (đăng ký trước, ghi vào PLAN § 4 thẻ N8 ngay 06/11)**: nhận cả khối N8+BCE vào final 1 nếu
  (a) Δ(pool so với N1+N2) CI 95 % > 0 trên 707 OOF, **và** (b) mỗi thành viên có đóng góp biên P>0 ≥ 0.8
  (pool so với pool-bớt-thành-viên-đó), **và** (c) PQ fold của N8 nằm trong ±0.01 so với N1 cùng fold
  (kiểm tra hợp lý đã có). Nếu (a) qua mà (b) hỏng với một thành viên thì chỉ nhận thành viên kia theo
  gate cũ. Không đổi gì nếu N8 không về trước 08/11.
- **Kỳ vọng**: ΔOOF +0.002…+0.004 (pool), xác suất qua (a) ≈ 50–60 %. LB: không phân biệt.
- **Rủi ro**: 20 model làm notebook tái lập chậm (§ 2.3); gate gộp nhận cả thành viên có Δ ≈ 0 (vô hại).
- **Ngày**: script + chạy thử trên N1+N2+BCE (để chắc pipeline) 06/11; gate thật 08/11; đóng băng 09/11.

### 3.2 (Quyết định của người dùng) Dùng ngày Colab của N8 cho một thành viên ở độ phân giải thứ ba thay cho seed 2

- **Nguồn**: số của mình (N1+N2 +0.0105 là hiệu ứng lớn duy nhất đo được; TTA theo tỉ lệ của *cùng*
  model thì âm −0.0031, tức lợi đến từ model *được train* ở tỉ lệ khác, không từ việc đổi tỉ lệ lúc
  infer); nguyên tắc chung "đa dạng > số lượng" trong ensemble (Lakshminarayanan 2017 § 3; Contrails
  1st place kết hợp U-Net + ViT thay vì nhiều fold hơn, 2-fold = 5-fold).
- **Việc làm**: `effb4_c1536_f0..f4` = công thức N2 (`--targets soft --crop 1024`) nhưng `--res 1536`
  (crop 1024 từ ảnh resize 1536). Cùng số pixel/bước với N1/N2 → cùng ~75 phút/fold L4, 6.3 h cho 5
  fold, vừa một ngày Colab. Infer test ở 1536 TTA 8 rồi resize lên 2048 trong `ensemble_probs.py`
  (đã hỗ trợ resize).
- **Vì sao chuyển**: thành viên thứ 3 ở độ phân giải mới có trần ≈ +0.0035 (1/3 của +0.0105, § 2.2) so
  với seed 2 ≤ +0.002; trên test không bị phóng đại 5 lần như seed.
- **Gate**: giống thẻ N8 (CI > 0 trên 707 OOF, kiểm tra hợp lý từng fold ±0.01 so với N1), hoặc gate
  gộp § 3.1 với BCE.
- **Kỳ vọng**: ΔOOF +0.001…+0.005. **Rủi ro**: công thức chưa chạy (N2 đơn lẻ kém N1 −0.005 với +643 FP;
  1536 có thể nằm giữa), không còn ngày Colab để sửa nếu hỏng; chỉ làm khi bundle N8 chưa được upload.
- **Ngày**: đóng gói lại bundle 06/11, chạy 06–07/11, về 08/11.

### 3.3 Để các run đang chạy kết thúc, không thêm biến thể trọng số

`ens_weights_oof.txt` (2:1, 1:2) sẽ trả lời câu hỏi trọng số. Dự báo |Δ| < 0.002, CI chứa 0: với 2
thành viên và ~700 ảnh, trọng số bằng nhau rất khó thắng ("forecast combination puzzle", Smith &
Wallis 2009; tổng quan [arXiv 2205.04216](https://arxiv.org/pdf/2205.04216): lợi của trọng số tối ưu
thường nhỏ hơn sai số ước lượng chúng). Không làm trọng số theo fold hay theo kích thước instance — đó
là thêm nút hậu xử lý, đã cạn (RESEARCH § 13).

## 4. Hướng chết (có bằng chứng) — không làm

| Hướng | Vì sao chết | Nguồn |
|---|---|---|
| **Model soup giữa các fold** (trung bình trọng số 5 model N1 thành 1 model) | Không thể kiểm chứng: model fold k được train trên 4 fold còn lại, nên **mọi** model khác đều đã nhìn ảnh val của fold k; soup của bất kỳ ≥ 2 fold không có ảnh held-out nào. Trái quy tắc "quyết định bằng OOF". Ngoài ra lợi của soup vốn ≤ ensemble output (Wortsman 2022 Fig. 1: soup xấp xỉ ensemble, không vượt) | Wortsman et al. 2022 [arXiv 2203.05482](https://arxiv.org/abs/2203.05482) |
| **Soup giữa seed trong cùng fold** (N1 f_k + N8 f_k) | Soup cần cùng khởi tạo để nằm cùng "basin" (Wortsman 2022; Neyshabur 2020 [arXiv 2008.11687](https://arxiv.org/abs/2008.11687)). Ở đây encoder chung ImageNet nhưng **decoder U-Net khởi tạo ngẫu nhiên khác** (seed 0 so với 2) và train từ đầu với lr 1e-3 → đối xứng hoán vị tạo rào cản loss khi nội suy tuyến tính (Entezari 2021 [arXiv 2110.06296](https://arxiv.org/abs/2110.06296); Ainsworth 2022 Git Re-Basin [arXiv 2209.04836](https://arxiv.org/abs/2209.04836) cần căn chỉnh hoán vị mới hết rào cản). Nếu tò mò: test 20 phút Mac trên fold 0 (trung bình state_dict, `infer_probs.py` 142 ảnh, compare với N1 f0); gate CI > 0; kỳ vọng âm | như trên |
| **SWA / checkpoint averaging epoch cuối** | Không có checkpoint trung gian trên đĩa (§ 1); `.pt` = epoch 30 ở cả 5 fold; lr OneCycle về 4e-9 → "khi lr ủ về 0, weight averaging hội tụ về chính model đã ủ" ([arXiv 2502.06761](https://arxiv.org/abs/2502.06761), Conformer/LibriSpeech); bằng chứng của mình: v2 raw ≈ EMA 0.998. TGS Salt 9th place: SWA +0.004 pixel accuracy, nhưng trên lịch lr không ủ hết. Muốn thử phải train lại với lưu checkpoint — không trong 4 ngày | Izmailov 2018; [arXiv 2502.06761](https://arxiv.org/abs/2502.06761); EXPERIMENTS G2 |
| **Thêm `--ema 0.998` vào N8** để có so sánh EMA/raw "miễn phí" | EMA horizon ≈ 500 bước ≈ 3.5 epoch trong khi lr đã ≈ 0 ở 5 epoch cuối → EMA ≈ raw (đúng như v2). Phải đóng gói lại bundle. Kỳ vọng 0 | EXPERIMENTS G2 |
| **Trung bình logit thay xác suất** | Map lưu uint8 đã trung bình TTA trong không gian xác suất (không hoàn tác được); 0/255 → logit ±∞ phải kẹp; N1 bão hoà (72 % thành phần đỉnh ≥ 0.99, F51) sẽ lấn N2/BCE trong không gian logit — ngược với lý do ensemble N1+N2 có lợi (N2 giảm FP). Tài liệu: logit averaging tăng độ tự tin, lợi cho model **under-confident** ([TUM, impact of averaging logits over probabilities](https://portal.fis.tum.de/en/publications/the-impact-of-averaging-logits-over-probabilities-on-ensembles-of/)); hysteresis với seed tune được đã trung hoà hiệu chỉnh (F51). Nếu Mac rảnh: 1 h test trên 707 OOF, gate CI > 0, kỳ vọng −0.002…+0.001 | F51; TUM |
| **Bỏ fold yếu / trọng số theo fold** | Trên OOF mỗi ảnh chỉ có 1 model fold → không đo được việc chọn fold; trên test 5 fold PQ in-train lệch nhau ≤ 0.015 và đều nằm trong nhiễu | history.csv N1 |
| **Train lại toàn bộ dữ liệu (E11)** | Không gate được bằng OOF (không có ảnh held-out), nên không vào final 1 theo quy tắc. Chọn epoch thì dễ (best luôn = 30) và winner có làm (CZII cryo-ET 1st place: 2 "fullfit" `fold -1` mỗi model, [MONAI tutorial](https://github.404890603.workers.dev/garciadias/tutorials/blob/68e0dcf51b1b4eae1ff2947759c2a5797adc7ed0/competitions/kaggle/Cryo-ET/1st_place_solution/README.md)), nhưng lợi 80 % → 100 % dữ liệu nhỏ: Contrails 2-fold (50 %) = 5-fold (80 %) trong 0.0001. Chi phí 707/566 × 2.3 h ≈ 2.9 h T4/model. Chỉ cân nhắc làm **final 2** (ô dự phòng không cần gate) nếu còn Colab sau N8 — ưu tiên thấp | Koda README; MONAI |
| **TTA theo tỉ lệ / 1152** | Đã đo âm (−0.0031 fold 0) | EXPERIMENTS TTA1152 f0 |

## 5. Lịch đề nghị cho phạm vi này

| Ngày | Việc | Lượt nộp liên quan |
|---|---|---|
| 06/11 | Ghi gate gộp § 3.1 vào PLAN § 4 (thẻ N8); viết `scripts/research/pool_gate.sh` (ensemble 4 thư mục theo fold + 3 compare); chạy thử trên N1+N2+BCE để chắc pipeline và thời gian. Nếu người dùng chọn § 3.2: đóng gói lại bundle Colab | như lịch 5b (UP2048, trọng số) |
| 07/11 | Chờ N8; không thêm thí nghiệm trọng số sau khi `ens_weights_oof` ra số | theo 5b |
| 08/11 | N8 về → gate gộp; bảng đóng góp biên từng thành viên cho report | 1. N1+N2+BCE+N8 (20 model) · 2. N1+N2+N8 · 3–5. theo 5b |
| 09/11 | Đóng băng; final 1 theo kết quả gate | final 1 / final 2 |

## 6. Cho report (phân tích giới hạn)

Bảng "số thành viên so với OOF": 1 (N1) 0.4179 → 2 (+N2) 0.4284 → 3 (+BCE) 0.4302 → 4 (+N8) dự báo
≤ 0.432, đặt cạnh đường cong ImageNet của Deep Ensembles và inter-annotator 0.34: ensemble đã vào vùng
bão hoà, phần còn lại của 1 − PQ là bất đồng annotator, không phải phương sai model.
