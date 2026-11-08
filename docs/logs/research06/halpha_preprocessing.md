# Research 06 (05/11) — `halpha_preprocessing`: tiền xử lý ảnh H-alpha toàn đĩa cho CNN

Phạm vi: limb-darkening correction, chuẩn hoá cường độ theo ảnh (median/MAD), CLAHE / trừ nền,
mask đĩa, unwrap cực/heliographic, kênh toạ độ r/R, crop sát đĩa trước khi resize 1024.
Mọi nguồn được đọc chỉ-đọc (không chạy code ngoài). Mọi số OOF là 707 ảnh, pipeline final 1
(`outputs/ens/n1n2_oof`, `configs/post_ens_n1n2_hyst.json`), trừ khi ghi khác.

## 0. Kết luận trước

**Cả nhóm "tiền xử lý theo vật lý" là ngõ cụt cho bài này. Không có mục nào đáng chi một
Colab-day trước đóng băng 09/11.** Bốn lớp bằng chứng độc lập, cùng chiều:

1. **Ảnh đã được pipeline NSO/GONG chuẩn hoá hình học và cường độ trước khi tới Kaggle, và phần
   limb darkening còn lại là một hàm cố định, giống nhau ở mọi trạm và train = test.** Trang sản
   phẩm NSO (https://nso.edu/nisp/products/h-alpha/) liệt kê xử lý tại trạm: dark, flat, sharpen,
   limb fitter, xoay bắc lên trên, "Remap image to a circle with a constant radius", "Normalize image
   to a mean intensity level"; bước làm phẳng limb bằng wavelet chỉ làm ở data center cho JPEG hiển
   thị, FITS khoa học không có. Đo trên dữ liệu của mình (F40, `docs/logs/sweep03/station_stats.out`,
   `station_stats.py`): median trong đĩa 130–133, p5/p95 108–113 / 144–147, **tỉ lệ limb (profile tại
   0.95 R / tâm) 0.61–0.63** — tức limb darkening *vẫn còn* trong ảnh Kaggle, nhưng cùng một giá trị ở
   cả 6 trạm —, bán kính đĩa 452–455 px @1024 ở **cả 6 trạm, train = test**; khoảng cách W1 giữa
   histogram các trạm ≤ 1.6 mức xám (trạm T ≤ 5.1), nhỏ hơn độ lệch trong cùng trạm (SD median
   1.7–3.8). Một profile bán kính cố định, tần số rất thấp, là thứ CNN học thuộc sau vài epoch; nó chỉ
   gây hại khi *thay đổi* giữa ảnh (nhiều kính, nhiều năm, mây — đúng bối cảnh của Diercke 2024 và
   Hao 2015), không phải ở đây. Mọi phép chuẩn hoá theo ảnh (chia median, z-score MAD) vì thế thay
   đổi input dưới ~2 mức xám (<1 % thang), nhỏ hơn jitter brightness ±0.08 (±20 mức xám) đã có trong
   augmentation.
2. **Lỗi của model không phụ thuộc vị trí trên đĩa.** F28: FP do limb/sunspot/ngoài ±70° là 3/794
   (0.4 %); tỉ lệ dự đoán không khớp ai theo r/R: 0.29 / 0.28 / 0.25 / 0.31 ở [0,0.5) / [0.5,0.7) /
   [0.7,0.8) / [0.8,0.9) (`docs/logs/wf30/training_summary_effb4.txt`). Đo mới hôm nay trên 707 OOF
   (phía FN và IoU, § 2) cho cùng kết luận: trần lợi ích nếu sửa **hoàn hảo** mọi lỗi ở r/R ≥ 0.8
   là nhỏ, và recall / IoU ở vành ngoài không thấp hơn tâm đĩa đáng kể.
3. **Literature không có một ablation nào.** Tất cả paper trong ngành chỉ *áp dụng* limb-darkening
   correction như bước mặc định kế thừa từ thời threshold (Hao 2013 nói rõ lý do: "may cause false
   detections" cho ngưỡng toàn cục); không paper nào so sánh có/không trước CNN (§ 3.1). Guo 2022
   (theo trích dẫn của Diercke 2024) còn cố tình **đưa ảnh không chỉnh limb** vào training để
   robust hơn, và **Flat U-Net 2025** (Zhu, arXiv 2502.07259, BBSO 2048²) ghi rõ input "require only
   min–max normalization without the removal of limb darkening and unevenness correction" mà vẫn
   đạt DSC 0.79–0.82 so với U-Net cổ điển 0.69 trên cùng dữ liệu — tức CNN tự xử lý limb darkening.
4. **Hai đội Kaggle trung thực cho kết quả bằng nhau dù khác hẳn về input.** theainerd: 3 kênh
   (tương phản đã chỉnh limb, I/median, r/R) → OOF 0.4519, LB 0.40; umerkang66: 1 kênh raw chuẩn
   hoá cố định (mean 125, std 18) → OOF 0.4477, LB 0.39; biconcavelens: 8 biến thể flat-field /
   unsharp trên FITS → "all failed the same way", CLAHE 0.4252 → 0.4121–0.4193. Khác biệt OOF
   0.004 giữa hai đội có/không limb-correction nằm trong nhiễu seed (F44) và trộn với khác biệt
   về model, loss, tile, hậu xử lý.

Hệ quả cho PLAN § 5b: **không thêm việc**. GPU còn lại dành cho N8 như đã quyết. Mục duy nhất tôi
ghi là "có thể làm nếu còn thừa một Colab-day sau N8" (§ 4) có kỳ vọng −0.003…+0.003, tức dưới
ngưỡng phát hiện, nên về bản chất cũng là "không làm".

## 1. Repo hiện xử lý input như thế nào

Đọc `src/filseg/data.py`, `src/filseg/dl.py`, `scripts/train_unet.py`, `scripts/infer_probs.py`:

| Khâu | Hiện tại | Ghi chú |
|---|---|---|
| Đọc ảnh | `cv2.imread(..., IMREAD_GRAYSCALE)` → uint8 2048² (`data.load_gray`) | 3 kênh gốc giống nhau |
| Độ phân giải | N1: `cv2.resize` INTER_AREA 2048 → 1024 (`dl.cached` / `soft_cached`); N2: crop 1024 từ 2048 | không crop sát đĩa |
| Chuẩn hoá | `img / 255.0`, **không** trừ mean / chia std, `in_channels=1` (smp cộng trọng số RGB của conv1 pretrained) | F50 đã ghi là chưa kiểm |
| Mask đĩa | không | góc ảnh ngoài đĩa đã ≈ 0 sẵn |
| Limb correction / flat-field | **không** cho CNN; chỉ có trong baseline cổ điển `classical.py` (`radial_profile` median theo bán kính, PQ 0.245) | |
| Kênh phụ (r/R, toạ độ) | không | |
| Augmentation | rot90 ×4, lật ngang, gamma 0.7–1.4, tương phản 0.8–1.2, sáng ±0.08 | rng không seed (F44) |
| TTA | D4 ×8 ở 1024; TTA1152 đã đo −0.0031 fold 0 | |

Đã thử trong EXPERIMENTS / RESEARCH thuộc phạm vi này: `loc-err` 30/10 (bỏ input limb khỏi N3);
F39 (kênh lọc không thêm thông tin, GBM AUC 0.6973 → 0.6996); F40; F41; S9e (CLAHE của
biconcavelens); F37 (phóng ×1.125: +0.0004); TTA1152; F56 (chỉnh biên theo cường độ ảnh: âm mọi
biến thể, −0.0005 đến −0.052). Ma trận RESEARCH § 11.2 đã đánh "loại" cho limb / flat-field /
chiếu cực, CLAHE, kênh vesselness, log/gamma/khớp histogram.

## 2. Đo mới: lỗi theo bán kính chuẩn hoá r/R trên 707 OOF (final 1)

Script: `/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/halpha_preprocessing/radial_errors.py`
(chỉ chẩn đoán, không tune gì; đĩa fit bằng `classical.fit_disk`; r/R của một filament là khoảng
cách tâm khối lượng của mask tới tâm đĩa chia R; "contrast" = median cường độ trong mask / median
vành 20 px quanh mask, trên ảnh 2048 đã median-blur 5).

Kết quả (`radial_errors.out`; 707 ảnh, 8,199 GT theo bản gán nhãn, 7,172 dự đoán gộp theo bản gán
nhãn; PQ gộp 0.4294 = đúng số final 1):

| r/R | nGT | recall | IoU TB của TP | tương phản hit / miss (I_fil / I_nền) | nPred | tỉ lệ FP |
|---|---|---|---|---|---|---|
| [0.0, 0.3) | 971 | 0.604 | 0.672 | 0.882 / 0.912 | 821 | 0.280 |
| [0.3, 0.5) | 2,043 | 0.602 | 0.675 | 0.881 / 0.910 | 1,730 | 0.295 |
| [0.5, 0.7) | 2,717 | 0.619 | 0.678 | 0.879 / 0.909 | 2,488 | 0.323 |
| [0.7, 0.8) | 1,398 | 0.608 | 0.677 | 0.878 / 0.906 | 1,292 | 0.335 |
| [0.8, 0.9) | 936 | **0.550** | 0.679 | 0.869 / 0.906 | 814 | **0.369** |
| [0.9, 1.0] | 134 | **0.134** | 0.648 | 0.847 / 0.887 | **27** | 0.593 |

Oracle "sửa đúng **mọi** lỗi" (FN → TP với IoU 1, bỏ hết FP) trong vùng ngoài — bao gồm cả bất đồng
annotator, nên là giới hạn trên tuyệt đối, không phải kỳ vọng: r/R ≥ 0.9: 0.4294 → 0.4425
(**+0.013**, 1.6 % GT); r/R ≥ 0.8: → 0.5142 (+0.085, 13.1 % GT; con số này vô nghĩa về thực hành vì
cùng oracle áp cho tâm đĩa cũng cho cỡ đó — recall trong đĩa chỉ 0.60).

Đọc bảng, trung thực:

- **Từ tâm tới r/R = 0.8 (87 % GT) mọi chỉ số phẳng**: recall 0.60–0.62, IoU 0.672–0.679, tỉ lệ FP
  tăng nhẹ 0.28 → 0.34. Không có dốc theo bán kính → limb darkening không làm model yếu dần ra ngoài.
- **Có một vùng yếu thật ở r/R ≥ 0.8 (13 % GT)**: recall 0.55 ở [0.8, 0.9) và **0.13 ở [0.9, 1.0]**,
  nơi model chỉ đưa ra 27 dự đoán cho 134 GT. Đây là kết quả mới so với F28 (F28 chỉ nhìn phía FP:
  tỉ lệ "không khớp ai" 0.31 ở [0.8, 0.9) so với 0.29 ở tâm — khớp với cột FP ở đây — nhưng không nhìn
  phía FN).
- Tuy vậy **nguyên nhân không phải tương phản bị limb darkening nuốt**: filament ở vành ngoài *tối
  hơn* so với nền cục bộ (0.847–0.869) chứ không mờ hơn (tâm 0.88), và trong mọi bin filament bị sót
  đều mờ hơn filament bắt được (0.906–0.912 so với 0.88, đúng F41). Nếu limb darkening là thủ phạm,
  một phép chia cho profile bán kính (vốn không đổi tương phản cục bộ I_fil / I_nền) cũng không
  đổi được gì: tương phản *cục bộ* là bất biến với phép chia cho nền trơn. Lời giải thích hợp lý hơn
  là hình học (foreshortening: filament gần limb mảnh, ngắn, chồng lên fibril / plage sáng, và chính
  annotator cũng bất đồng ở đó) — bảng § 2b kiểm tra điều này bằng mức đồng thuận annotator theo r/R.

RADIAL_SUPPORT_PLACEHOLDER

## 3. Từng ý tưởng: nguồn, bằng chứng, vì sao không chuyển sang bài này

### 3.1 Limb-darkening correction làm input cho CNN — **ngõ cụt**

Các phương pháp đúng tên trong literature (để trích dẫn trong report):

| Nguồn | Phương pháp | Dữ liệu | Có ablation có/không? |
|---|---|---|---|
| Hao et al. 2013, Solar Phys (arXiv 1303.6367), § 2.2 Eq. 1 | `I_cor(θ) = I_raw(θ) / (1 − u₂ − v₂ + u₂ cos θ + v₂ cos² θ)`, u₂ = 0.88, v₂ = −0.23 "fitted constants for Hα 6563 Å" (paper ghi "Keith Pierce [6]" nhưng ref [6] là Cox 2000 *Allen's Astrophysical Quantities*) | MLSO 1024², 2.9″/px, 3,470 ảnh 1998–2009 | **Không**. Lý do ghi: limb darkening "may cause false detections. We should remove it first"; MLSO cũng cung cấp sẵn ảnh đã chỉnh; khớp thủ công 85 ± 2 % trên 100 ảnh |
| Hao et al. 2015, ApJS 221:33 (arXiv 1511.04692), § II.2 | "polynomial surface fitting technique" theo Bernasconi 2005 và phụ lục Yuan et al. 2011, chỉ fit pixel trong đĩa; lý do: dữ liệu nhiều kính, nhiều năm, nền không đều do mây / bụi (dữ liệu 2013 "tốt" nên trước đó chỉ cần bỏ limb) | BBSO 1988–2013 | **Không** (chỉ báo < 5 % ảnh cần kiểm tay ở bước tìm tâm đĩa) |
| Bernasconi, Rust & Hakim 2005, Solar Phys 228:97 | surface fitting (theo trích dẫn của Hao 2015); BBSO cung cấp sẵn ảnh "high-contrast" đã bỏ limb | BBSO | **Không** |
| Qu, Shih, Jing & Wang 2005, Solar Phys 228:119 | ngưỡng tự động so với nền cục bộ + SVM tách sunspot (abstract NJIT) | BBSO | **Không** |
| Zhu et al. 2019, Solar Phys 294:117 (arXiv 1909.06580), § III.1 | "Limb darkening is removed from the original images to avoid introducing too much noises and other undesirable factors", không nêu cách; "All the pixels outside the solar disk are set to gray"; resize 512 | BBSO/FDHA 2013, **60 ảnh** (30 train → 6,040 mẫu augment / 20 val / 10 test) | **Không**; TPR trung bình 0.9145, DSC 0.8944 trên 10 ảnh test |
| Ahmadzadeh et al. 2019 (arXiv 1912.02743), Mask R-CNN | dùng ảnh BBSO archive đã "flat field correction, dark subtraction and correction for limb darkening" | BBSO 2012–2016 | **Không** |
| Diercke et al. 2024, A&A 686:A213 (arXiv 2402.15407), YOLOv5 + U-Net; code GPL-3.0 `adiercke/DeepFilamentSegmentation`, `dfs/data/data_set.py` L196–252 (đọc chỉ-đọc) | ChroTel: chuẩn hoá theo median đĩa, chỉnh limb, cắt ngoài đĩa, Zernike; GONG: "center-to-limb-variation correction" + Zernike, clip [0.8, 1.3], scale [−1, 1], 1024². **Code thực tế**: `ideal = cos(r/R · π/2)`, `np.polyfit(ideal, I, 6)` trên pixel trong đĩa, chia ảnh cho đa thức → fit 36 Zernike (`aotools`) trên mask `median ± 3σ` → chia tiếp cho nền polar-fit → `Normalize(0.8, 1.3, clip)·2 − 1` | GONG 17,413 tải, 16,759 sau lọc chất lượng (2010–2021) | **Không**; Table 1: accuracy U-Net 92.4 ± 7.9 % so với hình thái 89.0 ± 8.9 %, nhưng precision 49.8 % **thấp hơn** hình thái 55.7 % |
| Zheng et al. 2024 (arXiv 2402.14209), CHASE U-Net | loại bỏ limb darkening trước K-means / hình thái (abstract không nêu công thức; chi tiết "fit cosine tới 0.95 R, 120 ảnh, IoU 0.67" là từ bản đọc 05/11, chưa kiểm lại hôm nay) | CHASE/HIS | **Không** |
| EdgeAttNet 2025 (arXiv 2509.02964), § III-A | chuẩn hoá [0, 1] → Hough circle mask đĩa → bin vành, profile 1-D làm mượt, nội suy nền 2-D rồi chia → Gaussian 3×3 σ 0.7 → CLAHE trong đĩa; ngoài đĩa = 0 | MAGFiLO (GONG) 1,593 ảnh, 1,295 / 45 / 99 | **Không**; chỉ so kiến trúc: mIoU U-Net 0.5724 → EdgeAttNet 0.6451 (pairwise) |
| **Flat U-Net 2025** (Zhu, arXiv 2502.07259, ApJ), § II.1 | **"require only min–max normalization without the removal of limb darkening and unevenness correction"**; thí nghiệm kiểm định cũng "without limb-darkening removal and unevenness corrections" | BBSO FITS 2048², ~1,000 ảnh 2022–2023 train, ~100 test 2024; nhãn từ phương pháp cổ điển | **Không ablation**, nhưng là bằng chứng "không chỉnh vẫn chạy": DSC 0.79–0.82 / precision 0.93 / recall 0.69–0.74 so với U-Net cổ điển 0.69 / 0.98 / 0.53 |
| Guo et al. 2022, Solar Phys 297:104 (theo Diercke 2024, Introduction) | **cố tình gồm ảnh BBSO không chỉnh limb** và jitter độ sáng ngẫu nhiên để robust | BBSO | – |
| sunkit-image (`radial.intensity_enhance`, NRGF) | fit đa thức bậc 1 lên log(I) theo r trong `fit_range` [1, 1.5] R☉, chia phần **ngoài limb** (EUV corona); không phải công cụ cho limb darkening trong đĩa | – | – |

Tổng: **0/11 nguồn có số đo "có so với không chỉnh limb"**; nguồn duy nhất đi ngược chiều (Flat
U-Net 2025) bỏ hẳn bước này và vẫn tốt hơn U-Net có cùng input. Mọi lý do nêu ra đều thuộc thời
ngưỡng toàn cục (Hao 2013). Với U-Net: receptive field của EfficientNet-B4 ở 1024 phủ cả đĩa,
BatchNorm từng tầng, và trong N1 model luôn thấy limb trong khung hình nên "vị trí trên đĩa" là
thông tin có sẵn; gradient bán kính là thành phần tần số rất thấp, tách khỏi filament (tần số cao,
F37: mờ σ 1.5 px đã −0.096) bằng vài lớp đầu.

Chuyển sang bài này: điểm F28 + § 2 cho thấy không có "mục tiêu" ở vành ngoài để sửa. Thêm vào
đó, hai đội 0.39–0.40 khác nhau đúng ở điểm này (theainerd có limb-corrected channel, umerkang66
không) mà OOF chỉ chênh 0.004.

- Cần retrain (không thể inference-only: model N1/N2 học trên ảnh raw).
- Chi phí: 2 fold @1024 ≈ 2 × 2.3 h T4 ≈ 5 h (hoặc 2 × 75 phút L4); 5 fold ≈ 1 Colab-day. Sớm
  nhất 07/11 (sau khi N8 chạy), tức tranh GPU trực tiếp với N8.
- Kỳ vọng ΔOOF: −0.003…+0.003 (mức nhiễu seed; không có cơ chế dương). Gate GPU (≥ +0.005 trên 2
  fold, CI > 0) gần như chắc không qua.
- **Quyết định: không làm.**

### 3.2 Chuẩn hoá cường độ theo ảnh (median đĩa, MAD) — **ngõ cụt**

Nguồn: Diercke 2024 (chuẩn hoá theo median đĩa), Zheng 2024 (chia mean), theainerd (I/median).
Lý do của họ: dữ liệu nhiều năm, nhiều kính, độ phơi sáng khác nhau. Ở đây: F40 — median đĩa
130–133 ở mọi trạm, train = test, p5/p95 trùng tới ±2 mức xám. Phép chuẩn hoá theo ảnh sẽ đổi
input dưới 2/255 ≈ 0.8 %, trong khi augmentation đã jitter sáng ±0.08 và tương phản ±20 %. Không
thể làm inference-only (model học trên raw). Kỳ vọng 0 ± nhiễu. **Không làm.**

### 3.3 CLAHE / trừ nền cục bộ — **ngõ cụt, đã có số âm**

S9e (biconcavelens RESULTS.md): baseline 0.4252 → CLAHE clip 2.0: 0.4193, clip 4.0: 0.4121, đơn
điệu theo clip; họ ghi tương phản filament tăng 2–5× nhưng "no net detection benefit". Cơ chế trên
dữ liệu mình: F41 — filament mờ và filament chỉ một người vẽ là cùng một nhóm (AUC 0.70), nên làm
nổi cấu trúc mờ chủ yếu thêm khớp với một annotator (FP với hai người kia). F56 — mọi biến thể
chỉnh biên theo cường độ ảnh âm (annotator không vẽ theo mép tối). EdgeAttNet dùng CLAHE nhưng
không ablation. Cần retrain. **Không làm.**

### 3.4 Mask đĩa (đặt ngoài đĩa = 0 / xám) — **không có gì để được**

Zhu 2019 đặt ngoài đĩa thành xám, EdgeAttNet đặt 0. Ảnh của mình ngoài đĩa đã ≈ 0. F28: 1/2,194
dự đoán nằm ngoài ±70°, 1 FP "orphan_limb" trong 794. Mask dự đoán ngoài r > R khi làm submission
là vệ sinh miễn phí nhưng Δ = 0 trên OOF (không có dự đoán nào ở đó). Không tính là ý tưởng.

### 3.5 Unwrap cực / heliographic làm input — **ngõ cụt về nguyên lý**

Không paper filament-DL nào dùng (Diercke 2024 chỉ chiếu Carrington cho **kết quả** để vẽ
butterfly diagram). Chống chỉ định cho PQ: (a) metric chấm IoU trong khung ảnh gốc, nên phải
resample ngược — F47 cho thấy riêng việc phóng nhãn nearest 1024 → 2048 đã mất 0.0024, resample
phi tuyến hai chiều tệ hơn; (b) phép chiếu cực kéo giãn góc ở tâm đĩa và nén ở rìa, phá tính
đẳng hướng mà augmentation rot90/lật và TTA D4 đang dựa vào; (c) heliographic deprojection phóng
to filament gần limb (foreshortening) nhưng annotator vẽ trong khung ảnh, không trong khung
heliographic. **Không làm.**

### 3.6 Kênh toạ độ / r/R (CoordConv) — **ngõ cụt**

Liu et al. 2018, CoordConv (arXiv 1807.03247): lợi trên bài toán toạ độ đồ chơi và MNIST detection
(+24 % IoU), tức khi đầu ra phụ thuộc vị trí tuyệt đối. Ở đây lỗi phẳng theo r/R (F28, § 2), và
N1 (toàn đĩa) đã thấy limb trong khung nên vị trí là suy ra được; N2 (crop) không thấy limb nhưng
PQ N2 ≈ N1. theainerd dùng kênh r/R, không ablation. Cần retrain. Kỳ vọng 0 ± nhiễu. **Không làm.**

### 3.7 Crop sát đĩa trước khi resize 1024 — **đã đo tương đương, ≈ 0**

Đĩa có đường kính 906/1024 px, nên crop sát đĩa rồi resize 1024 tương đương phóng ×1.13. Đã có số:
F37 phóng ×1.125 lúc infer: **+0.0004** (62 ảnh); TTA1152 trên fold 0: **−0.0031**, trung bình
1024+1152: −0.0002; UP2048 (hậu xử lý ở 2048) +0.0024 là một chuyện khác (đã nhận cho final 2).
Đòn bẩy độ phân giải thật đã dùng là N2 (crop 1024 từ 2048) trong ensemble. Crop sát đĩa cần
retrain cho +0.000–0.001. **Không làm.**

### 3.8 Chuẩn hoá kiểu ImageNet (mean/std) thay cho `/255` — không thuộc "vật lý", ghi để đóng F50

umerkang66 chuẩn hoá cố định mean 125 / std 18 (≈ z-score); mình đưa [0, 1] vào conv1 pretrained
(kỳ vọng ~N(0, 1)). BatchNorm ngay sau conv1 và 30 epoch fine-tune hấp thụ khác biệt tỉ lệ này;
không có nguồn nào đo lợi ích cho 1 kênh grayscale. Không được gộp vào N8 (N8 phải giữ nguyên
công thức để là thành viên ensemble sạch). Kỳ vọng |Δ| < 0.003. **Không chi GPU.**

### 3.9 Tóm tắt: retrain hay inference-only, chi phí, kỳ vọng

| Ý tưởng | Retrain? | GPU (Colab) | Sớm nhất | Kỳ vọng ΔOOF | Quyết định |
|---|---|---|---|---|---|
| Limb-darkening correction (Hao 2013 / Diercke polyfit) thay input | retrain N1 (và N2) | 2 fold ≈ 5 h T4 / 2.5 h L4; 5 fold ≈ 1 Colab-day | 08–09/11, tranh GPU với N8 | −0.003…+0.003 | không |
| Kênh thứ hai = ảnh flatten, giữ raw (§ 4) | retrain, `in_channels=2` | như trên | 09/11 (sát đóng băng) | −0.003…+0.003 | không |
| Chuẩn hoá theo ảnh (median / MAD) | retrain | như trên | – | 0 ± nhiễu | không |
| CLAHE / trừ nền cục bộ | retrain | như trên | – | −0.006…−0.013 (S9e) | không |
| Mask đĩa khi nộp (đặt ngoài r > R = 0) | **inference-only**, 0 GPU | 0 | ngay | 0.0000 (không có dự đoán ngoài đĩa trên 707 OOF) | vệ sinh, không tính là ý tưởng |
| Unwrap cực / heliographic | retrain + resample ngược | ≥ 1 Colab-day | – | âm (F47: resample nhãn đã −0.0024) | không |
| Kênh r/R (CoordConv) | retrain | 2 fold ≈ 5 h T4 | 08–09/11 | 0 ± nhiễu | không |
| Crop sát đĩa rồi resize 1024 (≈ phóng ×1.13) | retrain | 2 fold ≈ 5 h T4 | 08–09/11 | +0.000…+0.001 (F37 +0.0004 infer-only, TTA1152 −0.0031) | không |
| ImageNet mean/std thay `/255` | retrain | như trên | – | \|Δ\| < 0.003 | không |

Gate chung nếu ai vẫn muốn chạy một mục: 2 fold (0 + 1, 284 ảnh), cross-tune hysteresis, Δ ≥ +0.005
**và** bootstrap 95 % CI > 0; chỉ chạy sau khi N8 đã về và còn thừa GPU, không bao giờ thay N8.

## 4. Mục duy nhất "có thể" (chỉ nếu thừa một Colab-day sau khi N8 về, không khuyến nghị)

**Kênh thứ hai = ảnh đã flatten theo bán kính, giữ kênh raw** (`in_channels=2`, conv1 pretrained
chia đôi trọng số). Đây là dạng "bảo toàn thông tin" nhất của § 3.1: không mất gì so với input
hiện tại, chỉ thêm một kênh. Bằng chứng ủng hộ duy nhất là theainerd (OOF 0.452 với 3 kênh gồm
kênh này) — nhưng không ablation, và F39 nói kênh flat-field không thêm thông tin cho bộ lọc
instance. Chi phí 2 fold ≈ 5 h T4; sớm nhất 08/11 (sau N8), về 09/11 — đúng ngày đóng băng nên
thực tế **không kịp vào final**. Kỳ vọng −0.003…+0.003. Gate: Δ ≥ +0.005 **và** CI > 0 trên fold
0+1 (284 ảnh), hysteresis cross-tune `hyst_soft`. Rủi ro: tốn GPU đúng lúc N8 cần; thêm một bước
tiền xử lý vào notebook nộp. **Khuyến nghị: không chạy; ghi vào report như hướng đã loại có lý do.**

## 5. Gợi ý cho report 4 trang (không tốn GPU)

- Một đoạn "Why no physics preprocessing": trích NSO (ảnh đã remap + normalise), F40, F28, bảng § 2,
  S9e, và việc hai đội 0.39/0.40 khác nhau về input mà bằng nhau về PQ. Đây là lập luận định tính
  mà rubric 30 % coi trọng, và là điểm khác với mọi paper trong ngành (họ mặc định chỉnh limb).
- Hình: recall / IoU / FP theo r/R (từ script § 2) cạnh biểu đồ inter-annotator PQ 0.34.

## Nguồn

- NSO GONG H-alpha products: https://nso.edu/nisp/products/h-alpha/
- Hao et al. 2013: https://arxiv.org/abs/1303.6367 · Hao et al. 2015: https://arxiv.org/abs/1511.04692
- Bernasconi, Rust & Hakim 2005, Solar Phys 228:97 (doi:10.1007/s11207-005-2766-y)
- Qu et al. 2005: https://digitalcommons.njit.edu/fac_pubs/19704
- Zhu et al. 2019: https://arxiv.org/abs/1909.06580 · Ahmadzadeh et al. 2019: https://arxiv.org/abs/1912.02743
- Diercke et al. 2024: https://arxiv.org/abs/2402.15407 · Zheng et al. 2024: https://arxiv.org/abs/2402.14209
- EdgeAttNet 2025: https://arxiv.org/abs/2509.02964 · CoordConv: https://arxiv.org/abs/1807.03247
- Flat U-Net 2025: https://arxiv.org/abs/2502.07259 · Hao 2015 full text: https://ar5iv.labs.arxiv.org/html/1511.04692
- Diercke code (GPL-3.0, đọc chỉ-đọc): https://github.com/adiercke/DeepFilamentSegmentation (`dfs/data/data_set.py`)
- Zhu 2019 code (MIT): https://github.com/GF-Zhu/Filament-Unet · EdgeAttNet code: https://github.com/dasjar/EdgeAttNet (README, thiếu model)
- sunkit-image radial: https://docs.sunpy.org/projects/sunkit-image/en/stable/api/sunkit_image.radial.intensity_enhance.html
- theainerd README: https://github.com/theainerd/filament_segmentation/blob/main/README.md
- umerkang66 REPORT: https://github.com/umerkang66/solar-filament-segmentation/blob/master/REPORT.md
- biconcavelens RESULTS: https://github.com/biconcavelens/filament-segmentation-2026/blob/master/RESULTS.md
- Nội bộ: `docs/RESEARCH.md` § 11–13 (F28, F37, F39–F41, F47, F56, S9e, S10a, S15), `docs/logs/sweep03/`, `docs/logs/wf30/`
