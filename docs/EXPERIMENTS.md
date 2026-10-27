# Thí nghiệm

Mỗi thí nghiệm một mục: mục tiêu, cấu hình, kết quả, kết luận. Log gốc (đã lọc warning) nằm ở
[`docs/logs/`](logs/). Checkpoint, probability map và submission nằm trong `outputs/` (không
commit vì nặng); config đã tune nằm trong `configs/`.

**Quy ước đánh giá**: val = 30% số ảnh train (split cố định theo ảnh, seed 42) = 213 ảnh / 337
bản gán nhãn; PQ gộp trên mọi bản gán nhãn, đúng cách của host
(xem [DATA_AND_METRIC.md](DATA_AND_METRIC.md)). Chọn model và tham số theo val, không theo LB.

## Tổng hợp

| ID | Model | Độ phân giải | val PQ (thô) | val PQ (đã tune post) | LB | Config / log |
|---|---|---|---|---|---|---|
| E1 | Classical, tham số mặc định | 1024 (xử lý) | 0.05 | – | – | – |
| E2 | Classical + radial flatten + hysteresis | 1024 | 0.17 | – | – | – |
| E3 | Classical, random search | 1024 | – | 0.242 | 0.21 | `configs/classical_best.json`, `logs/search_r*.log` |
| E4 | U-Net ResNet34 | 512 | 0.337 | 0.354 | 0.29 | `configs/post_unet512.json`, `logs/train_unet512.log`, `logs/tune_512.log` |
| E5 | U-Net ResNet34 | 768 | 0.362 | **0.383** | **0.31** | `configs/post_unet768.json`, `logs/train_unet768.log`, `logs/tune_768.log` |
| E6 | Trung bình map E4 + E5 | 768 | – | 0.380 | – | `configs/post_ens_512_768.json`, `logs/tune_ens.log` |
| E5w | E5 + post-processing watershed | 768 | – | 0.384 | – | `configs/post_unet768_ws.json`, `logs/tune_768_watershed.log` |
| E5f | E5 + lọc FP theo instance (GBM, CV 5 fold trên val) | 768 | – | 0.389 | **0.33** | `configs/fpfilter_unet768.pkl`, `logs/fpfilter_768.log` |
| E7 | U-Net ResNet34 (Colab L4) | 1024 | 0.357 | 0.391 → 0.396 (lọc FP) | 0.33–0.34 | `configs/post_unet1024_r34.json`, `configs/fpfilter_unet1024_r34.pkl` |
| E5+E7 | Trung bình map 768 + 1024 | 1024 | – | 0.394 → 0.399 (lọc FP) | **0.34** (0.33 không lọc) | `configs/post_ens_768_1024r34.json`, `configs/fpfilter_ens_768_1024r34.pkl` |
| E8 | U-Net R34 + kênh group | 1024 | 0.364 | 0.394 (cc; mode group 0.393) → 0.399 (lọc FP) | 0.34 | Colab `unet1024_r34_grp` |
| E9a | U-Net EfficientNet-B4 | 1024 | **0.381** | 0.404 → **0.408** (lọc FP) | **0.36** | Colab `unet1024_effb4` |
| E9b | U-Net ConvNeXt-Tiny (lr 5e-4) | 1024 | 0.342 | 0.384 → 0.392 | 0.33 | Colab `unet1024_cnxt` |
| E10 | R34, 60 epoch | 1024 | 0.364 | 0.399 → 0.402 | 0.34 | Colab `unet1024_r34_e60` |
| E11 | Model cuối, train trên toàn bộ ảnh | 1024 | sau E7–E10 | | | Colab `unet1024_final_full` |

## E1–E3: Classical baseline

`src/filseg/classical.py`, không học gì từ dữ liệu.

Pipeline: fit đĩa mặt trời (Otsu + minEnclosingCircle) → chia cho profile median theo bán kính
(limb darkening) → chia cho median-blur lớn → hysteresis threshold trên tỉ lệ tối → morphology →
bỏ blob nhỏ và sunspot (vùng rất tối, nhỏ) → gom mảnh bằng dilation → upsample 2048².

- E1 (threshold đơn giản): FP rất nhiều ở rìa đĩa, mask mỏng hơn GT nhiều. PQ 0.05.
- E2 (flatten theo bán kính + hysteresis): PQ 0.17.
- E3: random search cục bộ (đổi 1–3 tham số mỗi vòng), 80 + 60 vòng trên 160 ảnh. Bão hoà ở
  ~0.245. Lỗi chính: FP (fibril tối nhỏ) 1,441 so với TP 878 trên val.
- Chạy 13 s cho 180 ảnh test trên CPU.
- Lưu ý: search chạy trên split cũ (chia theo bản gán nhãn, có rò rỉ); val 0.242 là con số đo lại
  trên split đúng.

**Kết luận**: threshold cố định không phân biệt được filament với fibril/vùng tối khác; cần học.

## E4: U-Net ResNet34 @512

- `scripts/train_unet.py --res 512 --epochs 30` (bs 8, AdamW 1e-3, OneCycle, BCE + Dice), encoder
  ImageNet, input 1 kênh, augment rot90/flip/gamma/contrast. Target = hợp các mask của một bản gán
  nhãn (mỗi annotator là một sample).
- Mac M5 MPS: ~170 s/epoch, tổng ~1.5 h.
- Instance = connected components của map xác suất (threshold → bỏ blob nhỏ → gom mảnh bằng
  dilation `group_radius`).
- Tune post (192 tổ hợp): thresh 0.4, min_area 25, group_radius 4 → 0.354 (thô 0.337).

## E5: U-Net ResNet34 @768

- Giống E4, `--res 768 --bs 6`. MPS ~300 s/epoch, tổng ~2.5 h. PQ val tăng đều đến epoch 30
  (chưa bão hoà, có thể train lâu hơn).
- Tune post: thresh 0.4, min_area 50, group_radius 4 → **0.383**.
- Còn lại trên val: TP 1,385 · FP 965 · FN 981 · gt_split 107 · pred_merge 86.

**Kết luận**: độ phân giải cao hơn tăng ~0.03 PQ, nên thử 1024 (E7).

## E6: Ensemble 512 + 768

Trung bình map xác suất (resize lên 768) → 0.380, kém E5 dùng một mình. Model 512 làm mờ chi tiết
mảnh. Ensemble chỉ đáng thử giữa các model cùng độ phân giải hoặc khác kiến trúc.

## E5w: Watershed thay cho gom mảnh bằng dilation

`PostParams(mode="watershed")`: seed = pixel có xác suất > `seed_thresh`, gom seed bằng dilation,
mỗi pixel foreground nhận seed theo watershed trên −prob; thành phần không có seed bị bỏ.
Tune 144 tổ hợp: tốt nhất 0.3837 (thresh 0.4, seed 0.9, group_radius 8, min_area 50), so với cc là
0.3831.

**Kết luận**: watershed giảm số filament bị gộp (pred_merge 86 → 50) nhưng cắt đôi nhiều hơn
(gt_split 107 → 172), tổng gần như không đổi. Tách instance bằng luật cố định chỉ đổi lỗi này lấy
lỗi kia, nên cần cho model **học** cách gom (E8).

## E5f: Lọc FP theo instance

`src/filseg/instfilter.py`, `scripts/fp_filter.py`. Mỗi instance (cc, config E5) có 14 đặc trưng:
log diện tích, xác suất trung bình/max/p25, tỉ lệ pixel > 0.8, extent, độ thuôn, chiều dài trục
chính, độ tương phản so với vành xung quanh, độ sáng, khoảng cách tới tâm đĩa, số instance trong
ảnh, thứ hạng diện tích, giá trị group map (nếu có). Nhãn: khớp IoU > 0.5 với ít nhất một
annotator. HistGradientBoosting, CV 5 fold theo ảnh trên **val** (net không thấy val khi train),
quét ngưỡng trên điểm out-of-fold.

- 1,477 instance, 66% dương. Ngưỡng 0.225: PQ 0.3831 → **0.3893** (FP 965 → 753, TP 1,385 →
  1,321).
- Submit → **LB 0.33** (0.31 khi không lọc). Trên LB lọc giúp nhiều hơn trên val.
- Model cuối fit trên toàn bộ instance val, lưu ở `configs/fpfilter_unet768.pkl`; dùng với
  `make_submission.py --filter`.

**Kết luận**: giữ lại. Mỗi model mới cần chạy lại `fp_filter.py` trên map val của chính nó.

## E7–E11: Hàng đợi trên Colab

**Kết quả train (session L4, 2026-10-24 18:48–23:30 UTC)**. Val PQ "thô" = cc mặc định, không
TTA, đo trong lúc train (5 epoch/lần):

| Run | Thời gian | Epoch | val PQ thô tốt nhất | Ghi chú |
|---|---|---|---|---|
| `unet1024_r34` | 37 min | 30 | 0.357 | 1 min/epoch, val ~1 min |
| `unet1024_r34_grp` | 36 min | 30 | 0.364 | kênh group giúp nhẹ ngay cả khi đánh giá ở mode cc |
| `unet1024_effb4` | 61 min | 30 | **0.381** | 1.8 min/epoch; tốt nhất |
| `unet1024_cnxt` | 48 min | 30 | 0.342 | hội tụ chậm và dao động (val 0.21–0.34), lr 5e-4 có thể quá thấp/cao |
| `unet1024_r34_e60` | ~1 h 10 | 60 | 0.364 tại ep 45 | so với 0.357 sau 30 epoch: thêm epoch giúp ít |

So sánh: E5 (R34 @768 trên Mac) có val PQ thô 0.362. R34 @1024 (0.357) **không** hơn 768;
cải thiện đến từ encoder (EfficientNet-B4) và kênh group.

**Sự cố**:
- Upload `unet1024_r34` lỗi: slug `filseg-unet1024-r34` đã bị một notebook chiếm (lần push kernel
  Kaggle thất bại hôm 22/10), nhưng `kaggle datasets create` vẫn thoát với mã 0 nên notebook tưởng
  là đã xong. Đã sửa `scripts/kaggle_upload.py` để báo lỗi; run này dùng slug
  `filseg-unet1024-r34-ds` (`SLUG_OVERRIDES` trong watcher).
- Kaggle trả 404 khi tải nguyên dataset các run này (tải từng file vẫn được) → watcher đã đổi sang
  tải từng file, 16 luồng song song.

`colab/train_unet.ipynb`, dict `EXPERIMENTS` trong ô parameters; `RUN_NAMES` mặc định chạy cả
E7–E10 trong **một session** (~8 h A100 / ~16 h L4). Mỗi run xong được upload ngay lên
`nguyenchicuong/filseg-<run>` (dấu `_` đổi thành `-`).

**Theo dõi trên Colab**: ô train hiện log trực tiếp. Mỗi epoch có 4 dòng tiến độ, một dòng tổng
kết (loss, lr, thời gian, ETA của run và giờ dự kiến xong theo UTC), và cứ 5 epoch có val
PQ/SQ/Dice/TP/FP/FN. Bảng hàng đợi (trạng thái, số epoch, best PQ, thời gian đã dùng, thời gian
còn lại, giờ xong cả hàng đợi) in ra lúc bắt đầu, sau mỗi 5 epoch và sau mỗi run. Log và
`history.csv` của từng run nằm trên Drive (`MyDrive/filseg/runs/`); mục cuối notebook vẽ biểu đồ
loss / val PQ / val Dice.

**Tự động submit**: `scripts/watch_colab.py` chạy nền trên máy local, cứ 10 phút kiểm tra Kaggle;
run nào có dataset thì tải về, tune post-processing (thêm mode `group` nếu có group map), train FP
filter, tạo CSV và submit ngay. Khi có từ 2 run trở lên, nó thử trung bình map của các run; chỉ
submit ensemble nếu val cao hơn mọi run đơn. Trạng thái nằm ở `outputs/colab/watch_state.json`,
log tune/filter ở `docs/logs/`. Lượt submit bị từ chối do giới hạn 5/ngày sẽ được thử lại ở các lần
kiểm tra sau.

```bash
.venv/bin/python scripts/watch_colab.py --runs unet1024_r34 unet1024_r34_grp unet1024_effb4 unet1024_cnxt unet1024_r34_e60
```

Các bước watcher làm, nếu cần chạy tay:

```bash
kaggle datasets download nguyenchicuong/filseg-<run> -p outputs/colab/<run> --unzip
.venv/bin/python scripts/tune_post.py --probs outputs/colab/<run>/probs_val --out configs/post_<run>.json   # E8: --mode group
.venv/bin/python scripts/fp_filter.py --probs outputs/colab/<run>/probs_val --post configs/post_<run>.json --out configs/fpfilter_<run>.pkl
.venv/bin/python scripts/make_submission.py --probs outputs/colab/<run>/probs_test --filter configs/fpfilter_<run>.pkl --out outputs/submission_<run>.csv
```

- **E7** `unet1024_r34`: R34 @1024, so sánh với E5 để xem độ phân giải còn giúp không.
- **E8** `unet1024_r34_grp`: thêm kênh group (mask mỗi filament dilate ~6 px ở thang 1024, bỏ phần
  chồng giữa hai filament). Post-processing `--mode group`: instance = vùng liên thông của group map.
- **E9** `unet1024_effb4` (bs 6), `unet1024_cnxt` (ConvNeXt-Tiny, bs 6, lr 5e-4).
- **E10** `unet1024_r34_e60`: 60 epoch.
- **E11** `unet1024_final_full`: công thức tốt nhất của E7–E10 với `--split all`; không còn val nên
  dùng lại post config và FP filter của run tương ứng.
- Ensemble (#6): `scripts/infer_probs.py --ckpt a.pt b.pt` (cùng độ phân giải) hoặc
  `scripts/ensemble_probs.py`, rồi tune và lọc như trên.

## E12–E14: Hàng đợi Colab thứ hai — 5-fold, target mềm, crop ở độ phân giải gốc (R1–R3)

`colab/train_unet.ipynb`, `RUN_NAMES` mặc định (11 run, ~12–14 h trên L4):

| Run | Công thức | Mục đích |
|---|---|---|
| `r34_s1024_f0` | R34, target mềm, cả ảnh @1024, fold 0 | đối chứng: chỉ đổi sang target mềm |
| `r34_c2048_f0` | R34, target mềm, crop 1024 từ ảnh 2048, infer full 2048, fold 0 | so với dòng trên: tác dụng của độ phân giải gốc |
| `effb4_c2048_f0..f4` | EfficientNet-B4, target mềm, crop @2048, bs 6 | công thức chính, 5 fold |
| `r34_c2048_f1..f4` | R34, như `r34_c2048_f0` | thành viên thứ hai của ensemble |

- **Fold** (`filseg.data.fold_split`): 707 ảnh chia 5 fold theo ảnh (seed 42), khoảng 141 ảnh mỗi
  fold. Run `--fold k` train trên 4 fold còn lại và ghi map cho các ảnh của fold k. Ghép 5 fold lại
  được map out-of-fold (OOF) cho cả 707 ảnh.
- **Target mềm** (`SoftDataset`): mỗi ảnh là một sample; target của pixel = tỉ lệ annotator vẽ pixel
  đó. Ảnh được bốc theo trọng số bằng số bản gán nhãn (giống cách metric tính).
- **Crop** 1024 từ ảnh 2048: 70% crop lấy tâm quanh một pixel filament. Validation và infer chạy trên
  cả ảnh 2048 (mạng fully convolutional).
- **Watcher**: với mỗi recipe, fold 0 được tune và submit riêng để có tín hiệu sớm. Khi đủ 5 fold,
  watcher ghép map OOF, tune trên toàn bộ 707 ảnh (grid 36 điểm, scale theo độ phân giải), train
  bộ lọc FP trên OOF, submit trung bình map test của 5 fold. Khi có từ 2 recipe trở lên, nó thử
  ensemble các recipe trên OOF.

## R4–R5: Tách instance và chọn instance trên map có sẵn (ensemble 768 + 1024, val 213 ảnh)

| Cách | val PQ | Ghi chú |
|---|---|---|
| cc (gốc) | 0.3943 | thresh 0.5, min_area 90, group 6 |
| cc + lấp lỗ | 0.3944 | map gần như không có lỗ, nên không đổi |
| seed + gộp seed qua khoảng hở + watershed + lấp lỗ | 0.3943 | gộp ít hơn (81 → 80) nhưng tách nhiều hơn (106 → 120) |
| + bộ lọc classifier, 14 đặc trưng (cũ) | 0.3992 | |
| + bộ lọc classifier, 17 đặc trưng (thêm vị trí trên đĩa / kinh độ) | **0.4009** | dùng từ giờ |
| + regressor IoU kỳ vọng ("luật PQ/2") | 0.3986 | kém classifier; ngưỡng tốt nhất 0.087, xa mức PQ/2 ≈ 0.2 |

**Kết luận**: R4 không giúp trên map của mình (umerkang66 được +0.008, có lẽ vì model của họ cho map
khác). R5 chỉ giữ phần đặc trưng vị trí (+0.002); regressor IoU kém hơn. Log: `docs/logs/r4_*`,
`docs/logs/r5_*`. Lợi ích lớn hơn phải đến từ chất lượng map (R1–R3).

## Tự phản biện và kiểm chứng (2026-10-25) — `scripts/compare.py`

**Công cụ**: `scripts/compare.py` so sánh các pipeline trên cùng ảnh val bằng **bootstrap ghép cặp
theo ảnh** (2,000 lần lấy mẫu lại) và cho CI 95% của chênh lệch.
- Bộ lọc FP được fit lại 5 fold, không dùng bản đã fit trên chính các ảnh đó.
- Spec `tune:<mode>` là **cross-tune**: tune grid trên một nửa số ảnh, chấm trên nửa còn lại, đổi
  vai, lặp 3 lần chia ngẫu nhiên. Cách này loại thiên lệch do tune và chấm trên cùng một tập.
- Log: `docs/logs/critique_*.txt`.

**Kiểm tra lại các kết luận cũ** (holdout 213 ảnh):

| So sánh | Δ PQ | 95% CI | Kết luận |
|---|---|---|---|
| EffNet-B4 so với R34 @1024 | +0.013 | [+0.003, +0.023] | thật |
| EffNet-B4 + lọc FP so với không lọc | +0.004 | [0.000, +0.008] | thật nhưng nhỏ |
| R34 60 epoch so với EffNet-B4 | −0.005 | [−0.014, +0.005] | không phân biệt được |
| ensemble 6 model holdout (cross-tune) | −0.001 | [−0.009, +0.007] | **không giúp**: trộn model yếu làm loãng model mạnh |
| tune trên chính tập chấm so với cross-tune (EffNet-B4 cc) | 0.4042 so với 0.4015 | – | tune in-sample lạc quan ~0.003 |

**Val có dự đoán được LB không?** Qua 12 submission: Spearman 0.92, LB ≈ 0.84 × val. Riêng các U-Net
≥ 768: Spearman 0.81, LB thấp hơn val trung bình 0.059. Val vẫn dùng để xếp hạng được, nhưng các
chênh lệch < 0.01 không thể thấy trên LB (vì LB làm tròn 2 chữ số và nhiễu ±0.03).

**Nhãn test có thưa hơn không?** Model dự đoán trên test và val như nhau: 6.57 so với 6.37 instance
mỗi ảnh, foreground 3.25‰ so với 3.18‰. Vậy khoảng cách LB đến từ phía nhãn. Trên val, chia
annotator theo mật độ (trung vị 6.9 filament/ảnh): **giữ instance càng chặt thì PQ càng tăng ở cả
hai nhóm**, và tăng nhiều hơn ở nhóm thưa (+0.010 so với +0.006, seed 0.6 → 0.99).

**Ý tưởng mới, đã kiểm chứng bằng cross-tune** (EffNet-B4, so với cc cross-tune 0.4015):

| Ý tưởng | PQ | Δ | 95% CI | Ghi chú |
|---|---|---|---|---|
| **Hysteresis chặt** (`mode=hyst`: biên ở p > 0.5, giữ nếu đỉnh > 0.99) | 0.4083 | **+0.007** | [+0.002, +0.011] | FP 838 → 753, TP giữ nguyên |
| Chỉnh mask theo kích thước (instance < 200 px chỉ giữ p > 0.7) | 0.4064 | +0.005 | [+0.002, +0.008] | mask nhỏ dự đoán to hơn GT 27%, mask lớn nhỏ hơn 13% |
| Hysteresis + chỉnh theo kích thước | 0.4091 | +0.008 | [+0.002, +0.013] | **không cộng dồn**; tune trên toàn val tự tắt phần kích thước |
| Hysteresis + lọc FP (in-sample) | 0.4108 | +0.0005 so với hyst | – | lọc FP thừa khi đã có hysteresis |
| Hysteresis so với cc + lọc FP đang dùng | +0.002 | [−0.004, +0.008] | – | hai cách loại FP tương đương |

**Trên LB**: EfficientNet-B4 + hysteresis chặt được **0.36**, bằng cc + lọc FP. Kết quả này khớp
với dự báo (+0.002, không có ý nghĩa thống kê).

**Bài học**:
1. Thắng lợi từ hậu xử lý đã gần cạn: mọi cách loại FP chỉ cho +0.004 đến +0.008 và không cộng dồn.
2. Ensemble chỉ có ích khi các thành viên mạnh ngang nhau. Hàng đợi fold (cùng EffNet-B4) là
   hướng đúng.
3. Watcher giờ tune cả `hyst` và `cc` trên map fold/OOF rồi chọn cái tốt hơn.

**LB của các biến thể (2026-10-27)**: EffNet-B4 + hysteresis + co mask nhỏ 0.36; + min_area 150 0.35;
ensemble 6 + hysteresis 0.35; + co mask nhỏ 0.35. Không có gì vượt 0.36, đúng như dự báo từ CI.
Hậu xử lý trên map 1024 hiện tại đã chạm trần (~0.36 public).
