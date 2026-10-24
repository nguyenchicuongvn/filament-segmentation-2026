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
| E7 | U-Net ResNet34 (Colab) | 1024 | đang chạy (người dùng) | | | Colab `unet1024_r34` |
| E8 | U-Net R34 + kênh group | 1024 | chờ Colab | | | Colab `unet1024_r34_grp` |
| E9 | EfficientNet-B4 / ConvNeXt-Tiny | 1024 | chờ Colab | | | Colab `unet1024_effb4`, `unet1024_cnxt` |
| E10 | R34, 60 epoch | 1024 | chờ Colab | | | Colab `unet1024_r34_e60` |
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

`colab/train_unet.ipynb`, dict `EXPERIMENTS` trong ô parameters; chọn bằng `RUN_NAMES`. Mỗi run
xong được upload lên `nguyenchicuong/filseg-<run>` (dấu `_` đổi thành `-`). Quy trình sau khi tải
về:

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
