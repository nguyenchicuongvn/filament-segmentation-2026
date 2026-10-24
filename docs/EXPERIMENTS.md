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
| E7 | U-Net ResNet34 (Colab) | 1024 | đang chờ | | | `colab/train_unet.ipynb` |

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

## E7: U-Net ResNet34 @1024 (Colab) — đang chờ

`colab/train_unet.ipynb`, run `unet1024_r34` (`--res 1024 --bs 8 --epochs 30`). Kết quả sẽ được
upload lên private dataset `nguyenchicuong/filseg-unet1024-r34`; tải về rồi tune bằng
`scripts/tune_post.py`.
