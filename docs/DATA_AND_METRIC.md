# Dữ liệu và metric

## Dữ liệu (MAGFiLO 1.0, bản Kaggle 2026)

| | |
|---|---|
| Ảnh | GONG H-alpha, JPEG 2048×2048, 3 kênh giống hệt nhau (đọc grayscale) |
| Train | **707 ảnh**, **1,154 bản gán nhãn** (annotator-image), 8,199 filament |
| Annotator mỗi ảnh | 411 ảnh có 1 annotator · 145 ảnh có 2 · 151 ảnh có 3 |
| Test | 180 ảnh, không trùng train |
| Năm | 2011–2022 (tập trung 2011–2017), phân bố giống nhau giữa train và test |
| Trạm GONG | 6 trạm (B, C, L, M, T, U = ký tự cuối tên file), phân bố đều |
| Filament / bản gán nhãn | trung bình ~7 (từ 1 đến > 15) |
| Diện tích filament (px ở 2048²) | p10 410 · trung vị 1,228 · p90 4,685 · p99 13,700 |
| Annotation | COCO polygon (1 polygon / filament), kèm bbox, spine, nhãn chirality (Left/Right/Unidentifiable) |

- id của ảnh trong JSON có dạng `<annotator>-<stem>`, ví dụ `040301-20140609195854Bh`.
- **Luật cấm** dùng bbox, spine, nhãn chirality khi train; chỉ dùng mask. Inference chỉ dùng ảnh
  H-alpha của test.
- Quan sát trực quan: annotator vẽ mask **rộng và liền** hơn vùng tối thực tế; sunspot và nhiều
  fibril tối nhỏ **không** được gán nhãn; một số filament mờ bị bỏ sót.

## Metric: Panoptic Quality

Lấy từ notebook self-evaluation của host
(https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), cài lại ở
`src/filseg/metric.py`:

```
PQ = Σ IoU(TP) / (|TP| + 0.5|FP| + 0.5|FN|)
```

- TP: cặp (GT, dự đoán) có IoU > 0.5 (lớn hơn hẳn, IoU = 0.5 không tính).
- Mỗi **bản gán nhãn** (annotator-image) được so riêng với toàn bộ dự đoán của ảnh đó. Ảnh có 3
  annotator được tính 3 lần.
- TP/FP/FN được **cộng dồn trên toàn bộ tập** rồi mới chia, không lấy trung bình theo ảnh.
- Dự đoán của một ảnh không có GT đều thành FP.
- Submission: mỗi dòng một filament, `filament_id = <stem>_<k>`, `segmentation_rle` = COCO RLE
  counts (pycocotools) ở 2048×2048.

Hệ quả: một filament bị cắt đôi thường cho 0 TP + 2 FP + 1 FN; hai filament bị gộp làm một cho tối
đa 1 TP. Tách/gộp instance đúng quan trọng ngang chất lượng mask.

## Split validation

`filseg.data.split`: 70/30 **theo ảnh** (seed 42); mọi annotator của một ảnh nằm cùng một phía.
Val = 213 ảnh / 337 bản gán nhãn, train = 494 ảnh / 817 bản gán nhãn. Val không dùng để train,
chỉ để chọn checkpoint và tune post-processing.

## Khoảng cách val ↔ LB

| Model | val | LB | chênh |
|---|---|---|---|
| Classical | 0.242 | 0.21 | 0.03 |
| U-Net 512 | 0.354 | 0.29 | 0.06 |
| U-Net 768 | 0.383 | 0.31 | 0.07 |

Thứ tự giữa các model giữ nguyên nên val vẫn dùng để chọn được. Cập nhật 29/10: nguyên nhân chính
được cho là lệch annotator (PQ theo nhóm annotator 0.37–0.46), cộng với nhiễu lấy mẫu của public LB
(SD 0.014 trên 90 ảnh). Bằng chứng ở [RESEARCH.md § 3](RESEARCH.md#3-leaderboard-và-khoảng-cách-val--lb)
(F8–F11, H5).
