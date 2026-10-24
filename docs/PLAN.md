# Kế hoạch

Mục tiêu: lọt top-3 theo rubric của ban tổ chức (70% định lượng, 30% định tính; xem
[PRIZE_RULES.md](../PRIZE_RULES.md)). Muốn cạnh tranh thì cần PQ khoảng ≥ 0.5 trên LB, đồng thời
code và report phải tốt.

## Mốc thời gian

| Ngày | Việc |
|---|---|
| 2026-11-15 06:00 UTC (13:00 VN) | Hạn submission Kaggle; chọn 2 final submissions |
| 2026-11-15 11:59 pm CDT | Hạn Google form: report 4 trang (PDF) + link repo public |
| 2026-11-15, ngay khi đóng | Chuyển repo GitHub sang **public** (`gh repo edit ... --visibility public`) |
| 2026-11-30 | Công bố người thắng |

Nên chốt model trước ~2026-11-10 để còn thời gian viết report và dọn code.

## Việc tiếp theo (ưu tiên từ trên xuống)

| # | Việc | Lý do | Trạng thái |
|---|---|---|---|
| 1 | U-Net R34 @1024 trên Colab (E7) | 512 → 768 tăng ~0.03; mask filament mảnh | chờ chạy Colab |
| 2 | Tách instance tốt hơn: thêm kênh biên/tâm, hoặc watershed trên map xác suất | gt_split + pred_merge ~190 trên val; mỗi lỗi tách/gộp tốn TP | chưa làm |
| 3 | Giảm FP: classifier theo từng instance (diện tích, độ tối, độ dài, độ tin cậy trung bình) | FP 965 ≈ FN 981 | chưa làm |
| 4 | Encoder mạnh hơn (EfficientNet/ConvNeXt), train lâu hơn (E5 chưa bão hoà ở epoch 30) | | chưa làm |
| 5 | Train trên toàn bộ ảnh (`--split all`) cho model cuối, giữ post-processing đã tune | thêm 30% dữ liệu | cuối cùng |
| 6 | Ensemble các model cùng độ phân giải / khác kiến trúc | E6: trộn 512 + 768 không giúp | sau 1–4 |

## Hồ sơ nhận giải (bắt buộc)

- [x] License MIT
- [x] `requirements.txt` pin version
- [x] README: cài đặt, môi trường, thời gian chạy
- [ ] Jupyter notebook chạy toàn bộ pipeline (tạm hoãn, làm khi chốt model)
- [ ] Technical report 4 trang theo template Overleaf của host (viết khi có kết quả tốt)
- [ ] Repo public ngay khi cuộc thi đóng, giữ public đến khi công bố
- [ ] Nộp Google form trước Nov 15 11:59 pm CDT
- [ ] Hiểu và giải thích được mọi thành phần (luật cho phép dùng AI nhưng yêu cầu điều này)
