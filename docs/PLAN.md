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
| 1 | U-Net R34 @1024 trên Colab (E7) | 512 → 768 tăng ~0.03; mask filament mảnh | người dùng đang chạy |
| 2a | Watershed thay cho gom mảnh bằng dilation (E5w) | gt_split + pred_merge ~190 trên val | **xong**: 0.384 ≈ cc, không dùng |
| 2b | Kênh group để model học gom mảnh (E8) | luật cố định chỉ đổi lỗi tách lấy lỗi gộp | dựng xong, chờ Colab |
| 3 | Lọc FP bằng classifier theo instance (E5f) | FP 965 ≈ FN 981 | **xong**: val +0.006, LB +0.02, giữ lại |
| 4 | Encoder mạnh hơn (E9), train 60 epoch (E10) | E5 chưa bão hoà ở epoch 30 | dựng xong, chờ Colab |
| 5 | Train trên toàn bộ ảnh (`--split all`) cho model cuối (E11) | thêm 30% dữ liệu | cuối cùng, sau khi chọn công thức |
| 6 | Ensemble các model 1024 cùng độ phân giải / khác kiến trúc | E6: trộn 512 + 768 không giúp | công cụ có sẵn, chờ model |

Thứ tự đề xuất trên Colab: E7 (đang chạy) → E8 → E9 → E10 → E11 với công thức thắng.

## Hồ sơ nhận giải (bắt buộc)

- [x] License MIT
- [x] `requirements.txt` pin version
- [x] README: cài đặt, môi trường, thời gian chạy
- [ ] Jupyter notebook chạy toàn bộ pipeline (tạm hoãn, làm khi chốt model)
- [ ] Technical report 4 trang theo template Overleaf của host (viết khi có kết quả tốt)
- [ ] Repo public ngay khi cuộc thi đóng, giữ public đến khi công bố
- [ ] Nộp Google form trước Nov 15 11:59 pm CDT
- [ ] Hiểu và giải thích được mọi thành phần (luật cho phép dùng AI nhưng yêu cầu điều này)
