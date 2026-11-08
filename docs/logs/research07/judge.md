# Research 07 — `judge`: phản biện 6 lens và một kế hoạch duy nhất tới 09/11

Agent `judge` (06/11). Đọc 6 báo cáo trong thư mục này và đối chiếu với EXPERIMENTS / RESEARCH / PLAN § 5b,
`scripts/train_unet.py`, `src/filseg/dl.py`, receipt của Dariush (`solar_unet_receipt.json`, trong scratchpad
`competitor_gap/ds/`). Có chạy thêm một phép đo kiểm chứng (§ 2). Script nằm ở
`/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/judge/paired_b.py`.
Không sửa gì trong repo ngoài file này.

## 0. Kết luận ngắn

1. Hầu hết ý tưởng "đột phá" ở tầng giải mã đều đã bị chính các lens bác bỏ bằng số đo, và mình đồng ý với các
   phép bác bỏ đó. Oracle chéo annotator cho thấy: kể cả khi biết nhãn thật của các annotator khác, việc chọn
   keep/extent cũng chỉ được khoảng 0 (keep +0.0000, extent −0.0074). Khoảng 0.16 PQ còn lại là phong cách riêng
   của từng annotator, không có gì trong ảnh dự đoán được phần đó. **Tầng giải mã đã cạn.**
2. Chỉ có **một** đòn bẩy có bằng chứng ngoài đủ lớn để thu hẹp khoảng cách 0.02 tới các đội trung thực:
   **sao chép đúng recipe V2-S của Dariush** (crop 512 ở độ phân giải gốc 2048, bs 16, lr 3e-4, 10.5k bước,
   checkpoint cuối, infer cả ảnh 2048). Trong cùng pipeline của chính đội đó, recipe này đưa R34 → V2-S từ 0.4299 lên
   0.4461 OOF, và **LB từ 0.37 lên 0.39**. Bước nhảy 0.02 trên LB là mức LB 2 chữ số thấy được chắc chắn
   (kaggle_endgame). R34 của họ ở LB 0.37 cũng chính là mức của mình hiện nay. Cả recipe chỉ cần **đổi cờ**, không
   cần viết code mới (§ 3).
3. Phép đo kiểm chứng của judge cho (B) "ứng viên rộng + GBM u": so với pipeline **hiện tại** (đã có lọc FP), Δ chỉ
   là **+0.002 [−0.002, +0.007]**, P > 0 chỉ 0.77–0.83 qua 4 seed fold. Con số +0.0035 trong báo cáo là so với nền
   **chưa lọc**. Không qua F48 → **drop**.
4. Gate gộp (c) đã đăng ký trước có một lỗi: (c) loại mọi họ có PQ fold lệch quá ±0.02 so với N1. Nếu N9′ tốt như
   Dariush (+0.02…+0.03) thì chính (c) sẽ loại nó. Phải sửa (c) thành gate một phía **trước khi** có số (§ 4).

## 1. Phán quyết từng ý tưởng

| # | Ý tưởng (lens) | Phán quyết | Lý do phản biện chính |
|---|---|---|---|
| 1 | (A) Recipe Dariush: V2-S in21k, crop 512 @2048, bs 16, lr 3e-4, ~10.5k bước, ckpt cuối, infer cả ảnh 2048 (competitor_gap) | **colab_slot**: chạy N9′ ở slot đầu; N10 bỏ | § 3 |
| 2 | (C) Refiner theo instance trên crop native, stage 2 (competitor_gap) | **colab_slot có điều kiện**: lấy slot thứ 3 (N8) nếu code xong trước tối 07/11; không thì chuyển thành hướng tương lai trong report | § 5 |
| 3 | I3 nối các mảnh của một instance thành một khối (competition_structure) | **do_now** | rẻ (Mac 0.5 ngày); ΔPQ ≈ 0…+0.001; phục vụ tiêu chí hình thái mask và 1:n/n:1. Mới thật: chưa ai đo tỉ lệ instance nhiều mảnh (5.1% so với 0.4% ở GT). Nối phải nằm ngoài các instance khác; giữ assert rời nhau |
| 4 | (B) Tầng quyết định trên ứng viên cc p > 0.35 + GBM u, hoặc cây 3 mức + DP (competitor_gap) | **drop** | Judge đo lại: so với nền đã có GBM (0.4326), Δ +0.0022 [−0.0020, +0.0066], P>0 0.83. Với 3 seed fold khác: +0.0017 / +0.0017 / +0.0021, CI đều chạm 0. Cây + DP thì đã thử ở H6-tree (−0.004). Lần duy nhất đáng chạy lại là sau khi có map pool mới (script sẵn, 0 GPU), và chỉ nhận nếu CI > 0 |
| 5 | Target "annotator ảo" = trộn nhãn người với OOF map của teacher (data_leverage) | **drop** trước 09/11 | Mới thật, nhưng chỉ có proxy MSE, chưa có số PQ. Có bẫy rò nhãn bậc 2: teacher OOF của fold k đã học nhãn của fold j. Student ≈ teacher nên khi vào ensemble rất ít đa dạng. Chạy thì tốn một slot Colab, mà recipe (A) có bằng chứng mạnh hơn nhiều. Ghi vào report như hướng tương lai |
| 6 | Mask R-CNN độ phân giải cao làm thành viên, ghép theo instance (new_paradigm) | **drop** | N4-mac âm (−0.006). Phần ghép theo instance có hiệu chỉnh chưa được viết. Chính lens ước P(qua gate) ~25%. Bị (A) lấn át khi tranh slot |
| 7 | Annotator-sample heads, head theo team/annotator, giải mã theo annotator/nhóm, reweight theo năm/trạm, head nhiễu annotator (metric_decoding, annotator_modeling) | **drop** | Đã bị số đo bác bỏ: biết nhóm +0.0000; biết số annotator ≤ +0.0016; annotator test không đoán được (6.8%, thấp hơn mức ngẫu nhiên 10.9%). F51: hiệu chỉnh xác suất cho 0 PQ |
| 8 | Giải mã theo PQ kỳ vọng trên phân phối annotator, có generator hoàn hảo (new_paradigm) | **drop**; con số đưa vào report | Oracle dùng người thật: tree +0.0036 với CI chứa 0; keep +0.0000. Đây là bằng chứng "trần do annotator", nên ghi thành F57 |
| 9 | Foundation encoder (DINOv2/SAM), phương pháp instance trực tiếp cho vật mảnh, cascade/diffusion refiner (new_paradigm) | **drop** | Không kiểm chứng được trước 09/11. F37/F42: model cần chi tiết nhỏ hơn một patch ViT. DINOv3 không OSI. F56: mọi chỉnh biên theo cường độ đều âm |
| 10 | Frame GONG lân cận, dữ liệu filament ngoài, target đồng thuận ở mức instance (data_leverage) | **drop** | D1: đã thấy gần như cùng bầu trời mà OOF không lên (0.418 so với 0.435). D2: mọi cách gộp nhãn người ≤ 0.347, trong khi model đạt 0.424. Dữ liệu ngoài vừa trùng ngày test vừa có rủi ro luật (host là tác giả MAGFiLO) |
| 11 | "Model khớp chuyên gia hơn chuyên gia khớp nhau" trên cùng ảnh, có CI (competition_structure I2) + trần đồng thuận của người (metric_decoding) + D2 | **report_only** | +0.090 [+0.077, +0.103] trên 296 ảnh. Phải trình bày là "đồng thuận với nhóm chuyên gia" chứ không phải "đúng hơn chuyên gia": model ước trung bình, còn mỗi người chỉ là một mẫu. Trừ ~0.002 do post tune in-sample |
| 12 | Tối đa rubric thay vì đuổi PQ (I1); Pareto hiệu năng + chọn cặp final (I4) | **report_only** | Hợp lý và rẻ, nhưng không phải lý do để bỏ (A). Giả định "mỗi mục rubric chấm theo tỉ lệ với đội tốt nhất" chưa được host xác nhận. Đo thời gian 1/2/10 model là việc rẻ, nên làm cùng notebook 07/11 |
| 13 | Đọc lại thang suy giảm bằng điểm lọc cross-fit (annotator_modeling) | **report_only** (sửa docs) | Kết luận trong commit f2879ee ("instance điểm thấp giữ nhiều TP hơn trên test") dựa trên điểm in-sample. Bản cross-fit: drop15 −0.008, khớp LB −0.02 ± 0.01 → **không có bằng chứng lệch test**. Sửa PROGRESS/EXPERIMENTS; ngưỡng lọc giữ nguyên |

## 2. Phép đo kiểm chứng của judge: (B) so với pipeline hiện tại

`paired_b.py` dùng lại `analyze.py` và `tables.pkl` của competitor_gap. So D35 (cc p > 0.35, ≥ 30 px, không seed,
không gộp) + GBM u nested với nền (hậu xử lý final 1) + GBM u nested, tức là thứ tương đương final 1 đang dùng.
Bootstrap ghép cặp 1,000 lần theo ảnh. Để xem kết quả có phụ thuộc cách chia fold hay không, mình chia lại fold
của rescorer theo ảnh với seed 1–3.

| Fold rescorer | nền + u | D35 + u | Δ [CI 95%] | P > 0 |
|---|---|---|---|---|
| fold_split gốc | 0.4326 | 0.4348 | +0.0022 [−0.0020, +0.0066] | 0.83 |
| seed 1 | 0.4335 | 0.4352 | +0.0017 [−0.0023, +0.0059] | 0.78 |
| seed 2 | 0.4333 | 0.4350 | +0.0017 [−0.0024, +0.0061] | 0.77 |
| seed 3 | 0.4329 | 0.4349 | +0.0021 [−0.0018, +0.0063] | 0.83 |

Δ ổn định quanh +0.002 nhưng không qua F48. Ở cả 4 lần chia, lợi này nhỏ hơn mức LB 2 chữ số thấy được tới một
bậc độ lớn. Kết luận: không đáng một giờ Mac nào trước khi có map mới.

## 3. Vì sao (A) là đòn bẩy duy nhất đáng một slot, và phản biện nó

**Bằng chứng ủng hộ** (receipt của Dariush, chính tay mình đọc lại):

- Cùng recipe, cùng luật giữ instance, cùng kiểu fold theo ảnh: R34 0.4299 (LB 0.37) → V2-S 0.4461 (LB 0.39).
  Luật in-sample 0.4469 so với nested 0.4461, nên con số không phải do tune. Theo từng fold, V2-S đạt
  0.452 / 0.436 / 0.446 / 0.454 / 0.445, tức hơn R34 ở cả 5/5 fold.
- R34 của họ (0.4299 với một luật 4 tham số) đã ngang final 1 **10 model** của mình (0.4327). Trong khi đó N1 B4
  một họ chỉ đạt 0.4179–0.4249. Mà trong E9a, R34 → B4 cho +0.013. Suy ra phần lợi đến từ **recipe** chứ không chỉ
  từ encoder. F50 đã nghi đúng ba điểm: bs 4 với BatchNorm, chỉ 30 epoch (~28k mẫu so với ~168k của họ), chọn
  checkpoint trên chính fold val.
- Luật của họ áp lên map của mình chỉ ra 0.4237. Vậy khoảng cách nằm ở **map**, không nằm ở post.
- Recipe chỉ cần đổi cờ, không cần code mới. `SoftDataset` đã có crop với fg_frac 0.7. OneCycle với pct_start 0.05
  thực chất là warmup rồi cosine. Đặt `--val-every` bằng `--epochs` thì chỉ val một lần ở epoch cuối, tức lưu
  checkpoint cuối và không chọn ckpt trên fold OOF. N2 đã infer cả ảnh 2048.

**Phản biện và cách xử lý:**

| Phản biện | Đánh giá |
|---|---|
| N2 (crop 1024 @2048, bs 4) đơn lẻ kém N1 −0.005 với +643 FP. BSergey thấy crop 640 native kém 0.07 | N2 khác Dariush ở 4 chỗ cùng lúc (crop 1024, bs 4, 30 epoch, ckpt theo val). Vì vậy N2 không bác được recipe này. Phần FP đã có lọc GBM và luật mean p lo |
| theainerd B3 crop 512 native chỉ đạt 0.4228 cho một họ | Đúng: native-512 tự nó không phải phép màu. Vì vậy phải sao chép **nguyên** recipe (V2-S + bs 16 + 10.5k bước), không chỉ lấy một phần |
| Không có z-score theo đĩa | Chấp nhận lệch một chỗ này: GONG đã được chuẩn hoá ở trạm (median 130–133 ở cả 6 trạm, research06). Thêm z-score sẽ mở ra một đường code chuẩn hoá riêng cho một thành viên trong notebook, mà một bug lệch train/infer ở đó thì gây chết. Giữ /255 như mọi thành viên khác |
| Chi phí | Dariush train trên Kaggle T4 ×2. Ước trên L4 khoảng 1.2–1.5 h/fold, tức khoảng 6–7.5 h cho 5 fold, vừa một slot. Nếu bs 16 tràn bộ nhớ thì dùng bs 8 với cùng lr (họ chạy micro-batch 8) |
| Kích thước weights | 5 × V2-S fp32 ≈ 5 × 88 MB. Nếu pool thành 15–20 model thì vượt 1 GB, nên phải dùng fp16 (plan B, đã có gate \|Δ PQ\| < 0.0005) |
| Có thể không tái lập được | Có. Mình ước P(họ N9′ đơn lẻ ≥ N1 + 0.010 trên 707 OOF) ≈ 0.5 |

**Lệnh** (fold k = 0..4, bundle Colab hiện có, chỉ thêm một mục trong RUN_NAMES):

```
--encoder tu-tf_efficientnetv2_s.in21k_ft_in1k --targets soft --res 2048 --crop 512 --bs 16 --lr 3e-4
--epochs 184 --val-every 184 --fold k --name effv2s_c512_f{k}
```

184 epoch × khoảng 57 bước (≈ 920 mục train / 16) ≈ 10.5k bước. Infer cả ảnh 2048 với TTA8, giống N2.

## 4. Sửa gate gộp trước khi có số (đăng ký 06/11)

- (c) hiện yêu cầu PQ fold của họ mới nằm trong ±0.02 so với N1 cùng fold. Gate này sẽ loại oan một họ tốt
  thật. Sửa thành **một phía**: PQ fold ≥ N1 − 0.02 cho mọi fold. Giữ thêm kiểm tra "không fold nào hơn N1
  quá +0.05", vì mức đó gần như chắc là bug hoặc rò nhãn.
- Gate dừng sớm cho N9′ sau fold 0: chạy tiếp 4 fold nếu **hoặc** N9′ f0 đơn lẻ (`hyst_soft`) ≥ 0.4158 + 0.005,
  **hoặc** (N1 + N2 + N9′) f0 − (N1 + N2) f0 ≥ +0.005. Nếu trượt cả hai thì dừng và chạy N9 gốc.
- Nhận vào final: vẫn dùng gate gộp (a)(b) trên 707 ảnh như PLAN § 5b. Sau đó re-tune grid hysteresis và train
  lại bộ lọc FP trên map pool.

## 5. (C) refiner: phản biện

- Hai đội độc lập báo lợi, cả hai có CI: zhang +0.014, 5/5 fold; tush +0.009 sau một tầng quyết định đã rất mạnh.
  Lợi kiểu này không bị oracle chéo annotator chặn trên, vì refiner nhìn thêm **pixel gốc 2048**, tức thông tin
  mới, chứ không chỉ chọn giữa các mức ngưỡng của map. F37 cho thấy chi tiết nhỏ quan trọng (blur σ 1.5 → −0.096).
- Ngược lại, phần lợi của zhang chủ yếu là bớt FP (FP −607), mà mình đã có bộ lọc GBM. N2 và UP2048 cũng đã
  đem một phần thông tin 2048 vào map. Code mới mất khoảng 1 ngày (cắt crop, dataset, dán lại có cấp phát pixel
  rời nhau), lại rơi đúng tuần cần dồn sức cho tái lập. Ước Δ +0.002…+0.008, P(qua gate fold 0 +0.005) ≈ 0.3.
- Điều kiện: chỉ làm khi N9′ đã chạy trên Colab và notebook end-to-end 07/11 đã đạt. Pilot fold 0 trên Mac
  (R18-UNet 4 kênh, crop 256 native, train trên instance OOF của fold 1–4). Không xong trước **tối 07/11** thì bỏ,
  và đưa vào report như hướng tương lai, kèm trần F52.

## 6. Kế hoạch đột phá duy nhất (06/11 → 09/11 đóng băng → 15/11)

1. **Ngay hôm nay (06/11)**: thêm `effv2s_c512_f0..f4` (lệnh § 3) vào RUN_NAMES và đóng gói lại bundle.
   Nếu N9 (V2-S resize 1024) **chưa bắt đầu**: hàng đợi thành **N9′ → N9**, bỏ N10 và N8. Lý do: cặp V2-S
   1024-resize + 2048-native lặp lại đúng cấu trúc cặp N1 + N2 (+0.0105). N10 (B4 @1536) là độ phân giải trung
   gian của một họ đã có, nên ít đa dạng nhất. Nếu N9 **đang chạy**: để N9 chạy xong, rồi N9′ thay N10.
   Ghi sửa gate (§ 4) vào PLAN **trước** khi có số.
2. **06/11 (Mac)**: I3 (nối mảnh) với cross-tune OOF bằng `compare.py`; sửa docs phần thang suy giảm (#13); ghi F57
   (trần annotator 0.166 so với khoảng 0.004 khai thác được). Không làm thêm việc giải mã hay hậu xử lý nào khác.
3. **07/11**: fold 0 của N9′ về → gate dừng sớm (§ 4). Chạy notebook end-to-end như lịch, kèm bảng thời gian
   1/2/10 model (I4). Chỉ bắt đầu pilot refiner (C) khi cả hai việc này đã ổn.
4. **08/11**: 5 fold N9′ (và N9) về. Gate gộp (a)(b)(c đã sửa) trên 707 ảnh. Re-tune hysteresis, train lại bộ lọc
   FP. Chạy lại (B) trên map pool vì miễn phí, nhưng chỉ nhận nếu CI > 0. Nộp bài loại 3 cho N9′ 5 fold đơn: nếu
   recipe tái lập được thì LB sẽ lên 0.38–0.39, đủ để thấy.
5. **09/11**: đóng băng. Final 1 = pool OOF cao nhất đã qua gate. Final 2 = bước đã qua gate ngay trước đó
   (N1 + N2 orient). Pool 20 model → dùng fp16.
6. **10–15/11**: tái lập, rồi report. Headline của report là "model đồng thuận với nhóm chuyên gia hơn chuyên gia
   với nhau, +0.090 [CI]", kèm phân rã lỗi, trần annotator (F57, D1, D2) và các hướng đã loại có số.

**Xác suất thật** (ước lượng chủ quan, không có phép đo nào cho các con số này):

| Kết cục | Xác suất |
|---|---|
| N9′ về đủ 5 fold trước 09/11 | ≈ 0.75 (phụ thuộc người dùng chạy Colab liên tục) |
| Pool có N9′ qua gate với Δ OOF ≥ +0.008 (final 1 ≥ 0.441) | ≈ 0.45 |
| Final OOF ≥ 0.450 (ngang Dariush / theainerd) | ≈ 0.2 |
| LB final ≥ 0.39 | ≈ 0.3 |
| Lọt top 3 theo rubric của ban giám khảo | ≈ 0.03–0.08. Theo PQ thuần thì gần 0 (294 đội trung thực ở 0.38–0.42); cơ hội nằm ở số đội thật sự nộp đủ report + repo + notebook, và ở phần 30% định tính |

Không có "giải pháp đặc biệt" nào trong 6 lens vượt được mọi đối thủ. Lý do là trần do annotator đã được đo, và
những ý tưởng cao tay (giải mã theo annotator, generator, target đồng thuận) đều ≤ +0.005. Đường thực tế duy nhất để
bắt kịp tầng 0.39–0.40 là **sửa map bằng recipe đã được chứng minh**, kèm một report mạnh hơn các đội cùng mức.
