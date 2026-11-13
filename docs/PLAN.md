# Kế hoạch (bản 2026-11-13, phần đầu thay mọi bản trước; § 1 trở xuống là lịch sử)

## 0. Từ 13/11 đến 15/12: chặng cuối và giai đoạn xét giải

### 0.1 Vì sao không còn "30 ngày cải thiện model"

- Kaggle đóng **15/11 06:00 UTC (13:00 VN)**, tức còn khoảng 40 giờ. Sau mốc đó không nộp hay đổi
  final được nữa. 30 ngày tới gồm 2 ngày thi và 4 tuần xét giải (công bố 30/11, IEEE BigData 14–17/12).
- Không còn GPU: quota Kaggle 60 h/tuần đã hết, model đóng băng từ 09/11. Train một family 5 fold
  cần khoảng 6 h L4. Kể cả có GPU, một cải thiện cỡ +0.005 OOF cũng không chứng minh được trong 40 h
  với CI ±0.01, và LB không thấy (độ phân giải 0.01 trên ~90 ảnh).
- Mọi nút hậu xử lý còn lại đều đã đo và phẳng: hedge study 12/11 (8 biến thể × 5 ngưỡng, các view
  theo annotator), thang recall trên LB (cả ba bài 0.38), TTA 1152 (âm), size-dependent growth (≈ 0).
- LB public 13/11: top 0.70, top-20 ≥ 0.56. Các điểm này nằm trên trần trung thực (~0.40 public,
  ~0.46 OOF; RESEARCH § 3) nên phần lớn là nhiễm nhãn test. Giải tính bằng rubric (70% định lượng,
  30% định tính) và chỉ xét đội có report, repo public, code tái lập được. **Đòn bẩy còn lại là hồ sơ
  xét giải**, không phải PQ.

Kết luận: không train, không nộp thêm bài dò. Hai final giữ nguyên (final 1 ref 58174035, final 2
ref 58269004). Toàn bộ thời gian dành cho report, repo public và khả năng tái lập.

### 0.2 Lịch

| Khi nào (giờ VN) | Việc | Ai | Xong khi |
|---|---|---|---|
| 13/11 | Tick 2 final trên Kaggle (58174035, 58269004) | người dùng | trang My Submissions hiện 2 ô đã tick |
| 13/11 | Compile report trên Overleaf, báo số dòng còn vượt 4 trang | người dùng | PDF ≤ 4 trang |
| 13–14/11 | **R1** Dọn repo cho bản public (§ 0.3) | Claude | checklist § 0.3 xanh |
| 14/11 | **R2** Chạy notebook `final_inference.ipynb` có output (ảnh minh hoạ, PQ trên vài ảnh val, đường dẫn CSV) | Claude (Mac) | notebook lưu kèm output, chạy lại từ clone sạch |
| 14/11 | **R3** Kiểm tra tái lập từ clone sạch: venv mới, `pip install -r requirements.txt`, tests, notebook trên 3 ảnh test | Claude | CSV từng ảnh trùng byte với bản local |
| 14/11 | Report bản cuối: số trong report khớp `docs/logs/final/*.json`, link repo đúng | Claude + người dùng | PDF cuối |
| 15/11 trước 13:00 | Không nộp gì thêm. Tag `final` trên commit đang chạy final 2 | Claude | tag đã push |
| **15/11 13:00** | Kaggle đóng. **Chuyển repo sang public** ngay sau mốc này | người dùng (hoặc Claude khi được bảo) | `gh repo view` báo PUBLIC |
| 15/11 tối | Điền Google form: report PDF, link repo, tên/tài khoản Kaggle, tên đội | người dùng | email xác nhận của form |
| **16/11 trước 11:00** | Hạn form "Nov 15, 11:59 pm CDT". Ngày 15/11 Mỹ đã về giờ chuẩn (CST, UTC−6) nên hạn thật là 16/11 12:59 VN; nếu hiểu theo CDT (UTC−5) là 11:59 VN. Lấy mốc sớm hơn | người dùng | – |
| 16–29/11 | Giữ repo public, không force-push, không đổi lịch sử. Chuẩn bị hồ sơ người thắng (§ 0.4) | cả hai | – |
| 30/11 | Công bố. Nếu được báo thắng: trả lời email **trong 1 tuần** | người dùng | – |
| 30/11–14/12 | Ký giấy tờ **trong 2 tuần**, W-8BEN, onboarding AURA, nhận chuyển khoản từ ngân hàng Mỹ | người dùng | – |
| 14–17/12 | IEEE BigData (Phoenix), tham dự là tuỳ chọn | người dùng | – |

### 0.3 R1: dọn repo cho bản public (30% định tính chấm "code quality, modularity, documentation")

1. README: viết lại phần Environment (đang ghi "CPU only, classical baseline", đã cũ), thêm mục
   **Reproduce** ba bước: (a) inference từ weights trong repo bằng notebook, thời gian đo được;
   (b) train lại 10 model qua `colab/train_unet.ipynb` + `scripts/package_colab.py`, khoảng 17 GPU-giờ L4;
   (c) ghép map, hậu xử lý, lọc FP bằng `configs/` đã lưu. Thêm sơ đồ thư mục `src/` / `scripts/`
   / `configs/` / `weights/` và bảng kết quả ngắn (OOF, LB, metric rubric).
2. Ghi rõ trong README rằng `docs/` là nhật ký làm việc bằng tiếng Việt; report và README là bản
   tiếng Anh dành cho ban giám khảo.
3. Tách rõ code chính và code nghiên cứu: `scripts/research/` và các script thử nghiệm đã bỏ
   (Mask R-CNN, classical, agree filter) được liệt kê trong README là "không nằm trong pipeline final".
   Không xoá, vì EXPERIMENTS dẫn tới chúng.
4. Nhánh trên remote: 6 nhánh thử nghiệm sẽ public cùng repo. Đề xuất giữ (minh bạch); nếu muốn gọn
   thì xoá nhánh đã merge hoặc đã bỏ. **Người dùng quyết định**, vì xoá nhánh remote không hoàn tác được.
5. Kiểm tra trước khi public: không có token, `kaggle.json`, dữ liệu thi hay MAGFiLO trong lịch sử git
   (đã quét 13/11: chỉ hai CSV log nhỏ trong `docs/logs`, không có dữ liệu thi); file lớn nhất 89 MB
   (`weights/effv2s_c512_f*.pt`), dưới giới hạn 100 MB của GitHub.
6. `pytest` xanh (13/11: 29 passed), `requirements.txt` đủ gói và pin phiên bản, LICENSE MIT.

### 0.4 Chuẩn bị cho vòng kiểm tra người thắng (16–29/11)

- Luật: người thắng phải "hiểu, giải thích, bảo vệ và tái lập mọi thành phần chính"; dựa vào code AI mà
  không giải thích được có thể bị loại. Viết `docs/DEFENSE.md` (tiếng Anh, 1–2 trang): mỗi thành phần
  (U-Net 2 family, hysteresis, oriented grouping, lọc FP) trả lời ba câu: làm gì, vì sao chọn, bằng chứng
  nào (dòng EXPERIMENTS, Δ OOF, CI). Người dùng đọc và tự trả lời được trước 30/11.
- Đo lại thời gian end-to-end của pipeline 10 model trên GPU nếu quota Kaggle reset (hỏi người dùng
  trước), để thay con số ước tính 25 phút trong report bằng số đo.
- Hồ sơ nhận giải: W-8BEN; tài khoản nhận được chuyển khoản từ ngân hàng Mỹ (luật ghi: không nhận được
  thì mất giải); kiểm tra email Kaggle hằng ngày từ 30/11.

### 0.5 Việc không làm (và lý do)

| Ý tưởng | Lý do bỏ |
|---|---|
| Train thêm family / seed / full-data refit | Không GPU; không kịp đo; lợi dự báo ≤ +0.002 (research06) |
| Nộp thêm bài dò nút hậu xử lý | LB không phân giải Δ < 0.01; hedge study đã phẳng |
| Đổi final theo LB | Quy tắc đăng ký trước 12/11 không kích hoạt (thang recall đều 0.38) |
| Dùng MAGFiLO 1.0 public hay pseudo-label test | Vi phạm luật / rò nhãn test; mất tư cách xét giải |


Tài liệu này trả lời bốn câu hỏi: còn bao nhiêu thời gian và tài nguyên; thí nghiệm nào đáng chạy
và vì sao; chạy xong thì quyết định thế nào; đang ở đâu. Bằng chứng nằm ở
[RESEARCH.md](RESEARCH.md) (mã nguồn `S*`, `F*`, `H*`). Kết quả đo được ghi ở
[EXPERIMENTS.md](EXPERIMENTS.md). Tiến độ hằng ngày ghi ở [PROGRESS.md](PROGRESS.md).

## 1. Mục tiêu

- **Thước đo quyết định**: PQ out-of-fold (OOF) trên 707 ảnh train, hoặc PQ trên một fold khi so
  hai công thức. Luôn so **ghép cặp theo ảnh** (`scripts/compare.py`, bootstrap) và **cross-tune**
  hậu xử lý. LB public không dùng để quyết định, vì nó nhiễu ±0.03 và lệch annotator
  ([RESEARCH § 3](RESEARCH.md#3-leaderboard-và-khoảng-cách-val--lb)).
- **Mục tiêu số**: OOF ≥ 0.44 (các đội trung thực đầu bảng đạt 0.443–0.461, [RESEARCH § 6.2](RESEARCH.md#62-các-đội-trung-thực-có-công-bố-s5)). Mốc hiện tại:
  holdout 0.408 (cross-tune, 213 ảnh), LB 0.36.
- **Mục tiêu giải**: rubric 70% định lượng (Dice, PQ, phân bố Dice/IoU, tỉ lệ 1:n / n:1) và 30%
  định tính (`S8`). Vì vậy báo cáo cần đủ metric rubric và lập luận rõ ràng, không chỉ PQ.
- **Tiêu chí của ban giám khảo** (`S6b`): chất lượng mask trên các ví dụ họ chọn, và tính khả thi
  (chạy nhanh vừa phải, không phức tạp hoá, dễ hiểu). Mỗi thành phần thêm vào phải có lợi đo được;
  thành phần nào không qua tiêu chí thì không giữ trong pipeline final.

## 2. Thời gian và tài nguyên

| Mốc | Ngày | Còn (tính từ 29/10) |
|---|---|---|
| Đóng băng model (không train thêm) | **2026-11-09** | 11 ngày |
| Notebook end-to-end chạy được, weights trong repo | 2026-11-10 | 12 ngày |
| Chọn 2 bài final trên Kaggle; hạn submission | 2026-11-15 06:00 UTC (13:00 VN) | 17 ngày |
| Report 4 trang + Google form; chuyển repo sang public | 2026-11-15 11:59 pm CDT | 17 ngày |

| Tài nguyên | Ràng buộc | Hệ quả |
|---|---|---|
| GPU Colab (L4) | **~25 h** (người dùng xác nhận 29/10). EffNet-B4 @1024 mất ~75 phút một fold: 61 phút cho 494 ảnh ở E9a, fold có 566 ảnh | Mỗi lần train là một khoản chi; ước tính dưới đây tính theo giờ L4 |
| GPU Kaggle | Không dùng nếu chưa hỏi người dùng | Chỉ là phương án dự phòng |
| Mac M5 (MPS) | Train một fold mất ≥ 50 h, không khả thi ([EXPERIMENTS § Tự phản biện lần 2.5](EXPERIMENTS.md#5-train-trên-mac-có-thay-được-colab-không)) | Mac dùng để viết code, smoke test, infer, tune, phân tích |
| Lượt submit | 5 lượt/ngày, còn ~80 lượt | Không khan hiếm. LB chỉ để tham khảo, không quyết định gì |
| Repo / weights | Weights phải nằm trong repo (`S6`). EffNet-B4 fp32 nặng 82 MB, fp16 ~41 MB; GitHub chặn file > 100 MB | Lưu weights fp16 thì không cần Git LFS. Phải kiểm tra map fp16 so với fp32 (task F2) |

**Ước tính GPU cho toàn bộ kế hoạch** (L4): Q3 ~8 h (đã xếp hàng) + Q4 ~4 h + Q5 ~10 h + dự phòng
3 h, tổng **~25 h**. Nếu ngân sách thấp hơn thì cắt theo thứ tự ưu tiên ở § 5.

## 3. Quy trình thí nghiệm (áp dụng cho mọi run mới)

1. **Đăng ký trước**: trước khi train, ghi thẻ thí nghiệm (§ 4) gồm giả thuyết, nhóm đối chứng, tiêu
   chí đạt và việc làm tiếp theo cho cả hai kết quả. Không đổi tiêu chí sau khi đã thấy kết quả.
2. **So sánh**: hai nhánh chạy trên cùng fold, cùng số epoch, cùng cách chọn checkpoint, cùng grid
   hậu xử lý (cross-tune). Dùng `scripts/compare.py` để lấy Δ PQ, CI 95% và P(Δ > 0).
3. **Độ nhạy** (đã đo): trên 213 ảnh, CI của Δ ghép cặp rộng khoảng ±0.010 (`docs/logs/critique_models.txt`).
   Một fold có ~141 ảnh nên CI vào khoảng **±0.012**, hai fold ~±0.009. Hệ quả:
   - Hiệu ứng < 0.01 **không thấy được** với ngân sách hiện có. Vì thế các thay đổi nhỏ được **gộp
     thành gói** (N3) thay vì ablation từng cái.
   - Quy tắc: **chấp nhận** nếu Δ ≥ +0.012 và CI > 0 trên một fold, hoặc CI gộp > 0 trên hai fold.
     **Bỏ** nếu Δ ≤ 0. Ở giữa thì chạy thêm fold 1 (chỉ khi thẻ cho phép).
4. **Thiên lệch đã biết**, giống nhau ở hai nhánh nên phần lớn triệt tiêu khi so:
   - Checkpoint được chọn theo PQ trên chính fold held-out (3 ứng viên: epoch 10/20/30), nên OOF hơi
     lạc quan.
   - Hậu xử lý tune trên chính tập chấm thì lạc quan ~0.003. Khi quyết định luôn dùng cross-tune.
5. **Ghi chép**: mỗi run có một dòng trong sổ cái [EXPERIMENTS.md](EXPERIMENTS.md#sổ-cái) ngay trong
   ngày, cập nhật bảng theo dõi [PROGRESS.md](PROGRESS.md), rồi commit.
6. **Submit**: chỉ nộp (a) ensemble fold của một công thức mới được chấp nhận, hoặc (b) ứng viên
   final. Kết quả LB được ghi lại, không dùng để chọn.

## 4. Thẻ thí nghiệm

Trạng thái: ⏳ đang chạy/chờ · 📝 chuẩn bị · ✅ xong · ❌ bỏ.

### N1 — Baseline 5 fold: EffNet-B4 @1024 + target mềm (⏳ Q3 trên Colab)

| | |
|---|---|
| Giả thuyết | Trung bình 5 model fold (mỗi model thấy 80% ảnh) cho test tốt hơn E9a (1 model, 70% ảnh) |
| Bằng chứng | Mọi đội ở [RESEARCH § 6.2](RESEARCH.md#62-các-đội-trung-thực-có-công-bố-s5) nộp ensemble fold. Ngược lại: ensemble các model **yếu hơn** của mình không giúp (Δ −0.0001, CI [−0.008, +0.008]) |
| Thiết kế | `effb4_s1024_f0..f4`, 30 epoch, val mỗi 10 epoch. Watcher ghép map OOF, tune trên 707 ảnh, nộp trung bình 5 fold |
| Đo được gì | OOF PQ là **baseline B1** cho mọi so sánh về sau. Lợi ích của việc ensemble **không đo được tại chỗ** (OOF chỉ là model đơn), chỉ biết qua LB |
| Giới hạn | Target mềm và cách chia dữ liệu đổi cùng lúc, nên không tách được tác dụng riêng của target mềm (nếu report cần thì chạy 1 run target cứng trên fold 0) |
| Chi phí | 5 × ~75 phút ≈ 6.3 h L4 (đã xếp trong Q3) |
| Tiêu chí | Không phải tiêu chí chấp nhận/bỏ: đây là baseline. Kiểm tra hợp lý: PQ từng fold nằm trong 0.38–0.44 và OOF cross-tune ≥ 0.40. Nếu thấp hơn nhiều thì nghi có bug (ví dụ ngưỡng của target mềm) và dừng lại tìm |
| Sau đó | Nộp ensemble 5 fold: ứng viên final "an toàn". Mọi thẻ sau dùng các fold này làm đối chứng |

### N2 — Độ phân giải: @1024 cả ảnh so với crop 1024 từ ảnh 2048 (⏳ fold 0 trong Q3)

| | |
|---|---|
| Giả thuyết | Độ phân giải gốc cứu filament nhỏ: 54% filament < 400 px không có prediction nào chạm (`F12`) |
| Ủng hộ | theainerd dùng tile 512 native (`S10a`); tushhaaarrrr làm end-to-end @2048 (`S15`). Trần IoU: 1024 → 1536 tăng 0.894 → 0.909, riêng filament < 800 px tăng 0.860 → 0.881 (`F14`) |
| Phản đối | BSergey2000: native 2048 với crop 640 **kém 0.07** (`S13`). Trần độ phân giải chỉ tăng ~0.015 IoU, trong khi SQ hiện tại 0.668, nên độ phân giải không phải nút thắt chính. R34 768 → 1024 không tăng (0.362 so với 0.357 PQ thô) |
| Kỳ vọng trước | Δ ≈ 0 ± 0.02, không chắc chắn |
| Thiết kế | `effb4_c2048_f0` so với `effb4_s1024_f0`, cùng 141 ảnh fold 0, cross-tune grid đã scale theo độ phân giải |
| Việc cần làm trước | **P1**: `compare.py` phải scale `min_area`/`group_radius` theo độ phân giải map (grid hiện tại tính cho map 1024; watcher đã scale nhưng compare chưa) |
| Chi phí | 1 run (~1.5 h, đã xếp trong Q3) |
| Tiêu chí | Δ ≥ +0.012 và CI > 0 → **chấp nhận**. Thêm điều kiện (workflow 30/10, RESEARCH F30): N2 phải giảm nhóm FN "sót dù có người khác vẽ" và "gần trúng béo quá" (`scripts/research/wf30/errors/`), không chỉ PQ; vì phần lớn filament nhỏ bị sót là do bất đồng giữa annotator, PQ tăng mà các nhóm này không đổi thì nghi nhiễu. Lưu ý: chỉ một cặp run trên một fold, nên không loại được nhiễu seed; nếu Δ vừa qua ngưỡng thì chạy thêm fold 1 @2048 trước khi chạy cả 5 fold |
| Nếu đạt | Chạy fold 1–4 @2048 (~6–7 h) và dùng làm công thức chính. N3 cũng làm ở @2048 |
| Nếu không | Giữ @1024 (rẻ hơn), dồn giờ GPU cho N3/N4. Nếu 0 < Δ < 0.012 thì có thể dùng làm thành viên ensemble khi còn giờ |

### N3 — Gói "công thức v2": EMA + blob loss + loss thành phần giả kiểu BiCC (📝 code xong, chờ Q4)

| | |
|---|---|
| Giả thuyết | Loss chuẩn hoá theo instance (blob Dice + phạt thành phần dự đoán không chạm GT) cộng EMA giảm FP (khoảng cách chính với đội top: FP/TP 0.57 so với 0.42) và cho +0.012 trở lên. Từng cái riêng thì quá nhỏ để đo (§ 3.3) |
| Bằng chứng | **Blob loss** + head skeleton/core + deep supervision + loss JML1: +0.0092, CI **90%** [+0.006, +0.012], ablation proxy 5 fold R34 @1024 (umerkang66, `S11a`). Đó là một gói; blob loss riêng không có số. **EMA** 0.998: umerkang66 và ranjeet258 có dùng (`S11b`, `S12`), không có số tách riêng. **Thành phần giả kiểu BiCC** (`S10b`, +0.0095 trung bình, tăng ở 4/5 fold, không có CI): blob loss chỉ phạt theo instance GT; "nửa kia" của BiCC phạt từng thành phần dự đoán không chạm GT, nhắm vào ~120 FP không chạm GT nào và FP mảnh vụn |
| Đã bỏ khỏi gói | **Input hiệu chỉnh limb**: hai phân tích độc lập đo lỗi do vị trí/limb/vết đen < 1% số lỗi (3/794 FP; tỉ lệ không khớp phẳng theo r/R 0.25–0.31), nên không có mục tiêu đo được, lại đổi đường input ở khâu test. **Head skeleton/core**: cần thêm target, để sau |
| Cài đặt | `filseg.dl`: `instance_labels` (thành phần 8-liên thông của target mềm; thành phần < `inst_min_area` gán −1), `blob_dice_loss` (dạng đóng, scatter_add, Dice từng instance trên cả ảnh với các instance GT khác bị che), `false_component_loss` (thành phần của p > 0.5 có diện tích ≥ min_area và không chạm target > 0; loss = trung bình p trong thành phần), `train_loss`. `scripts/train_unet.py`: `--blob-weight 1.0 --fpc-weight 0.3 --fpc-thresh 0.35 --ema 0.998` (ngưỡng 0.35 vì map target mềm thấp hơn: validate cũng dùng 0.35; đặt trước khi chạy, theo review 30/10; `--inst-min-area` 0 = 10 px ở 1024, scale theo `--res`). Mặc định giữ nguyên đường code cũ (các run Q3). Checkpoint `model` = trọng số EMA, `model_raw` = trọng số thô. Khi bật EMA, history ghi thêm PQ/TP/FP/FN của trọng số thô (`raw_val_pq`...) để phân biệt EMA trễ với tác dụng của loss; chọn checkpoint vẫn theo EMA. Unit test `tests/test_instance_loss.py` (so với vòng lặp brute-force, gradient, ca rỗng, label khớp sau xoay/lật/crop, EMA lưu/nạp) |
| Thiết kế | `effb4_v2_s1024_f0` và `effb4_v2_s1024_f1` (đã có trong EXPERIMENTS, chưa vào RUN_NAMES) so ghép cặp với `effb4_s1024_f0/f1` (N1): cùng encoder, độ phân giải, 30 epoch, `--val-every 10`, cùng luật chọn checkpoint tốt nhất trong 3, cùng ảnh của fold 0 và fold 1 (~282 ảnh). Nếu G1 chọn 2048 thì thêm `effb4_v2_c2048_f0` (`--res 2048 --crop 1024 --bs 6`) và so với `effb4_c2048_f0` |
| Lệnh đo | `compare.py n1=outputs/colab/effb4_s1024_f0/probs_val,tune:hyst v2=outputs/colab/effb4_v2_s1024_f0/probs_val,tune:hyst` (bootstrap ghép cặp theo ảnh, cross-tune); `tune:cc` là kiểm tra phụ. Combo tốt nhất nằm ở mép grid thì mở rộng grid cho cả hai nhánh như nhau rồi chạy lại |
| Chi phí | GPU: 2 run (~2.6 h), thêm 3 fold (~3.9 h) nếu chấp nhận |
| Tiêu chí | Gộp fold 0 + 1: Δ ≥ +0.010 **và** CI 95% > 0 **và** Δ dương ở cả hai fold → chấp nhận. Còn lại → bỏ, giữ N1. Lý do luôn chạy 2 fold (review 30/10): CI bootstrap chỉ tính nhiễu do chọn ảnh, không tính nhiễu giữa hai lần train (seed); một cặp run trên một fold có thể cho Δ +0.012 chỉ vì seed. Hai fold là hai lần train độc lập. Không bao giờ quyết bằng LB |
| Chẩn đoán cơ chế | Báo cáo, không dùng để quyết (bằng `scripts/analyze_errors.py`, và script phân loại FP/FN của workflow 30/10 sẽ được đưa vào repo trên nhánh `research/new-solution`): số FP near-miss+partial, FP cô lập (không chạm GT), FN merged/split, IoU TP trung bình (SQ), tỉ lệ 1:n/n:1 ở cả hai nhánh. Chỉ được quy công cho loss instance nếu FP near-miss+partial giảm ≥ 10% hoặc FP cô lập giảm ≥ 20% |
| Nếu đạt | Chạy 3 fold còn lại; v2 thay N1 làm thành viên U-Net chính |
| Nếu không | Giữ N1. Không tách gói ra để thử lẻ (không đủ độ nhạy) |
| Ghi chú cho report | Nếu chấp nhận thì nói rõ đây là gói, không quy công cho từng thành phần |

### N4 — Thành viên detector: Mask R-CNN (torchvision), ghép với U-Net (📝 code xong; thử trên Mac âm, chờ quyết định GPU)

| | |
|---|---|
| Giả thuyết | Model instance-level ít FP và ít gộp/tách sai hơn U-Net + cc. FP/TP của mình là 0.57, của theainerd 0.42 (`F9`). Ghép với U-Net được +0.02–0.04 |
| Bằng chứng | biconcavelens, cùng cách chia: Mask R-CNN **0.429** so với U-Net ngữ nghĩa ConvNeXt-T của họ **0.396** (+0.033, `S9a`). Bỏ phiếu mask có trọng số điểm +0.007–0.012, nhưng hệ thống đầy đủ chỉ có CI > 0 ở 3/5 lần chia (`S9c`). Bỏ hiệu chỉnh isotonic thì mất 0.021 (`S9b`), nên ghép mà không hiệu chỉnh là sai. Mask R-CNN của họ train @2048 chỉ 6 epoch trên một GPU 16 GB (`S9f`). Lưu ý: U-Net của họ yếu hơn của mình (0.396 so với ~0.41), nên lợi thực tế có thể nhỏ hơn. Thêm detector đi ngược tiêu chí "không phức tạp hoá" (`S6b`), nên chỉ giữ nếu lợi rõ |
| Rủi ro | Lớn nhất trong kế hoạch: code mới (dataset instance, train, infer 2048 với ~30 instance, ghép với map U-Net, hiệu chỉnh). TTA cho detector phức tạp hơn. Mask của detector có thể chồng nhau: phải giải chồng lấn về một ảnh nhãn và thêm assert rời nhau vào `make_submission.py` (luật `S3`) |
| Thiết kế | (1) Code + smoke test trên Mac (torchvision Mask R-CNN đã chạy được trên MPS). (2) Train fold 0 trên Colab @1024. (3) Đo trên fold 0: Mask R-CNN một mình so với U-Net N1; ensemble bỏ phiếu so với U-Net N1 |
| Cài đặt (01/11, nhánh `exp/maskrcnn`) | `filseg.maskrcnn`: Mask R-CNN R50-FPN v2 (COCO, BSD-3), mỗi bản gán nhãn là một sample (giống cách metric chấm); detection lưu cache `<stem>.npz`. Hai cách ra instance: `detections_to_labels` (detector một mình, tô theo score, bỏ detection bị che quá nửa) và `fuse_with_unet` (U-Net giữ pixel; detector quyết cách chia: gộp các thành phần U-Net cùng nằm trong một detection, tách thành phần mà ≥ 2 detection cùng nhận, tuỳ chọn thêm detection U-Net bỏ sót / bỏ thành phần không detection nào chạm). Kết quả rời nhau theo cách dựng. `compare.py` spec `tune:det` và `tune:fuse` (cross-tune grid hysteresis của U-Net trên nửa A trước, rồi grid ghép trên nửa A). Colab: run `mrcnn_s1024_f{k}` (chưa vào `RUN_NAMES`); watcher tải về, không tự nộp. 5 test |
| Thử trên Mac (01/11) | Mask R-CNN **@512, 8 epoch, MPS** (~1.5 h), train trên split 70% của E9a, đo trên 213 ảnh holdout so với hysteresis cross-tune 0.4083: detector một mình **0.250** (−0.158 [−0.173, −0.143]); ghép với U-Net **0.402** (−0.006 [−0.012, +0.000]); grid chọn score ≥ 0.9 + gộp. Log `docs/logs/n4_mrcnn512_holdout_*`. Giới hạn: @512 và 8 epoch yếu hơn nhiều so với bản dự kiến (@1024, 24 epoch); biconcavelens train @2048. Không bác bỏ hẳn N4, nhưng **không có tín hiệu dương nào** |
| Chi phí | GPU ~1.5 h cho fold 0 @1024 (ước tính, chưa đo trên L4); 5 fold ~7 h |
| Tiêu chí | Ensemble (U-Net + Mask R-CNN) hơn U-Net một mình: Δ ≥ +0.012 và CI > 0 trên fold 0 |
| Hạn go/no-go | **2026-11-05**: nếu đến hôm đó fold 0 chưa có kết quả đạt thì bỏ N4. Sau thử nghiệm Mac: chỉ chạy `mrcnn_s1024_f0` nếu còn giờ GPU sau N1/N2/N3' (xếp cuối hàng đợi); không chạy fold 1–4 trừ khi fold 0 qua tiêu chí |
| License | YOLO11-seg (AGPL-3.0) mạnh hơn (0.437) nhưng repo dùng MIT. Chỉ dùng nếu người dùng đồng ý (§ 7) |

### N5 — Metric rubric và phân tích lỗi cho report (📝 Mac, không tốn GPU)

| | |
|---|---|
| Lý do | 70% điểm là định lượng: Dice trung bình (torchmetrics), PQ, phân bố Dice/IoU, tỉ lệ 1:n / n:1 (`S8`). Hiện mới có PQ, Dice tự tính và 1:n / n:1 |
| Việc | Thêm `torchmetrics` (Apache-2.0) vào requirements; tính Dice bằng torchmetrics để khớp cách host tính; vẽ histogram IoU/Dice; bảng theo nhóm annotator. Chạy trên OOF của model final |
| Tiêu chí | Script tái lập được số trong report; có unit test so với cách tính tay |

### N6 — Model final và hồ sơ nộp (📝 từ 2026-11-09)

| | |
|---|---|
| Việc | F1: chọn 2 bài final. F2: weights fp16 trong repo (kiểm tra chênh lệch map fp16 so với fp32 < 1e-3 và PQ không đổi). F3: notebook end-to-end load weights từ repo. F4: report 4 trang. F5: chuyển repo sang public lúc đóng cuộc thi |
| Quy tắc chọn final | Bài 1: pipeline có OOF cross-tune cao nhất. Bài 2: ensemble 5 fold N1 (rủi ro thấp nhất). Nếu hai bài trùng nhau thì bài 2 là phương án tốt thứ nhì theo OOF |

### N8 — Thành viên thứ 4: công thức N1 với seed 2, 5 fold (📝 Colab, người dùng duyệt 05/11)

| | |
|---|---|
| Giả thuyết | Thêm một bộ 5 fold cùng công thức N1 nhưng seed khởi tạo và sampler khác làm thành viên thứ 4 của ensemble cho +0.003–0.005 OOF, như hiệu ứng "thêm thành viên" đo được với BCE (F51: +0.004 trên fold 0+1) |
| Bằng chứng | Mọi đòn bẩy khác còn lại đều ≤ +0.005 và mang tính suy đoán (RESEARCH § 13); đây là cách dùng GPU duy nhất có hiệu ứng đã đo. Ngược lại: ensemble các model yếu hơn trước đây không giúp (Δ −0.0001), nên lợi chỉ có khi thành viên mạnh ngang N1 |
| Thiết kế | `effb4_s1024_s2_f0..f4` = N1 + `--seed 2` (`scripts/train_unet.py`, seed cho torch: init và sampler; rng augmentation vẫn không seed như trước). Chạy trên **Colab** (`colab/train_unet.ipynb`, `RUN_NAMES` đã trỏ vào 5 run này) vì GPU Kaggle đã dùng ~20.8 h. Người dùng upload `build/colab/filseg-colab.zip` (đã đóng gói 05/11) lên `MyDrive/kaggle/filament-segmentation-2026/` rồi chạy notebook; map về dataset riêng `filseg-effb4_s1024_s2_f<k>`. Mình tải về bằng `kaggle datasets download`, **không** đưa vào watcher để khỏi tốn lượt nộp cho từng fold |
| Chi phí | ~75 phút/fold trên L4 (~6.3 h), ~2.3 h/fold trên T4 (`--bs 4`). Phải về trước **08/11** để kịp gate và đóng băng 09/11 |
| Tiêu chí (đăng ký trước) | Trên 707 ảnh OOF, cross-tune `hyst_soft`: ensemble có N8 hơn ensemble không có N8 (cùng các thành viên khác đã qua gate) với **CI 95% > 0**. Đạt thì N8 vào final 1; không đạt thì final 1 giữ nguyên. Kiểm tra hợp lý từng fold: PQ fold của s2 nằm trong ±0.01 so với N1 cùng fold |
| Sau đó | Nộp ensemble có N8 (08/11, lịch § 5b). Nếu chỉ về được 2–3 fold trước 08/11 thì gate trên các fold đó, nhưng chỉ nhận nếu CI > 0 trên tập ảnh đó |

### N9' — Chép công thức Dariush: EfficientNetV2-S, crop 512 ở 2048, bs 16 (✅ qua gate gộp 707 ảnh 09/11: N1 + N2 + N9' +0.0056 [+0.0034, +0.0077]; BCE bị bỏ ở (b). **Final 1 = N1 + N2 + N9'**, OOF 0.4372, LB 0.38)

| | |
|---|---|
| Giả thuyết | Khoảng cách 0.02 LB với đội trung thực là do công thức map, không phải hậu xử lý (F57, F60). Chép công thức có số công khai cho một họ model tốt hơn N1 +0.005…+0.025, và pool +0.006…+0.015 |
| Bằng chứng | Notebook Dariush Afshar (cùng luật): R34 0.4299 / LB 0.37 → V2-S 0.4461 / LB 0.39, 5/5 fold. Ngược chiều: BSergey crop 640 @2048 kém 0.07; N2 của mình (crop 1024 @2048) đơn lẻ −0.005 |
| Thiết kế | `effv2s_c512_f0..f4`: `--res 2048 --crop 512 --bs 16 --lr 3e-4 --encoder tu-tf_efficientnetv2_s.in21k_ft_in1k --targets soft --epochs 184 --val-every 184` (~57 bước/epoch → ~10.5k bước, ckpt cuối), infer cả ảnh 2048 + TTA. Chỉ đổi cờ, không code mới. Khác Dariush còn lại: lịch lr (OneCycle thay warmup + cosine), chuẩn hoá /255 thay z-score theo đĩa, không ép tỉ lệ crop có filament |
| Chi phí | ~1.5–2 h L4 mỗi fold (~1.6 lần N1), ~9 h cho 5 fold; T4 `--bs 8` |
| Tiêu chí (đăng ký trước 06/11) | **Fold 0 (dừng sớm):** chạy tiếp fold 1–4 chỉ khi `hyst_soft` fold 0 đơn lẻ ≥ 0.4163 (N1 f0 0.4158 + 0.0005) **hoặc** N1 + N2 + N9' trên fold 0 hơn N1 + N2 ≥ +0.005. **Vào final:** gate gộp § 5b (a)(b) và (c) một phía |

### N11 / N12 — Thêm thành viên theo công thức N9' (❌ cả hai không qua gate 707, 09/11: N11 −0.0026, N12 −0.0004)

| | |
|---|---|
| Lý do | N9' đơn (5 model) + lọc FP: OOF 0.4322 in-sample, **LB 0.38**, bằng pool 15 model N1 + N2 + N9' (0.4372, LB 0.38) và hơn final 1 (LB 0.37). Công thức crop 512 ở 2048, 184 epoch là thứ duy nhất đưa LB lên khỏi 0.37 |
| Thiết kế | **N11** `effv2s_c512_s2_f0..f4` = N9' với `--seed 2`. **N12** `r34_c512_f0..f4` = cùng công thức, encoder `resnet34` (nửa còn lại của ensemble Dariush: R34 + V2-S 0.4502 so với V2-S 0.4461). Hàng đợi Colab: N11 → N12 |
| Chi phí | N11 ~9 h L4, N12 ~5–6 h; về muộn nhất 11/11 tối |
| Tiêu chí | Gate gộp § 5b (a)(b)(c), so với pool đã chọn tốt nhất lúc đó (N1 + N2 + N9' nếu qua gate 707). Họ chưa đủ 5 fold trước **12/11 06:00 UTC** thì không vào final |
| Lịch | Đóng băng dời từ 09/11 sang **12/11** (người dùng 09/11: "tiếp tục improve"); weights vào repo, notebook tái lập và report dồn vào 12–14/11 |

### N9 / N10 — Thành viên khác encoder (EfficientNetV2-S) và độ phân giải thứ ba (1536) (📝 Colab, 05/11 tối)

| | |
|---|---|
| Giả thuyết | Lợi của một thành viên ensemble tăng theo độ khác biệt với N1 (đo: N2 khác độ phân giải +0.0105, BCE khác loss +0.0018). Một họ khác encoder (N9) hoặc độ phân giải 1536 (N10) cho +0.002…+0.006 mỗi họ, nhiều hơn seed 2 (N8) |
| Bằng chứng | `docs/logs/research06/train_recipe.md`, `ensemble_diversity.md`: Wenzel 2020 (hyper-deep ensembles hơn deep ensembles chỉ khác init); notebook Kaggle 5 R34 + 5 EffNetV2-S OOF 0.450; E9a R34 → B4 +0.013. Ngược lại: N2 đơn lẻ kém N1 (−0.005, +643 FP), N10 có thể thừa hưởng FP; V2-S với 1 kênh chưa từng chạy ở đây |
| Thiết kế | `effv2s_s1024_f0..f4` = N1 với `--encoder tu-tf_efficientnetv2_s.in21k_ft_in1k` (timm, Apache-2.0; build kiểm tra trên Mac: 22.1 M tham số, output đúng kích thước). `effb4_c1536_f0..f4` = N2 với `--res 1536 --crop 1024`. Hàng đợi Colab: N9 → N10 → N8 |
| Chi phí | ~6.5 h L4 mỗi họ; T4 chỉ kịp N9 |
| Tiêu chí (đăng ký trước) | Gate gộp § 5b (a)(b)(c). Không đổi sau khi thấy số |

## 5. Cổng quyết định và lịch dự kiến

| Ngày | Colab (người dùng chạy) | Mac (mình làm) | Cổng |
|---|---|---|---|
| 29–30/10 | Q3: `effb4_s1024_f0`, `effb4_c2048_f0`, `effb4_s1024_f1..f4` | P1 (compare scale), code N3, N5 | – |
| 30/10 | | so sánh N2 trên fold 0 | **G1**: chọn độ phân giải |
| 31/10 | Q4: N3 fold 0 (+ Mask R-CNN fold 0 nếu code xong) | OOF B1, nộp 5 fold N1, code N4 | – |
| 01–02/11 | (Q4 tiếp) | so sánh N3 | **G2**: nhận/bỏ gói v2 |
| 02–05/11 | Q5: 4 fold công thức thắng; Mask R-CNN fold 0 → 4 | ghép ensemble, hiệu chỉnh | **G3 (05/11)**: go/no-go N4 |
| 06–08/11 | fold còn lại | ensemble trên OOF, tune cuối, nộp ứng viên | – |
| 09/11 | – | chọn final, weights fp16, notebook | **Đóng băng** |
| 10–15/11 | – | report, dọn code, kiểm tra tái lập | nộp |

**Thứ tự ưu tiên khi thiếu GPU**: N1 (bắt buộc) → N2 fold 0 (đã xếp hàng) → N3 (rẻ, bằng chứng có
CI) → N4 (lợi lớn nhất nhưng tốn nhất). N4 được viết code song song trên Mac từ đầu để không phải
chờ, nhưng chỉ tiêu giờ GPU khi đã qua G2.

**Nếu Colab lại không ra kết quả**: sau 12 h không có run nào lên Kaggle thì hỏi người dùng log và
hỏi có dùng Kaggle GPU không (30 h/tuần). Không tự ý dùng Kaggle GPU.

## 5b. Lịch 06–15/11: ngày nào nộp gì (bản 2, 05/11 tối, sau workflow research06)

### Xuất phát điểm (05/11 tối)

- **Final 1** = N1 ×5 + N2 ×5 (1:1), hysteresis t 0.4 / seed 0.99 / area 200 / r 12 + lọc FP: OOF
  **0.4284** cross-tune, LB 0.37. Trọng số 2:1 / 1:2 đo trên 707 OOF: +0.0004 / −0.0018 → giữ 1:1.
- **Final 2** = N1 ×5 **UP2048** (map phóng lên 2048 trước hậu xử lý): **+0.0050 [+0.0029, +0.0069]**
  so với N1 @1024 trên 707 OOF (0.4229), qua F48. CSV `outputs/submission_n1up.csv` dựng sẵn.
- 33 lượt đã dùng; 7 bài gần nhất đều 0.37. Còn ~50 lượt (5/ngày, reset 00:00 UTC = 04:00 giờ máy).
- GPU: Kaggle hết (~20.8 h). Colab ~25 h L4 (người dùng chạy). Đóng băng model 09/11.

### Kết luận workflow research06 (4 sweep + critic, `docs/logs/research06/`)

| Sweep | Kết luận | Hệ quả cho lịch |
|---|---|---|
| `train_recipe` | Mọi thay đổi loss/aug/schedule/pseudo-label đã có số âm hoặc ≈ 0 ở đây hoặc ở đội khác. Hướng duy nhất có lý: **thành viên khác encoder** (EfficientNetV2-S in21k, giữ skip stride 2, 22 M tham số, đã kiểm tra build được với `build_model`) | thêm **N9** vào Colab, chạy **trước** N8 |
| `ensemble_diversity` | Lợi của thành viên tăng theo độ khác biệt: độ phân giải (N2) +0.0105 ≫ loss (BCE) +0.0018 > seed (dự báo ≤ +0.002, nhiều khả năng trượt gate CI > 0). Model soup, SWA, logit-average, tune trọng số: ngõ cụt có lý do. Đề xuất **gate gộp** cho các thành viên "rẻ" và một thành viên độ phân giải thứ ba (1536) | thêm **N10** (N2 ở 1536); gate gộp đăng ký trước (dưới) |
| `halpha_preprocessing` | Ngõ cụt: ảnh GONG đã được trạm remap + chuẩn hoá (median đĩa 130–133 ở cả 6 trạm, train = test), lỗi model không phụ thuộc r/R, không paper nào có ablation limb-darkening trước CNN, hai đội Kaggle có/không chỉnh limb bằng nhau | không thêm việc; một đoạn "vì sao không tiền xử lý vật lý" cho report |
| `kaggle_endgame` | LB 2 chữ số trên ~90 ảnh: P(thấy bậc) ≈ 0.12–0.26 cho Δ ≤ 0.003, bằng tỉ lệ của thay đổi rỗng 0.12; thấy chắc chắn khi \|Δ\| ≥ 0.02. Bỏ các bài dò knob; dùng lượt cho **tái lập**, **thang suy giảm** (đo độ dốc LB/OOF) và **ablation cho report**. Chạy notebook end-to-end **sớm** (07/11, không đợi 11/11). Final 2 nên là bước đã qua gate trước đó, không phải N1 đơn | lịch nộp viết lại; final 2 xét lại ở 09/11 |

### Luật dùng lượt nộp (thay bản 05/11)

Một bài chỉ được nộp nếu thuộc một trong ba loại, và dòng giả thuyết + Δ OOF dự báo được ghi vào bảng
submission (PROGRESS) **trước** khi nộp:

1. **Tái lập**: CSV từ notebook Kaggle / clone sạch phải trùng từng byte với CSV từ Mac và cho cùng LB.
2. **Đo có độ phân giải**: \|Δ OOF dự báo\| ≥ 0.01 (thang suy giảm, model đơn so với ensemble).
3. **Số cho report**: LB của từng thành viên / từng fold (bảng "ensemble so với model đơn trên test").

Knob hậu xử lý (trọng số, min_area, seed_thresh, bán kính gộp, cc/hyst, có/không lọc FP) **không**
nộp nữa: OOF đã trả lời, LB không thấy. Vì vậy hai bài `min_area 120/300` và `2:1 / 1:2` của lịch cũ
bị bỏ (CSV đã dựng vẫn giữ trong `outputs/`). Lượt không có bài hợp lệ thì để trống — dự kiến dùng
~25 / 50.

### Gate gộp cho thành viên mới (đăng ký trước 05/11, trước khi có số N8/N9/N10)

Ứng viên: BCE (đã có), N8 (seed 2), N9 (V2-S), N10 (1536). Mỗi họ 5 fold trung bình trước, các họ
trộn trọng số bằng nhau; trên 707 OOF, cross-tune `hyst_soft`, bootstrap 2000 theo ảnh.

- (a) Pool = N1 + N2 + mọi họ đã về, so với N1 + N2: **CI 95% > 0**.
- (b) Mỗi họ trong pool: đóng góp biên (pool so với pool bỏ họ đó) có **P > 0 ≥ 0.8**; họ nào trượt
  thì bỏ họ có đóng góp biên thấp nhất rồi kiểm lại (a)(b) **một lần**; vẫn trượt thì final 1 giữ N1 + N2.
- (c) Hợp lý từng fold: PQ fold của họ mới **≥** N1 cùng fold − 0.02 (V2-S, 1536, N9') hoặc − 0.01
  (seed 2). *(Sửa 06/11, trước khi có số nào của N8/N9/N9'/N10: bản cũ ±0.02 hai phía sẽ loại cả một
  họ tốt hơn hẳn N1.)*
- Họ về chỉ 2–3 fold trước 09/11: không vào final (OOF thiếu ảnh → không so được trên 707 ảnh).

### Hàng đợi Colab (người dùng chạy; bundle đóng gói lại 05/11 tối)

| Thứ tự | Run | Chi phí L4 | Về muộn nhất |
|---|---|---|---|
| 0 | **N9'** `effv2s_c512_f0..f4` (công thức Dariush, thẻ N9'; fold 0 là pilot dừng sớm) | ~9 h | fold 0: 06/11; đủ: 08/11 |
| 1 | **N9** `effv2s_s1024_f0..f4` (N1 + encoder `tu-tf_efficientnetv2_s.in21k_ft_in1k`) | ~6.5 h | 07/11 |
| 2 | **N10** `effb4_c1536_f0..f4` (N2 ở `--res 1536 --crop 1024`) | ~6.5 h | 08/11 |
| 3 | N8 `effb4_s1024_s2_f0..f4` (seed 2) | ~6.3 h | 08/11 tối |

Nếu người dùng đã bắt đầu N8 trước khi tải bundle mới thì cứ để N8 chạy xong; queue mới bỏ qua run đã
xong. Trên T4: `--bs 4`, ~2.3 h/fold → chỉ kịp **một** họ trước 09/11 (chọn N9).

### Lịch theo ngày (ngày UTC)

| Ngày | Mac / Colab | Lượt nộp (loại · giả thuyết) | Cổng |
|---|---|---|---|
| **06/11** Thứ 6 | Người dùng tải bundle mới, chạy Colab (N9 → N10 → N8). Mac: dry-run gate gộp trên N1 + N2 + BCE. Bắt đầu `notebooks/final_inference.ipynb`. *(Đã làm 05/11 tối: thang suy giảm trên 707 OOF: bỏ 15/30/50% instance điểm lọc FP thấp nhất → Δ +0.005 / −0.019 / −0.085; gộp theo hướng L16 r6 qua F48 → final 1 mới.)* | 1. **final 1 + gộp theo hướng** `submission_ens_n1n2_orient.csv` (OOF +0.0021 cross-tune, sau lọc FP 0.4312 → 0.4327; LB dự báo 0.37) · 2. **final 2 = N1 UP2048** `submission_n1up.csv` (+0.005; LB dự báo 0.37–0.38) · 3–5. **thang suy giảm** `submission_ens_n1n2_drop{15,30,50}.csv` (Δ OOF +0.005 / −0.019 / −0.085; dự báo LB 0.37 / ~0.35 / ~0.30) | thang: drop30 và drop50 phải thấp hơn rõ và đúng thứ tự, không thì nghi lỗi file nộp |
| **07/11** Thứ 7 | N9 về → gate gộp bước 1 (N1 + N2 + BCE + N9). Notebook end-to-end chạy trên kernel Kaggle mới (repo + weights fp32 qua dataset riêng, không Internet), ghi SHA256 CSV, đo thời gian | *(Sửa 06/11: notebook chưa xong lúc reset nên 07/11 nộp bảng model đơn, notebook dời sang 08/11.)* 1–4. **N1 fold 1, 2, 3, 4 đơn** (loại 3; fold 0 đã 0.36) · 5. **N2 5 fold đơn** (loại 3, dời từ 06/11) | notebook đạt nếu `cmp` trùng; lệch thì so map uint8 và ghim phiên bản torch/timm/smp |
| **08/11** CN | N10, N8 về → gate gộp đầy đủ (a)(b)(c). Bảng OOF mọi tổ hợp. Rubric N5 cho các ứng viên | 1. **pool đã qua gate** (loại 3: chỉ ghi số, không quyết định) · 2. **N9 5 fold đơn** (loại 3) · 3–4. **N1 fold 3, fold 4 đơn** (loại 3) · 5. N10 5 fold đơn nếu đã về (loại 3) *(Thực tế 08/11: chưa họ mới nào về, N9' fold 0 bắt đầu 08:26 UTC. Nộp: ablation mode cc (0.36), final 1 và final 2 dựng lại bằng notebook trên kernel Kaggle (0.37 / 0.37, CSV trùng từng byte). Còn 2 lượt không dùng: không còn ứng viên đúng quy tắc.)* | gate gộp; chưa đủ 5 fold thì họ đó ở ngoài |
| **09/11** Thứ 2 | **Đóng băng**, tag git `freeze`. Chọn theo OOF: final 1 = pipeline OOF cao nhất trong số đã qua gate; final 2 = bước đã qua gate ngay trước đó (ví dụ N1 + N2 nếu pool vào final 1; N1 UP2048 nếu không có gì mới). Xuất notebook cho cả hai | 1. final 1 từ notebook (loại 1) · 2. final 2 từ notebook (loại 1) · 3. N8 5 fold đơn nếu về (loại 3) · 4–5. trống | chốt 2 final |
| **10/11** Thứ 3 | Weights vào `weights/` + `SHA256SUMS`: fp32 nếu tổng ≤ ~1 GB (10 model = 817 MB; mỗi file 81.7 MB < 100 MiB), quá thì fp16 (plan B, gate: \|Δ PQ OOF\| < 0.0005 và LB bằng). Không dùng Git LFS (băng thông 10 GiB/tháng). Clone sạch + pytest | 1–2. final 1 / final 2 từ **clone sạch** (loại 1) · 3–5. trống | trùng byte |
| **11/11** Thứ 4 | Notebook chạy từ tag `freeze` trên kernel mới; ghi phiên bản torch/timm/smp của kernel vào README | 1–2. dự phòng cho sửa tái lập | – |
| **12/11** Thứ 5 | Report 4 trang bản 1: phân rã lỗi (F52), inter-annotator 0.34, bảng thành viên / fold OOF + LB, thang suy giảm, "vì sao không tiền xử lý vật lý", các hướng đã loại có số | trống | – |
| **13/11** Thứ 6 | *(Chốt 12/11: final 1 = kernel `submission_final1.csv` ref 58174035; final 2 = N1 + N9' CSV local ref 58269004.)* Report bản cuối; **tick 2 final bằng tay** trên Kaggle (không để Kaggle tự chọn giữa 7 bài 0.37), chụp màn hình; tag `final` | không nộp sau khi tick | – |
| **14/11** Thứ 7 | Không đổi code, không nộp | – | – |
| **15/11** CN | Hạn 06:00 UTC. Sau khi đóng: repo sang public, thử clone ẩn danh, nộp Google form buổi sáng UTC | – | – |

### Kỳ vọng thật

Nếu N9 và N10 cùng qua gate gộp: OOF final 1 khoảng **0.430–0.436** (+0.002…+0.008); một mình N8
nhiều khả năng không qua. LB gần như chắc vẫn 0.37–0.38: khoảng cách val → LB chủ yếu do lệch annotator
(inter-annotator PQ 0.34), và LB không thấy Δ < 0.01. Phần đáng giá nhất của tuần này là **tái lập
chắc chắn** và **report có số** (rubric 30% định tính), không phải thêm điểm LB.

<details><summary>Lịch bản 1 (05/11 sáng)</summary>

### Bản 1 (05/11 sáng, đã thay bằng bản 2 ở trên)

#### Xuất phát điểm (05/11)

- OOF tốt nhất: N1+N2 (10 model) **0.4294** cross-tune; LB 0.37 (bằng N1 5 fold). BCE làm thành viên
  thứ ba: +0.004 trên fold 0+1, đang chạy gate trên 707 ảnh.
- Nghiên cứu 05/11 (RESEARCH § 13, `docs/logs/research05/`): mọi đòn bẩy còn lại đều **≤ +0.005 OOF**.
  Đã thử và âm hoặc ≈ 0 trong ngày: chỉnh biên theo cường độ ảnh (−0.001 đến −0.05), ngưỡng theo
  kích thước (+0.0001), gộp theo hướng (−0.001), làm mượt (−0.0003). Còn dương nhẹ: gộp mảnh bằng GBM
  (+0.0036, CI chạm 0). Paper trong ngành không có cách tách instance nào hơn heuristic, và Hao & Li
  2026 đo trên MAGFiLO thấy các loss skeleton/component **không** cho lợi ổn định.
- GPU: đã dùng ~20.8 h Kaggle trong ~20 h người dùng cho. Quota Kaggle là 30 h/tuần (thường reset
  thứ Bảy 00:00 UTC, cần kiểm tra trên trang quota). Train thêm cần người dùng đồng ý (§ 7 mục 4).
- Lượt nộp: 5/ngày, reset 00:00 UTC (04:00 giờ máy). Từ 05/11 đến hạn còn ~50 lượt.

#### Cách dùng lượt nộp (sửa quy tắc § 3 mục 6)

LB chỉ hiện 2 chữ số và nhiễu ±0.01 (`docs/logs/lb_noise_effb4_hyst.txt`), nên **một** bài nộp không
phân biệt được hai phương án cách nhau < 0.01, và LB vẫn **không quyết định** gì. Nhưng lượt nộp
không khan hiếm, nên dùng hết 5 lượt/ngày cho ba việc mà OOF không đo được hoặc không thay được:

1. **Hiệu ứng chỉ có trên test**: lợi do trung bình nhiều model (OOF là model đơn), TTA, lệch
   annotator giữa train và test. Đo bằng **xu hướng qua nhiều điểm** (ví dụ 3 ngưỡng trong một ngày),
   không bằng một bài.
2. **Kiểm tra tái lập**: weights fp16, notebook end-to-end chạy trên Kaggle phải cho đúng LB của bài
   nộp từ Mac.
3. **Ablation cho report**: bỏ từng thành phần của final để có bảng "mỗi thành phần đóng góp gì" trên
   cả OOF và LB (rubric 30% định tính coi trọng lập luận rõ).

Mỗi bài nộp ghi **trước** vào bảng lịch sử submission (PROGRESS) một dòng: giả thuyết và Δ OOF dự
báo; sau khi có LB ghi kết quả. Không nộp bài nào không có giả thuyết.

#### Lịch theo ngày (ngày UTC)

| Ngày | Việc trên Mac / GPU | 5 lượt nộp (giả thuyết) | Cổng |
|---|---|---|---|
| **05/11** Thứ 5 | Gate BCE trên 707 OOF (đang chạy); gộp GBM v1/v2 (đang chạy); tối: đo TTA (flip/rot) trên fold 0 bằng Mac; viết RESEARCH § 13, commit | 1. N1+N2+BCE 15 model (15 so với 10 model: +0.00–0.01) · 2. N1+N2 + gộp GBM (≈ +0.003) · 3. N1+N2+BCE + gộp GBM · 4. N1+N2 thresh 0.3 · 5. N1+N2 thresh 0.5 (điểm P/R: nếu test thưa annotator hơn train thì 0.5 ≥ 0.3) | BCE vào final 1 nếu CI > 0 trên 707 ảnh. Gộp GBM vào nếu CI > 0 **và** Δ > 0 ở ≥ 4/5 fold |
| **06/11** Thứ 6 | *(Sửa 05/11: mọi map đã có TTA 8 hướng từ lúc infer, nên "TTA" là TTA theo tỉ lệ. Screen TTA1152 trên fold 0 đã chạy ngay 05/11: **−0.0031**, trung bình 2 tỉ lệ −0.0002 → bỏ.)* UP2048 chỉ liên quan **final 2** (N1 đơn; final 1 vốn chạy ở 2048 vì N2 là map 2048): đo trên 707 OOF N1, nhận theo luật F48 (CI > 0). Tune trọng số N1:N2 trên 707 OOF (chạy từ 05/11) | 1. final 2 = N1 5 fold UP2048 (+0.0024 fold 0) · 2. N1:N2 = 2:1 · 3. N1:N2 = 1:2 · 4. min_area 120 · 5. min_area 300 (điểm P/R theo kích thước) | UP2048 vào final 2 / trọng số vào final 1 chỉ khi CI > 0 trên 707 OOF |
| **07/11** Thứ 7 | **Quyết định GPU** (§ 7 mục 4). Nếu được: đẩy N8 = công thức N1, seed 2, fold 0–4 (2 kernel ≈ 9 h T4), cách dùng GPU duy nhất có lợi đo được (+0.004/thành viên, F51). Song song: tune trọng số ensemble trên OOF | 1. trọng số N1:N2 tune OOF · 2. N1:N2 = 2:1 · 3. N1:N2 = 1:2 · 4. seed_thresh 0.9 · 5. group_radius 6 (xu hướng gộp lỏng/chặt trên test) | N8 chỉ chạy khi người dùng đồng ý và quota còn |
| **08/11** CN | N8 về (nếu chạy): gate thành viên thứ 4 trên OOF. Chạy rubric N5 cho các ứng viên | 1. ensemble 4 thành viên · 2. 4 thành viên + TTA · 3. final 1 bỏ gộp (r = 0) · 4. final 1 mode cc (không hysteresis) · 5. final 1 không lọc FP (ablation) | thành viên 4 vào nếu CI > 0 trên 707 OOF |
| **09/11** Thứ 2 | **Đóng băng**. Bảng OOF cuối cho mọi tổ hợp đã qua gate; chọn **final 1** và **final 2** theo OOF cross-tune | 1. final 1 · 2. final 2 · 3–5. model đơn fold 1, 2, 3 (bảng "ensemble so với model đơn" trên test cho report) | Chốt 2 final |
| **10/11** Thứ 3 | Weights fp16 vào repo; đo map fp16 so với fp32 (max \|Δ\|, PQ fold 0); dọn `configs/` | 1. final 1 từ weights fp16 (phải bằng LB fp32) · 2. final 2 fp16 · 3. N2 5 fold một mình · 4. BCE 5 fold một mình · 5. dự phòng | fp16 đạt nếu LB bằng và PQ fold 0 lệch < 0.001 |
| **11/11** Thứ 4 | Notebook end-to-end chạy trên kernel Kaggle mới (chỉ repo + ảnh test) → CSV; đo thời gian inference | 1. CSV từ notebook final 1 (phải trùng) · 2. CSV từ notebook final 2 · 3–5. dự phòng cho sửa lỗi tái lập | Notebook đạt nếu CSV trùng từng byte với bài nộp từ Mac |
| **12/11** Thứ 5 | Report 4 trang bản 1 (rubric, phân rã lỗi, ablation, so với inter-annotator 0.34); bảng kết quả README | ablation còn thiếu cho hình trong report; còn lại để dự phòng | – |
| 13–14/11 | Report bản cuối; chọn 2 final trên Kaggle; chuẩn bị Google form | chỉ nộp lại nếu phát hiện lỗi | – |
| 15/11 | Hạn 06:00 UTC. Sau khi đóng: repo sang public, nộp form | – | – |

#### Kỳ vọng thật

Nếu BCE, gộp GBM, TTA và N8 đều qua gate thì OOF vào khoảng **0.435–0.44**; LB nhiều khả năng vẫn
0.37–0.38 vì LB hiện 2 chữ số và lệch annotator chiếm phần lớn khoảng cách val → LB. Phần lớn
1 − PQ còn lại là bất đồng giữa annotator (inter-annotator PQ 0.34, F31: 59% dự đoán gần trúng là TP
của annotator khác), nên không kỳ vọng bước nhảy nào từ hậu xử lý. Các hướng cần GPU và mang tính
suy đoán (head distance kiểu Omnipose, head affinity + mutex watershed, lớp "khe hở") **không** làm
trong tuần này: mỗi hướng tốn ~9 h GPU cho hiệu ứng dự kiến dưới ngưỡng phát hiện (±0.009).

</details>

## 6. Không làm (và lý do)

| Việc | Lý do | Nguồn |
|---|---|---|
| Thêm biến thể hậu xử lý / nộp để dò LB | 9 lượt nộp gần nhất đều trong khoảng 0.35–0.36, đúng như CI dự báo; hậu xử lý đã cạn | [EXPERIMENTS § Tự phản biện](EXPERIMENTS.md#tự-phản-biện-và-kiểm-chứng-2026-10-25--scriptscomparepy) |
| Ensemble model holdout yếu | Δ −0.0001 [−0.008, +0.008] | `docs/logs/critique_models.txt` |
| Thêm đặc trưng cho bộ lọc instance (đồng thuận model, cây quyết định H6) | đồng thuận −0.0004 [−0.003, +0.002]; cây −0.004 [−0.011, +0.003]; trên hyst chính bộ lọc 17 đặc trưng cũng 0.000 | `docs/logs/wf_agree_filter.txt` |
| clDice, CLAHE, pseudo-label test, SimSiam, Mask2Former, SAM | biconcavelens đã thử và kém đi | `S9e` |
| Closing/dilate mask theo instance, co/giãn cả submission | umerkang66: −0.003 đến −0.016; public 0.37/0.35 | `S11e` |
| Chuyển nhãn sang frame GONG lân cận | model ngữ nghĩa một mình 0.404 → 0.372; trong hệ thống đầy đủ không đổi | `S9d` |
| MAGFiLO 1.0 công khai, mask chồng nhau, tune theo LB | Cấm hoặc vô nghĩa | `S1`, `S3` |

## 6b. Kết quả workflow nhiều agent 30/10

Chi tiết ở [RESEARCH § 10](RESEARCH.md#10-workflow-nhiều-agent-3010-tìm-ý-tưởng-cải-thiện). Tóm tắt
cho kế hoạch:
- Tầng chọn instance trên map hiện tại đã cạn: H6 bị bác bỏ (−0.004, oracle +0.125 không học
  được), bộ lọc đồng thuận model cho −0.0004. **Không làm thêm hậu xử lý.**
- N3 đổi thành **N3'** (nhánh `exp/train-v2`): blob Dice + phạt thành phần không chạm GT + EMA; bỏ
  input limb (lỗi vị trí chỉ 0.4%). Tiêu chí mới: luôn chạy fold 0 và fold 1, vì CI bootstrap không
  tính nhiễu seed.
- N2 có thêm điều kiện về nhóm lỗi (xem thẻ N2).
- N4 giữ cổng G3. Lỗi gộp/cắt chiếm ~25% FN là mục tiêu của nó.
- Kỳ vọng trung thực: các bước còn lại cộng lại khoảng +0.01–0.03 OOF; chưa có ý tưởng nào hứa hẹn
  bước nhảy lớn.

## 6c. GPU Kaggle từ 03/11 (người dùng cho phép, ~20 h tuần này)

Colab vẫn chưa chạy, nên từ 03/11 GPU đi qua Kaggle. `scripts/kaggle_gpu.py` làm các việc sau:
- đưa code lên dataset riêng tư `filseg-code`;
- mỗi kernel (T4 ×2) train **2 run song song**, mỗi GPU một run;
- ghi map giống hệt Colab;
- `fetch` chuyển map về `outputs/colab/<run>` kèm marker LOCAL, rồi watcher xử lý như cũ.

`daemon` đẩy kernel theo thứ tự ưu tiên mỗi khi có slot trống: tài khoản có tối đa 2 phiên GPU,
và các project khác của người dùng cũng dùng chung slot. Mọi run dùng `--bs 4`, vì bs 6 có nguy cơ
tràn bộ nhớ trên T4 16 GB. Hai nhánh của mỗi so sánh luôn cùng bs.

| Kernel | Run (GPU0 / GPU1) | Mục đích | Ước tính |
|---|---|---|---|
| 1 | N1 f0 / N1 f1 | đối chứng G2, bắt đầu 5 fold | ~2.5 h |
| 2 | v2 f0 / N2 c2048 f0 | G2 (nhánh thử), G1 | ~3 h |
| 3 | v2 f1 / N1 f2 | G2 (fold thứ 2), 5 fold | ~2.8 h |
| 4 | N1 f3 / N1 f4 | xong 5 fold N1 (final an toàn) | ~2.5 h |
| (sau G1/G2) | rollout công thức thắng: 3–4 fold | | ~5–6 h |

Tổng ~11 h cho 4 kernel đầu, ~17 h nếu có rollout. Nếu GPU Kaggle tính quota theo thời gian phiên
(không tính theo số GPU) thì nằm trong ~20 h. Chưa xác nhận điều này; sau kernel đầu tiên cần xem
đồng hồ quota trên Kaggle.

**Cổng được sửa**:
- **G1 (N2)**: giữ tiêu chí Δ ≥ +0.012 và CI > 0, chấm bằng grid `hyst_soft`. Thêm chẩn đoán
  F36: phần tăng phải tập trung ở tam phân vị sắc nét và ở recall filament có người khác cùng vẽ.
  Nếu tăng đều mọi tam phân vị thì nghi nhiễu seed. Chỉ một fold, nên nếu qua thì cần thêm fold 1
  @2048 trước khi rollout.
- **G2 (v2)**: như thẻ N3', chấm bằng `hyst_soft` (F43) trên fold 0 + 1.
- **Đo miễn phí sau G2**: trung bình map N1 + v2 trên fold 0/1, so với model đơn tốt hơn. Đây là
  số đo thật duy nhất về lợi ích của ensemble.

**Kết quả (04/11)**: N1 5 fold xong (OOF hyst 0.4208, LB 0.37). G1 không qua (N2 đơn −0.0018
[−0.011, +0.007]). G2 bỏ (v2 −0.050 / −0.080). Đo miễn phí trên fold 0: **N1 + N2 +0.0133
[+0.0066, +0.0203]**, N1 + v2 +0.0109 [+0.0053, +0.0170]. Chi tiết ở ledger (ENS-n1n2, ENS-n1v2).

**Hàng đợi tiếp theo** (đẩy 04/11 07:45, ~6 h GPU, tổng ~14 h / 20 h):

| Kernel | Run (GPU0 / GPU1) | Mục đích |
|---|---|---|
| 5 | N2 f1 / N2 f2 | thành viên ensemble N2 |
| 6 | N2 f3 / N2 f4 | thành viên ensemble N2 |

**Cổng G4 (ensemble N1 + N2)**, đăng ký trước khi có số:
- so `ens=avg(N1 fk, N2 fk)` với N1 fk trên từng fold, grid `hyst_soft` cross-tune, gộp 707 ảnh;
- nhận nếu Δ ≥ +0.010, CI > 0, và Δ > 0 trên ít nhất 4/5 fold;
- nếu qua: final 1 = 10 model (N1 ×5 + N2 ×5), trọng số 1:1, hậu xử lý tune trên OOF ensemble;
  final 2 = N1 5 fold (an toàn, LB 0.37). Thời gian inference test: 10 model × TTA 8, cần đo trên
  Kaggle trước khi đóng băng 09/11;
- nếu không qua: final 1 = N1 5 fold.
- Lưu ý: OOF không đo được phần lợi do trung bình nhiều fold trên test (5 model N1 đã được trung
  bình ở test), nên lợi ích thật của N2 trên test có thể nhỏ hơn số OOF. Không dùng LB để quyết.

**Kết quả G4 (04/11)**: qua. OOF 707 ảnh +0.0105 [+0.0072, +0.0140], dương 5/5 fold. Final 1 là
ensemble 10 model. Việc còn lại cho final: đo thời gian inference 10 model × TTA 8 trên Kaggle,
weights fp16 (10 × ~35 MB, kiểm tra giới hạn repo), notebook end-to-end, metric rubric (N5) trên
final. GPU còn ~6 h: chưa dùng; ứng viên rẻ nhất là một họ thứ ba khác biệt (ví dụ EfficientNetV2-S
@1024, F32) nếu thời gian cho phép, nhưng cần cổng riêng trước khi đẩy.

**Thẻ N7: map hiệu chỉnh (BCE, bỏ Dice)**, đăng ký 04/11 trước khi có số.
- Lý do (rà soát 04/11): target mềm là tỉ lệ annotator, nhưng loss Dice đẩy map về 0/1. Trên 60 ảnh
  fold 0 của N1: 70% thành phần có đỉnh p = 1.0; pixel dự đoán 0.5–0.9 chỉ có 37–45% annotator vẽ.
  Map bão hoà thì không biết filament nào chỉ một người vẽ, nên luật giữ tối ưu F26 không áp được và
  bộ lọc chỉ đạt AUC 0.72 (F34).
- Thiết kế: `effb4_bce_s1024_f0/f1` = N1 với `--dice-weight 0`, cùng bs 4, 30 epoch; một kernel
  (~2.3 h GPU). So với N1 f0/f1 bằng `hyst_soft`, và đo ensemble N1 + N2 + BCE so với N1 + N2.
- Tiêu chí: nhận làm công thức thay N1 nếu gộp 2 fold Δ ≥ +0.008, CI > 0, dương cả 2 fold; nhận
  làm thành viên ensemble thứ ba nếu N1 + N2 + BCE hơn N1 + N2 với CI > 0 trên 2 fold. Nếu qua, train
  fold 2–4 (~3.5 h, vượt ~1 h so với 20 h, cần hỏi người dùng).
- Kỳ vọng: không chắc. Dice thường giúp vật mảnh; BCE thuần có thể mất recall.
- **Kết quả (04/11 22:28)**: không qua cả hai tiêu chí. Thay N1: −0.0002 [−0.010, +0.009] (f0),
  −0.0018 [−0.012, +0.008] (f1). Thành viên thứ ba: +0.0060 [+0.0012, +0.0110] (f0), +0.0038
  [−0.0014, +0.0093] (f1). Gộp 284 ảnh, ngoài đăng ký: +0.0041 [+0.0004, +0.0077]. Map hiệu chỉnh tốt
  (seed chọn 0.6–0.7), nhưng bộ lọc trên map mới vẫn ≈ 0. Fold 2–4 chỉ train nếu người dùng duyệt
  thêm giờ GPU (lợi kỳ vọng ~+0.004 OOF; inference 15 model).
- **05/11: người dùng duyệt** train BCE fold 2–4 (kernel `bce-f2f3` + `bce-f4`, chạy song song,
  ~4.5 h giờ phiên GPU vì kernel 1 run vẫn tính cả phiên; tổng ~20.8 h). Tiêu chí đăng ký trước khi có
  số: nhận BCE làm thành viên thứ ba của final 1 nếu trên OOF 707 ảnh N1 + N2 + BCE hơn N1 + N2 với
  CI > 0 (cross-tune `hyst_soft`), **và** inference 15 model trên Kaggle vừa giới hạn thời gian
  notebook. Nếu không qua, final 1 giữ nguyên 10 model.
- **Kết quả gate (05/11 11:40, `docs/logs/kgpu04/n7ens_oof.txt`)**: trên 707 ảnh OOF, N1 + N2 + BCE
  0.4302 so với N1 + N2 0.4284: **+0.0018 [−0.0008, +0.0045], P>0 0.91**. CI chứa 0 → **không qua**,
  final 1 giữ 10 model. Hiệu ứng +0.004 trên fold 0+1 không giữ được trên fold 2–4 (BCE giảm 100 FP
  nhưng mất 25 TP, thêm 25 FN). Bài 15 model vẫn nộp một lần làm điểm dữ liệu LB "15 so với 10 model"
  (lịch § 5b), không vào final. Bài học: hiệu ứng +0.004 đo trên 2 fold không đủ để duyệt GPU; ngưỡng
  F48 (Δ ≥ +0.005, CI > 0 trên 2 fold) phải giữ nghiêm cho N8.

**Việc trên Mac, không tốn GPU**: TTA @1152 trên đủ 213 ảnh (F38; chỉ nhận nếu CI > 0); thêm chia
theo độ sắc vào báo cáo so sánh.

## 7. Cần người dùng quyết định

1. ~~Số giờ GPU Colab còn lại~~: **đủ ~25 h** (trả lời 29/10). Làm theo kế hoạch đầy đủ; dự phòng chỉ còn ~3 h, nên một run hỏng phải được phát hiện sớm.
2. ~~GPU Kaggle~~: **được dùng từ 03/11**, ~20 h tuần này, đẩy kernel khi có slot trống (trả lời 03/11).
3. **YOLO11-seg (AGPL-3.0)** hay chỉ Mask R-CNN (BSD)?
4. ~~GPU thêm cho N8~~: **trả lời 05/11: "GPU hết thì viết code qua thí nghiệm trên Colab"**. N8
   chạy trên Colab (thẻ N8); người dùng upload bundle và chạy notebook.
5. **Quy tắc "chỉ dùng mask" trong CLAUDE.md**: host đã cho phép spine/bbox/area (S20), nhưng
   nghiên cứu 05/11 cho thấy chúng gần như không mang thông tin mới so với mask, nên **đề nghị giữ
   nguyên** quy tắc.

Theo dõi thêm: [discussion 753665](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753665)
(`S18`). Host chưa trả lời về barb, filament đứt đoạn, và việc để ~800 MB weights thẳng trong repo.
Câu trả lời có thể đổi cách tách instance và cách lưu weights.

## 8. Các kế hoạch cũ đã thay thế

- R1–R9 (25/10): R1–R3 (5 fold, target mềm, crop 2048) chuyển thành N1/N2. R4/R5/R4b đã xong (xem
  EXPERIMENTS). R6 (ensemble encoder) bị bỏ vì ensemble model yếu không giúp. R7 chuyển thành N3,
  R8 thành N4, R9 thành N6.
- N0–N6 (sáng 29/10): giữ nguyên ý, chuyển thành các thẻ có tiêu chí ở trên. N0 (mở lại GPU) xong
  phần code: đã sửa bug cache và đóng bundle mới.

## Hồ sơ nhận giải (bắt buộc)

- [x] License MIT
- [x] `requirements.txt` pin version
- [x] README: cài đặt, môi trường, thời gian chạy
- [ ] Notebook chạy toàn bộ pipeline, **load weights có sẵn trong repo** (N6-F3)
- [x] Weights trong repo (N6-F2): **fp32**, 15 file trong `weights/` + `SHA256SUMS` (1.26 GB, mỗi file ≤ 85 MiB < 100 MiB), không dùng Git LFS; người dùng chọn fp32 để giữ trùng byte (10/11). Clone sạch từ GitHub khớp 15/15 hash
- [ ] Technical report 4 trang theo template Overleaf của host (N6-F4)
- [ ] Repo public ngay khi cuộc thi đóng, giữ public đến khi công bố (N6-F5)
- [ ] Nộp Google form trước Nov 15 11:59 pm CDT
- [ ] Hiểu và giải thích được mọi thành phần (luật cho phép dùng AI nhưng yêu cầu điều này)
