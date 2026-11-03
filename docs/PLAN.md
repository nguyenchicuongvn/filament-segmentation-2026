# Kế hoạch (bản 2026-10-29, thay mọi bản trước)

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

## 7. Cần người dùng quyết định

1. ~~Số giờ GPU Colab còn lại~~: **đủ ~25 h** (trả lời 29/10). Làm theo kế hoạch đầy đủ; dự phòng chỉ còn ~3 h, nên một run hỏng phải được phát hiện sớm.
2. **Có cho dùng GPU Kaggle** khi Colab không đủ hay hỏng không?
3. **YOLO11-seg (AGPL-3.0)** hay chỉ Mask R-CNN (BSD)?

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
- [ ] Weights trong repo (fp16, mỗi file < 100 MB) (N6-F2)
- [ ] Technical report 4 trang theo template Overleaf của host (N6-F4)
- [ ] Repo public ngay khi cuộc thi đóng, giữ public đến khi công bố (N6-F5)
- [ ] Nộp Google form trước Nov 15 11:59 pm CDT
- [ ] Hiểu và giải thích được mọi thành phần (luật cho phép dùng AI nhưng yêu cầu điều này)
