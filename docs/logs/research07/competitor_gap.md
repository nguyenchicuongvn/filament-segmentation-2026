# Research 07/11 — `competitor_gap`: vì sao đội trung thực đạt OOF 0.45 / LB 0.39–0.40 còn mình 0.433 / 0.37

Agent `competitor_gap` (06/11). Phạm vi: đọc mọi pipeline công khai, so từng khâu với pipeline của mình, đo xem
OOF có so được không, và đo trên OOF N1 + N2 (707 ảnh) những khác biệt rẻ nhất. Script và output nằm ở
`/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/competitor_gap/`
(`extract.py`, `analyze.py`, `oracle.py`, `analyze.log`, `oracle.log`). Repo clone chỉ đọc, không chạy.

## 0. Tóm tắt (đọc phần này là đủ)

1. **OOF của các đội so được với OOF của mình.** Cả 4 đội có code hoặc receipt (theainerd, Dariush, umerkang,
   tushhaaarrrr) đều chấm như host: gộp qua 1,154 bản gán nhãn, 8,199 GT, IoU > 0.5, fold theo ảnh (theainerd
   còn gộp theo ngày, tức chặt hơn mình). Phần tune tham số in-sample của họ và của mình chỉ lệch nhau khoảng
   0.002. Vậy khoảng cách **0.013–0.027 OOF là thật**, không phải do cách đo.
2. **LB ≈ OOF − 0.055 ở mọi đội** (8 điểm dữ liệu, bảng § 2). Muốn đạt 0.40 trên LB thì cần OOF khoảng
   ≥ 0.452. Hậu xử lý không thể đưa mình tới mức đó (điểm 3).
3. **Khoảng cách nằm ở map, không nằm ở hậu xử lý** (đo trên OOF của mình):
   - Áp đúng luật của Dariush (cc p > 0.35, peak > 0.99, area ≥ 200, mean p ≥ 0.85) lên map N1 + N2 cho
     **0.4237**, kém luật của mình (0.4313). Trong khi cùng luật đó trên map 5 × EfficientNetV2-S của Dariush cho
     **0.4461 nested**.
   - Bộ chấm lại kiểu theainerd (ứng viên rộng + GBM hồi quy u = IoU khớp kỳ vọng theo annotator + lõi p > 0.5,
     τ nested) cho 0.4325, đúng bằng final 1 (0.4327). Dùng cùng bộ chấm đó trên ứng viên cc p > 0.35 đơn giản:
     **0.4348, +0.0035 [−0.0008, +0.0078], dương 5/5 fold**. Đây là phần lợi duy nhất ở hậu xử lý, và nhỏ.
   - Một họ model của Dariush có **+255 TP, +166 FP, SQ 0.684 so với 0.676** so với 10 model của mình. Tức là
     map của họ vừa tìm được nhiều filament hơn, vừa vẽ biên đúng hơn.
4. Có ba thay đổi có khả năng giải thích phần lớn khoảng cách. Mỗi thay đổi đều có một thí nghiệm rẻ:
   (A) **encoder EfficientNetV2-S in21k, train trên crop 512 ở độ phân giải gốc, infer cả ảnh 2048**. Bằng chứng
   mạnh nhất: một họ model của Dariush đạt 0.4461, cao hơn N1 + N2 của mình +0.018 dù hậu xử lý đơn giản hơn.
   (B) **tầng quyết định chạy trên ứng viên rộng hoặc cây ngưỡng**, thay cho hysteresis rồi mới lọc. Đã đo:
   +0.0035. (C) **refiner theo từng instance trên crop native** (stage 2): hai đội độc lập báo +0.009 và +0.014.
   Mình chưa bao giờ thử.

## 1. So từng khâu: các đội trung thực có công khai

Nguồn: `theainerd/filament_segmentation` (README, `src/filament/*.py`); Dariush (notebook
`dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39` và dataset `solar-filament-unet-weights`: README +
`solar_unet_receipt.json`); umerkang66 `REPORT.md`; tushhaaarrrr notebook `solar-filament-segmentation-2026-0-40`
(filseg v4); zhang0894 (Anon Tokyo) `report/main.tex`; BSergey2000 README.

| Khâu | **Mình (final 1)** | Dariush (0.4502 / 0.39) | theainerd (0.4519 / 0.40) | umerkang (0.4477 / 0.39) | tushhaaarrrr (0.4597 / 0.40) | zhang (0.4427 / 0.38) |
|---|---|---|---|---|---|---|
| Input | 1 kênh, /255 | 1 kênh, **z-score theo đĩa** từng ảnh (r 950) | 3 kênh (bỏ limb darkening, I/median, bán kính) | 1 kênh, mean 125 std 18 | flat field (CLV + nền) | xám ×3 |
| Độ phân giải train | resize 1024 (N1); crop 1024 @2048 (N2) | **crop 512 ở 2048 gốc**, 70% quanh filament | crop 512 ở 2048 gốc, 75% quanh filament | 1024 / 1536 / 2048 | crop 512 / 384 | 1024 rồi fine-tune 1536 |
| Infer | 1024 TTA8 + 2048 | **cả ảnh 2048**, TTA8 | tile 512, chồng 128, Hann, 4 rot | TTA8, phóng lên 2048 | TTA8, @1024 rồi phóng 2048 | 3 view |
| Encoder | EffNet-B4 (N1, N2) | R34 ×5 + **EffNetV2-S in21k** ×5 | EffNet-B3 + scSE (×2 họ) | R34, **EffV2-S**, RegNetY, DINOv3 ConvNeXt-S | R34, ConvNeXt-T, UNet++ R34 + RF-DETR | ConvNeXt-T, ConvNeXt-V2-B |
| Loss / target | BCE + soft Dice, target mềm | BCE + soft Dice, target mềm | BCE + Tversky (+ BiCC 0.3) | BCE + Dice / JML1 + blob + aux | BCE + 0.5 Dice + spine head* | fg + biên |
| Lịch train | AdamW 1e-3, OneCycle, bs 4–6, 30 epoch (~920 mẫu/epoch), chọn ckpt theo val | AdamW **3e-4**, warmup + cosine, **bs 16**, 10,500 bước, ckpt cuối | AdamW 4e-4, bs 12, 30 × 500 bước | bs 8, EMA, 60–80 epoch | 5,000 bước, EMA | 30 + 24 epoch |
| Hậu xử lý | hysteresis t 0.4 / seed 0.99 / area 200 + gộp theo hướng + GBM lọc FP | cc p > 0.35, peak > 0.99, area ≥ 200, **mean p ≥ 0.85** | ứng viên rộng (0.5/0.3, nối 8 px, ≥ 30 px) → GBM hồi quy u, τ 0.25 → **lõi p > 0.5** | seed + gộp 16 px + LightGBM u + chọn extent | **cây 15 ngưỡng + hurdle û = ĥ·m̂ + DP chính xác**, τ 0.26 → **refiner stage 2** | ngưỡng 0.5 + nhóm 8 px → **refiner native + quality head** |
| OOF | fold theo ảnh, seed 42; post tune in-sample, lọc FP cross-fit | nested (luật chọn trên 4 fold) | fold theo **ngày**, rescorer fit lại không có fold | nested mọi tầng | cross-fit + τ nested | OOF công thức 1 fold |

\* spine lấy từ polyline MAGFiLO; luật nội bộ của mình chỉ dùng mask, và F54 đã cho thấy spine ≈ skeleton.

## 2. OOF có so được không? (đo + đọc code)

| Kiểm tra | Kết quả |
|---|---|
| Định nghĩa metric | theainerd `metrics.py`, Dariush receipt (`gt_filaments 8199`, `annotator_sets 1154`), umerkang § 9, BSergey `pq.py` đều gộp TP/FP/FN qua mọi annotator-image, giống `src/filseg/metric.py`. Tổng GT của mình trong `tables.pkl` = **8,199**, khớp |
| Fold | Mình, Dariush, umerkang, BSergey: theo ảnh. theainerd: theo **ngày chụp** (chặt hơn; train có 49 ngày có 2 ảnh, nên chênh lệch không đáng kể). tush: theo quan sát, phân tầng theo trạm × số annotator |
| Lạc quan do tune in-sample | Dariush 0.4523 in-sample so với 0.4502 nested (−0.002). Của mình: post 0.4313 in-sample so với 0.4305 cross-tune; final 1 0.4327 = post in-sample + lọc FP cross-fit, tức lạc quan khoảng +0.001. **So được trong khoảng ±0.002** |
| Cạm bẫy trong repo | `docs/logs/research06/degrade_ladder.txt` ghi "drop 0.00 PQ **0.4478**". Đây là bộ lọc FP chấm **in-sample** trên chính OOF nó đã học (FP 1,639 so với 1,919 ở bản cross-fit), **không** phải OOF. Không được đem số này so với đội khác (lạc quan +0.015) |
| zhang 0.4427, biconcavelens 0.461 | không so được: zhang là OOF của công thức 1 fold; biconcavelens là trung bình 5 lần chia calibration 70 ảnh |

**OOF → LB** (mọi số lấy từ receipt/README của chính đội đó):

| Đội / bài | OOF | LB | OOF − LB |
|---|---|---|---|
| Dariush 5 × R34 | 0.4299 | 0.37 | 0.060 |
| Dariush 5 × V2-S | 0.4461 | 0.39 | 0.056 |
| Dariush 10 model | 0.4502 | 0.39 | 0.060 |
| umerkang v1 / v2 / final | 0.4381 / 0.4438 / 0.4477 | 0.38 / 0.38 / 0.39 | 0.058 / 0.064 / 0.058 |
| theainerd | 0.4519 | 0.40 | 0.052 |
| **Mình final 1** | **0.4327** | **0.37** | 0.063 |

Độ dốc khoảng 1, độ lệch 0.05–0.06 ở mọi đội. Kết luận: LB của mình thấp **chỉ vì OOF thấp**, không có lỗi riêng
ở phía test. Để có LB 0.40 cần OOF khoảng ≥ 0.452–0.455.

## 3. Đo trên OOF N1 + N2 của mình (707 ảnh, nested theo `fold_split`)

Cách làm: với mỗi tập ứng viên (rời nhau), tính cho từng ứng viên A = tổng IoU khớp qua các annotator và
n_read. PQ cộng được theo ứng viên: PQ(S) = ΣA / (0.5·N_GT + 0.5·Σ n_read) (giống theainerd `exact_counts`,
umerkang § 8). GBM hồi quy u = A / n_read với 16 đặc trưng kiểu theainerd + 2 đặc trưng ảnh. Model của fold f
fit trên 4 fold còn lại; τ của fold f chọn trên 4 fold còn lại. Bootstrap ghép cặp 1,000 lần theo ảnh, so với
final 1 chưa lọc (0.4313).

| Biến thể | PQ | Δ so với nền [CI 95%] | Δ từng fold |
|---|---|---|---|
| nền: hậu xử lý final 1, không lọc | 0.4313 | – | – |
| nền + GBM u nested (≈ bộ lọc FP hiện tại) | 0.4326 | +0.0013 [+0.0001, +0.0025] | +.001 +.003 +.003 −.000 −.001 |
| **luật Dariush** trên map của mình (0.35 / 0.99 / 200 / 0.85) | 0.4237 | −0.0077 [−0.012, −0.003] | |
| luật Dariush, nới (peak 0.98, mean 0.80) | 0.4310 | −0.0003 | |
| theainerd: ứng viên 0.3/0.5 + nối 8 px, giữ cả extent | 0.4277 | −0.0036 [−0.008, +0.001] | |
| theainerd đầy đủ: như trên + **lõi p > 0.5** | 0.4325 | +0.0012 [−0.003, +0.005] | |
| **cc p > 0.35 (≥ 30 px), không seed, không gộp + GBM u nested** | **0.4348** | **+0.0035 [−0.0008, +0.0078]** | +.001 +.006 +.002 +.003 +.005 |

Oracle (giữ ứng viên khi u thật > PQ/2): trên ứng viên final 1 là 0.480; trên cc p > 0.35 là 0.504
(umerkang báo oracle 0.497–0.524 trên map của họ). Trần chọn ứng viên của mình và của họ **gần như bằng
nhau**. Phần họ hơn mình nằm ở map, không nằm ở số ứng viên.

Đếm (chưa lọc): mình TP 4,910 / FP 2,283 / SQ 0.676. Dariush 5 × V2-S (receipt): TP 5,165 / FP 2,450 / SQ
0.684. Riêng phần SQ chênh đã tương đương khoảng +0.005 PQ; phần TP chênh khoảng +0.010.

## 4. Ba khác biệt có khả năng giải thích khoảng cách, và thí nghiệm rẻ nhất cho từng cái

### (A) Công thức model: EfficientNetV2-S in21k, crop 512 ở độ phân giải gốc, infer cả ảnh 2048, bs 16, lr 3e-4

- **Bằng chứng nội bộ của Dariush** (cùng recipe, cùng luật): R34 0.4299 → V2-S 0.4461, tức encoder đóng góp +0.016.
  Của mình: R34 → B4 +0.013 (E9a). theainerd B3 ở crop 512 native: một họ 0.4228. Suy ra công thức native-512 tự
  nó không phải phép màu (B3 ≈ N1). Phần lớn lợi đến từ **V2-S in21k**, và có thể thêm một phần từ native.
- **Ngược chiều**: BSergey thấy native 2048 với crop 640 **kém 0.07**; zhang thấy native 2048 kém 1536 (0.4262 so với
  0.4318); N2 của mình (crop 1024 @2048, bs 4) đơn lẻ kém N1 −0.005, +643 FP. Các khác biệt chưa tách riêng được
  giữa Dariush và N2: crop 512 so với 1024, **bs 16 so với 4 (BatchNorm)**, lr 3e-4 so với 1e-3 OneCycle,
  z-score theo đĩa so với /255, checkpoint cuối so với chọn theo val.
- **Liên quan tới hàng đợi**: N9 hiện tại là V2-S ở **resize 1024** (thẻ N9, PLAN § 5b). Nó kiểm được giả thuyết
  "encoder". Nó không kiểm được giả thuyết "native 512 + cả ảnh 2048", nhưng đó là cách duy nhất có số 0.4461.
- **Thí nghiệm rẻ nhất**: nếu N9 chưa chạy, thì hoặc thay N10 (B4 @1536), hoặc chèn **một fold pilot** với
  `--encoder tu-tf_efficientnetv2_s.in21k_ft_in1k --res 2048 --crop 512 --bs 16 --lr 3e-4`, khoảng 10k bước, fg 0.7,
  infer cả ảnh 2048 (cần `predict` nhận 2048 không tile; V2-S 2048² bs 1 vừa L4). Ước 1–1.3 h L4 cho fold 0.
  Gate fold 0: `hyst_soft` ≥ N1 f0 (0.4158) + 0.010. Nếu qua, chạy 4 fold còn lại (~5 h). Thêm z-score theo đĩa
  vào dataloader (khoảng 5 dòng ở `dl.py` dòng 186 và 311) nếu đã đổi recipe thì đổi luôn. Không ablation riêng.
- **ΔOOF kỳ vọng**: đơn lẻ +0.005 … +0.025 so với N1. Khi vào ensemble N1 + N2: +0.006 … +0.015.
  Đây là đòn bẩy lớn nhất còn lại và là đòn bẩy duy nhất có bằng chứng ngoài ≥ 0.015.

### (B) Tầng quyết định trên ứng viên rộng (hoặc cây ngưỡng) thay cho "hysteresis rồi mới lọc"

- Cơ chế: hysteresis seed 0.99 + gộp đã loại sẵn những ứng viên mà GBM có thể giữ lại một cách có lợi. Khi cho
  GBM thấy mọi thành phần cc p > 0.35 thì nó tự học "peak, mean p, kích thước", tức đúng những thứ luật
  Dariush làm bằng tay, nhưng theo từng ứng viên. tush mở rộng ý này thành cây 15 ngưỡng (antichain) + DP chính
  xác với τ = PQ*/2 (Dinkelbach): một filament sáng lấy ở ngưỡng thấp, filament mờ bên cạnh tách ra ở ngưỡng cao.
- **Đã đo**: +0.0035 [−0.0008, +0.0078], dương 5/5 fold (bảng § 3). CI còn chạm 0.
- **Mới so với EXPERIMENTS?** Một phần. H6-tree (30/10) đã thử cây + GBM + DP và ra −0.004. Nhưng lần đó chạy
  trên **213 ảnh holdout của E9a**, nên GBM chỉ được train trên khoảng 170 ảnh. Bộ lọc FP hiện tại thì chạy trên
  ứng viên đã qua hysteresis. Ứng viên cc rộng + GBM u trên 707 ảnh OOF thì chưa ai thử.
- **Thí nghiệm rẻ nhất (Mac, khoảng 1 h dưới tải hiện tại)**: thêm 2–3 mức ngưỡng (0.35 / 0.5 / 0.7) thành cây,
  chọn bằng DP với û nested. Dùng đúng `extract.py` (thêm mức) và `analyze.py`. Gate F48: CI > 0 trên 707 ảnh.
  Nếu không lên được cây thì dùng biến thể cc p > 0.35 + GBM u. Biến thể này không tốn GPU nên có thể vào final
  nếu CI > 0 khi chạy lại với seed fold khác.
- **ΔOOF kỳ vọng**: +0.003 … +0.006. Đòn bẩy này không đủ để đóng khoảng cách, nhưng cộng được với (A).

### (C) Refiner theo từng instance trên crop native (stage 2)

- **Bằng chứng ngoài** (hai đội độc lập, cả hai có CI):
  - zhang: coarse 0.4287 → refined 0.4427, **+0.0139**, dương 5/5 fold, CI ghép cặp [+0.011, +0.017]. Lợi chủ yếu
    từ precision: FP −607, TP −61, qua quality head. Phần này mình đã có một phần nhờ bộ lọc FP, nên đừng kỳ vọng đủ +0.014.
  - tush: **+0.0094** sau một tầng quyết định đã rất mạnh (≈ 0.450). Nghĩa là phần lợi về extent/tách instance
    đứng riêng được. Họ dùng crop xoay 384×128 theo trục dài, 5 kênh (ảnh, p, spine, seed, các instance khác),
    target = đồng thuận của các annotator *có vẽ* filament đó, ngưỡng t* = J/(1+J) ≈ 0.40.
- **Mới hoàn toàn** so với EXPERIMENTS/RESEARCH (không có mục "refine/stage 2" nào). Cơ chế này nhắm thẳng vào
  F52: 1,047 dự đoán IoU 0.3–0.5 (trần +0.055) và gộp/tách nhầm, những chỗ hậu xử lý theo ngưỡng toàn cục không
  sửa được (tush: 16% GT chỉ có ứng viên IoU ≤ 0.5, 59% trong số đó là ứng viên quá nhỏ).
- **Chi phí**: code mới (cắt crop theo minAreaRect, dataset, U-Net R18/UNet++ nhỏ 4–5 kênh, dán lại, giải quyết
  chồng lấn) khoảng 1 ngày. Train 5 fold trên Colab L4 khoảng 1.5–2 h (crop nhỏ, khoảng 5k instance), hoặc Mac MPS
  khoảng 5 h. Crop train phải lấy từ OOF map của chính fold ngoài (zhang cảnh báo nếu lấy từ fold train thì quá dễ).
- **Thí nghiệm rẻ nhất**: chỉ fold 0, R18-UNet 4 kênh (ảnh, p, seed, các instance khác), crop 256×256 native quanh
  bbox mở rộng 48 px, 15 epoch, train trên instance OOF của fold 1–4, đánh giá trên fold 0. Gate: Δ ≥ +0.005 trên
  fold 0 rồi mới train 4 fold. Nếu thời gian bắt buộc phải chọn, (A) nên đi trước (C).
- **ΔOOF kỳ vọng**: +0.004 … +0.010 (thấp hơn số của zhang vì phần FP đã có bộ lọc lo).

## 5. Những khác biệt khác đã kiểm, không giải thích được khoảng cách

| Khác biệt | Vì sao loại |
|---|---|
| Lõi p > 0.5 của theainerd (ứng viên rộng nhưng mask hẹp) | đo: +0.0012 [−0.003, +0.005], ≈ final 1 |
| Luật "mean p ≥ 0.85" của Dariush | trên map của mình kém −0.008; bộ lọc FP đã có `mean_prob` |
| 3 kênh vật lý (theainerd), flat field (tush) | tush E1: CNN tự học được; RESEARCH § 11 (F39–F41) |
| BiCC loss (theainerd, +0.01 cho một họ, 4/5 fold) | gần với v2 của mình (âm, F49); để sau (A) |
| Định nghĩa OOF / fold | § 2: so được trong ±0.002 |
| Phía test (LB) | § 2: độ lệch OOF−LB của mình (0.063) nằm trong khoảng của các đội khác (0.052–0.064) |

## 6. Rủi ro luật

- (A), (B), (C) chỉ dùng mask train và ảnh H-alpha test, nên không vi phạm. Không dùng spine (khác tush), không
  dùng MAGFiLO 1.0, không dùng frame lân cận. Weights V2-S in21k lấy từ timm (Apache-2.0).
- (C) xuất mask rời nhau nhờ bước giải quyết chồng lấn theo thứ tự tin cậy (zhang: "confidence-ordered pixel
  allocation"). Phải chạy lại `validate` không chồng lấn trước khi nộp.
