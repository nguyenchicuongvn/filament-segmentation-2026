# Research 07, lens data_leverage: dữ liệu ngoài, nhãn yếu, cách dùng nhãn đa annotator

Ngày 06/11. Agent: data_leverage. Script và output (chỉ đọc repo, không sửa code):
`/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/data_leverage/`
(`temporal.py`, `fusion_loo.py`, `impute_proxy.py`, `impute_proxy.txt`). Map dùng: `outputs/ens/n1n2_oof`
(OOF 707 ảnh) và `outputs/post_research_oof_orient.npz` (`out_orient_ab`, PQ 0.4305 = final 1 trước lọc FP).

## TL;DR

1. **Dữ liệu ngoài có nhãn: không có nguồn nào vừa hợp luật vừa có ích.** Mọi tập filament GONG có nhãn
   người trong 2011–2022 đều là MAGFiLO (cấm) hoặc phái sinh từ nó. Các tập khác (BBSO, ChroTel, KSO,
   HEK/AAFDCC) theo **quy ước vẽ khác**, trong khi PQ ở đây là bài toán khớp quy ước của annotator MAGFiLO.
   Số đo mới (D1) cho thấy ngay cả ảnh train **cùng bầu trời** (cách < 24 h, cùng quy ước) cũng không giúp.
2. **Ảnh GONG không nhãn (hợp luật cho train theo S17) + pseudo-label: ≈ 0.** Bằng chứng ngoài đều âm
   (S9d, S9e), D1 nói thêm dữ liệu "gần" không phải nút thắt.
3. **Ý tưởng duy nhất có cơ chế đo được: "annotator ảo" từ teacher (D3).** Map OOF của ensemble dự đoán
   một annotator bị giữ lại **tốt hơn cả trung bình của 2 annotator thật** (MSE 0.0751 so với 0.0856).
   58% mẫu train (411 ảnh) chỉ có 1 annotator, tức target cứng 0/1 rất nhiễu (MSE 0.1165). Trộn target
   `(A + 2T)/3` giảm nhiễu target 38% trên các ảnh đó. Kỳ vọng trung thực **+0.003 đến +0.010 OOF**
   cho một model (và là thành viên ensemble đa dạng), cần **một** run Colab. Không phải đột phá +0.05.
4. **Gộp nhãn kiểu instance (D2) bị bác bỏ bằng số đo**: gộp 2 người tốt nhất chỉ đạt PQ 0.347 khi chấm
   với người thứ 3; model đạt 0.424 trên cùng 151 ảnh. Target không phải chỗ thiếu.

Kết luận thẳng: góc "dữ liệu" **không** chứa bước nhảy kiểu Kaggle thường thấy ở bài này, vì (a) dữ liệu
cùng quy ước duy nhất (MAGFiLO đầy đủ, 1,593 quan sát, so với 887 của cuộc thi) bị cấm vì chứa test, và
(b) model đã vượt mọi tổ hợp nhãn người (0.424 so với 0.347). Phần đáng làm là D3.

## 1. Rà soát nguồn dữ liệu và luật

### 1.1 Luật (từ RESEARCH § 6.1, không tìm thấy phát biểu mới của host)

- S19 (host 29/10): lúc infer **chỉ dùng ảnh test**, không metadata, không frame lân cận, không MAGFiLO công khai.
- S17 (host ~11/10): **được** dùng frame GONG lân cận để train (chuyển nhãn hoặc làm ảnh không nhãn), miễn
  bỏ các frame trong ±1 h quanh thời điểm ảnh test.
- S20: được dùng spine/bbox/area của JSON train (repo tự chặt hơn: chỉ mask).
- Không tìm thấy phát biểu nào về tập filament của **trạm khác** (BBSO, ChroTel...). Trang thảo luận Kaggle
  render phía client nên WebFetch không đọc được; tìm kiếm web không ra thread nào. Host chính là nhóm tác
  giả MAGFiLO (Ahmadzadeh, Kempton; cuộc thi cũng là IEEE BigData Cup 2026 challenge 02), và họ rà tay
  top bài. Dùng dữ liệu ngoài mà không hỏi trước là rủi ro hồ sơ.

### 1.2 Ngày test và nguy cơ trùng

180 ảnh test trải đều 2011–2022 (24/23/23/22/17/22/19/4/2/3/14/7 ảnh mỗi năm), giống phân bố train.
Khoảng cách từ mỗi ảnh test đến ảnh train gần nhất (chính dữ liệu cuộc thi): < 1 h: **3 ảnh** (2 cặp cách
**20 giây**, khác trạm: `20130208063134Lh`↔`20130208063154Uh`, `20150913082034Lh`↔`20150913082014Th`,
`20201228202510Mh`↔`20201228202430Ch`); 1–6 h: 18; 6–24 h: 42; 24–72 h: 72; > 72 h: 45. Đây là dữ liệu do
host phát, dùng train là hợp lệ; tra cứu nhãn train theo ngày lúc infer thì **vi phạm** S19 (dùng metadata),
nên không đề xuất.

Hệ quả cho dữ liệu ngoài: **bất kỳ** tập nhãn GONG nào phủ 2011–2022 đều chứa đúng các ngày test (và rất
có thể đúng các frame test). MAGFiLO 1.0 có 1,593 quan sát; cuộc thi dùng 887. ~700 quan sát MAGFiLO còn
lại là "dữ liệu thêm cùng quy ước" duy nhất tồn tại, và đã bị cấm.

### 1.3 Từng nguồn

| Nguồn | Nội dung | Trùng ngày test | Quy ước nhãn | Kết luận |
|---|---|---|---|---|
| MAGFiLO 1.0 (Sci. Data 2024) | 10,244 filament, 1,593 quan sát GONG 2011–2022 | chứa 139–180/180 ảnh test (S1) | đúng quy ước | **cấm** |
| "MAGFiLO 2.0" | tìm không thấy bản phát hành nào (chỉ có v1.0 trên Dataverse/mlecofi) | – | – | không tồn tại công khai |
| Diercke et al. 2024 (arXiv 2402.15407) | YOLOv5 train trên nhãn ChroTel → bbox trên cả kho GONG → U-Net bán giám sát | phủ cả kho GONG, tức trùng ngày test | mask **do model sinh** từ bbox, quy ước ChroTel | nhãn yếu, khác quy ước; không thấy dữ liệu phát hành; nếu dùng frame GONG ngày test thì phải bỏ ±1 h |
| BBSO (Zhu 2019, Guo 2022, Flat U-Net 2025), NADC China-VO "Solar Filament Segmentation Dataset" | ảnh BBSO 2048², nhãn bán tự động (ngưỡng + sửa tay) | trùng năm, khác trạm | ngưỡng cường độ, không "một mảnh không lỗ", không bỏ ±70° | khác miền và khác quy ước |
| HEK / AAFDCC (Bernasconi) | polygon từ detector tự động | trùng ngày | không phải nhãn người | nhãn yếu tệ hơn model của mình |
| Kanzelhöhe (KSO) | ảnh + detector tự động | trùng ngày | – | như trên |

Vì sao khác quy ước là chết ở đây: F56 cho thấy annotator **không** vẽ theo mép tối của ảnh (mọi chỉnh biên
theo cường độ đều âm), F41 cho thấy filament mờ và bất đồng là cùng một nhóm, F53 cho thấy 1 − PQ phần
lớn là bất đồng quy ước. Nhãn BBSO/ChroTel dạy một quy ước thứ ba, nên chỉ có thể kéo model xa khỏi
"kiểu trung bình MAGFiLO". Trường hợp Kaggle mà dữ liệu ngoài cho bước nhảy lớn (Sartorius 2021 với
LIVECell) là vì dữ liệu ngoài do **chính host** gán theo **cùng giao thức**; ở đây tương đương là MAGFiLO,
và nó bị cấm.

## 2. Số đo mới

### D1. Ảnh train "cùng bầu trời" không giúp (đo trên OOF final 1)

Với mỗi ảnh OOF, tính khoảng cách thời gian đến ảnh train gần nhất **mà model của fold đó đã thấy**.
Filament sống nhiều ngày, nên ảnh cách < 24 h gần như cùng bầu trời và cùng quy ước gán.

| Khoảng cách tới ảnh train đã thấy | n ảnh | PQ OOF | CI 95% (bootstrap ảnh) |
|---|---|---|---|
| < 1 h | 16 | 0.424 | [0.374, 0.490] |
| 1–6 h | 36 | 0.405 | [0.365, 0.447] |
| 6–24 h | 134 | 0.420 | [0.398, 0.439] |
| 24–72 h | 299 | 0.439 | [0.425, 0.451] |
| > 72 h | 222 | 0.430 | [0.414, 0.448] |
| **< 24 h / ≥ 24 h** | 186 / 521 | **0.418 / 0.435** | |

Có ảnh có nhãn của gần như cùng các filament **không** làm PQ tăng (còn hơi thấp hơn, trong nhiễu). Vậy
model không bị giới hạn bởi việc "đã thấy filament này chưa", mà bởi quy ước biên và bất đồng. Đây là
bằng chứng trực tiếp chống lại: chuyển nhãn sang frame GONG lân cận (khớp S9d của biconcavelens: một
model 0.404 → 0.372), thêm ảnh GONG không nhãn + pseudo-label, và mọi tập ngoài phủ cùng ngày.
Ghi chú phụ: 18 ảnh train có ảnh train khác cách < 1 h (khác trạm) nằm ở fold khác, OOF có thể lạc quan
một chút ở các ảnh đó, nhưng bảng trên cho thấy hiệu ứng không đáng kể.

### D2. Gộp nhãn kiểu instance không hơn được model (151 ảnh 3 annotator, leave-one-out)

Gộp nhãn của 2 annotator theo nhiều luật, chấm PQ với annotator thứ 3 (xoay vòng cả 3), gộp TP/FP/FN.

| Luật gộp 2 người | PQ vs người thứ 3 | TP | FP | FN |
|---|---|---|---|---|
| một người (A / B, trung bình) | 0.318 / 0.349 (≈ 0.334) | 1,552 / 1,622 | 1,558 / 1,282 | 1,454 / 1,386 |
| hợp pixel rồi CC (≈ ngưỡng thấp trên target mềm) | 0.332 | 1,694 | 1,772 | 1,312 |
| giao pixel rồi CC (≈ ngưỡng cao) | 0.314 | 1,417 | 1,334 | 1,589 |
| ghép cặp IoU > 0.2, giữ **hợp** của cặp, bỏ filament chỉ một người vẽ | **0.347** | 1,446 | 868 | 1,560 |
| như trên, giữ giao / giữ shape nhỏ / shape lớn | 0.328 / 0.341 / 0.340 | | | |
| **model final 1 trên cùng 151 ảnh (OOF)** | **0.424** | 923 / 3 ann. | | |

Nhận xét: (i) luật gộp tốt nhất của 2 người chỉ hơn một người +0.013; (ii) model hơn mọi tổ hợp nhãn người
**+0.077**. Ý tưởng "đổi target thành nhãn đồng thuận theo instance (STAPLE, medoid, giao/hợp cặp)"
không có chỗ: target trung bình hiện tại đã cho model tốt hơn bất cứ nhãn gộp nào. Đây cũng là bằng
chứng mạnh nhất cho report rằng bài này đã chạm "trần nhãn" (bổ sung F6/F53).

### D3. Teacher làm "annotator ảo" tốt hơn một annotator thật (proxy cho target)

Cùng 151 ảnh, vùng ROI = hợp mask 3 người ∪ T > 0.2, giãn 15 px. Annotator C giữ lại làm tham chiếu;
`A`, `B` là 2 người còn lại (mask hợp, 0/1); `T` = map OOF n1n2 (sạch với ảnh này). MSE pixel so với C
(sai số so với một nhãn nhiễu = sai số so với xác suất thật + phương sai nhãn chung, nên xếp hạng đúng):

| Target | MSE vs C |
|---|---|
| A (giống target hiện tại của ảnh 1 annotator) | 0.1165 |
| (A+B)/2 (target hiện tại của ảnh 2 annotator) | 0.0856 |
| T (chỉ teacher) | 0.0751 |
| (A+T)/2 | 0.0771 |
| **(A+2T)/3** | **0.0723** |
| (A+B+T)/3 | 0.0717 |
| **(A+B+2T)/4** | **0.0687** |

Hiệu bootstrap theo ảnh (1,000 lần): A − (A+2T)/3 = +0.0443 [+0.0410, +0.0474];
(A+B)/2 − (A+2T)/3 = +0.0133 [+0.0117, +0.0149]; (A+B)/2 − (A+B+2T)/4 = +0.0169 [+0.0157, +0.0181].

Ý nghĩa: một ảnh 1 annotator cộng teacher (trọng số 2) cho target **gần nhãn người mới hơn** ảnh có 2
annotator thật. Với 411/707 ảnh chỉ có target cứng, đây là cách "thêm 2 annotator miễn phí" cho 58% mẫu.

## 3. Ý tưởng

### Ý tưởng 1 (đề xuất duy nhất đáng tiêu GPU): target "annotator ảo" = trộn nhãn người với teacher OOF

- **Cơ chế**: PQ gộp theo bản gán nhãn thưởng cho dự đoán khớp **kiểu trung bình** của annotator (F26).
  Target trung bình nhiều người dạy kiểu đó; ảnh 1 người dạy cả "phong cách riêng" của người đó với độ
  tự tin 1.0 (target cứng). D3 đo được rằng teacher là xấp xỉ của trung bình người tốt hơn một người thật,
  nên `soft' = (n·mean_ann + k·T)/(n + k)` với k ≈ 2 làm giảm nhiễu target (giảm 38% MSE trên ảnh 1 người,
  20% trên ảnh 2 người). Đây là knowledge distillation trên **train**, không phải pseudo-label test.
  Khác Noisy Student/pseudo-label ở chỗ không thêm ảnh, chỉ làm sạch target, và nhãn người vẫn có mặt.
- **Mới so với EXPERIMENTS/RESEARCH**: gần nhất là target mềm đa annotator (N1, đang dùng), BCE hiệu chỉnh
  (N7, F51: hiệu chỉnh map không đổi PQ) và pseudo-label test (S9e, bị loại). Chưa ai trong repo hay trong
  các đội công khai (§ 6.2) dùng map OOF làm target cho ảnh train. F51 là cảnh báo: hiệu chỉnh tự nó không
  đổi PQ; khác biệt ở đây là thay đổi **thông tin** trong target (bớt nhiễu phong cách một người), không
  chỉ thang xác suất.
- **Bằng chứng đo**: D3 (proxy target). Chưa có số PQ. Trần không đo được nếu không train.
- **Bằng chứng ngoài**: Lemay 2022 (MELBA): trung bình rater so với lấy mẫu một rater 54.3 so với 50.6 Dice
  (+3.7 điểm); Wang 2023 QUBIQ: trung bình 77.31 so với random 76.30, trung bình có trọng số 79.85
  (train_recipe.md § 2.2). Tức giảm nhiễu target giúp được vài điểm Dice khi nhãn là nhiều người; ở đây
  chỉ phần ảnh 1 người được lợi lớn. Tương phản: biconcavelens pseudo-label **test** −0.024 (S9e), nhưng đó
  là thay nhãn bằng dự đoán trên ảnh không nhãn, không phải trộn trên ảnh có nhãn.
- **Rủi ro rò rỉ OOF (quan trọng)**: map OOF của ảnh x (fold i) do model train trên mọi fold ≠ i, tức có
  nhãn fold j. Student fold j học từ target đó thì gián tiếp thấy nhãn fold j → OOF lạc quan. Giao thức
  sạch: student fold j dùng teacher = **chính model fold j** (`outputs/colab/*_f{j}/*.pt`, đã có sẵn) dự đoán
  in-sample trên 4/5 ảnh train của nó. Map in-sample sát nhãn hơn nên khử nhiễu yếu hơn, tức ước lượng bảo
  thủ. Bản final (train full) dùng map OOF 5 fold làm teacher.
- **Chi phí**: 1 run Colab 5 fold (~6.5 h L4) + ~20–30 phút infer in-sample (5 × ~566 ảnh, có thể bỏ TTA).
  Code: ~20 dòng trong `soft_cached` (đọc map teacher, trộn, cache với tên khác).
- **Kỳ vọng ΔOOF**: +0.003 đến +0.010 cho model đơn so với N1 cùng công thức; làm thành viên ensemble thì
  ~+0.002–0.005 trên final 1 (thành viên mới thường được ~nửa so với đơn, xem G4/F51). Xác suất âm không
  nhỏ (~30%): F51 cho thấy map tốt hơn chưa chắc PQ tốt hơn.
- **Rủi ro luật**: thấp. Chỉ dùng mask train và model của chính mình; không đụng test. Nên ghi rõ trong
  report: "self-distillation on training images, teacher = own out-of-fold ensemble".
- **Bước đầu rẻ nhất, chạy hôm nay**: (1) chạy lại D3 với nhiều k (0.5–4) và theo kích thước filament
  để chọn k (CPU, 15 phút); (2) kiểm tra map in-sample của model fold 0 trên 50 ảnh train (MPS, ~10 phút):
  nếu MSE của `(A+2T_in)/3` vs annotator giữ lại vẫn < `(A+B)/2` thì giao thức sạch còn giữ được phần lớn lợi
  ích; (3) chỉ khi (2) dương thì thay N8 (seed 2) trên Colab bằng run này, giữ luật gate đã đăng ký.

### Ý tưởng 2 (bác bỏ, có số): thêm dữ liệu cùng bầu trời: frame GONG lân cận, chuyển nhãn, pseudo-label ảnh không nhãn

- **Cơ chế giả định**: thêm ảnh của cùng filament ở thời điểm khác làm model "nhớ" filament và robust với seeing.
- **Số đo**: D1: ảnh OOF có ảnh train đã thấy trong vòng 24 h đạt 0.418, ảnh không có đạt 0.435. Hiệu ứng
  "đã thấy filament này" ≈ 0 (âm, trong nhiễu).
- **Bằng chứng ngoài**: S9d (chuyển nhãn sang frame lân cận: −0.032 với một model, 0 trong hệ thống đầy đủ),
  S9e (pseudo-label −0.024, SimSiam 0), HuBMAP 2021 "rất hạn chế".
- **Kỳ vọng**: 0 đến −0.01. Không làm. Mới so với repo: chỉ số đo D1 là mới.

### Ý tưởng 3 (bác bỏ, có số): target đồng thuận theo instance (STAPLE / medoid / giao-hợp cặp)

- **Cơ chế giả định**: target trung bình pixel làm biên "giữa hai người"; một shape đồng thuận theo instance
  có thể khớp IoU > 0.5 với người mới tốt hơn.
- **Số đo**: D2: luật tốt nhất 0.347 (+0.013 so với một người), model 0.424 trên cùng ảnh. Không có luật gộp
  nhãn nào làm target tốt hơn cái model đã học được.
- **Kỳ vọng**: ≈ 0. Lemay 2022 cũng cho STAPLE ≈ trung bình. Không làm.

### Ý tưởng 4 (bác bỏ): tập filament ngoài (BBSO, ChroTel, KSO, HEK) hoặc "MAGFiLO 2.0"

- Không có MAGFiLO 2.0 công khai. Các tập khác: khác quy ước (§ 1.3), trùng ngày test, rủi ro hồ sơ khi host
  là tác giả MAGFiLO và rà tay. Pretrain trên chúng rồi fine-tune: SimSiam trên ảnh cùng miền đã = 0 (S9e),
  encoder ImageNet đã đủ (F39: model tự học các bộ lọc).
- **Kỳ vọng**: −0.01 đến +0.003, chi phí 1 run Colab và rủi ro luật. Không làm.

## 4. Ước lượng tác động nếu "dữ liệu" là đòn bẩy, để đặt kỳ vọng

- Đường cong dữ liệu thô: E9a (70% ảnh, holdout 213) 0.410 so với N1 (80%, OOF 707) 0.421 cùng hậu xử lý
  hyst; khác tập đánh giá nên chỉ là chỉ báo (+0.01 cho +14% ảnh, trong nhiễu ±0.01). Kể cả có thêm ~700
  quan sát MAGFiLO (gấp đôi dữ liệu, cùng quy ước) thì theo luật log-tuyến tính cũng chỉ ~+0.02–0.03; đó là
  trần của mọi hướng "thêm dữ liệu", và nó bị cấm.
- Khoảng cách tới các đội 0.40 LB (~+0.02 OOF) không đến từ dữ liệu: họ dùng cùng 707 ảnh.

## Nguồn

- MAGFiLO v1.0: Sci. Data 11:1031 (2024), DOI 10.1038/s41597-024-03876-y; NSF PAR 10538884 (1,593 quan sát, 2011–2022).
- IEEE BigData Cup 2026, challenge "Pixel-Precise Segmentation of Solar Filaments" (bigdataieee.org/BigData2026/cup/), dùng MAGFiLO, host Kaggle.
- Diercke et al. 2024, arXiv 2402.15407 (ChroTel → GONG, YOLOv5 + U-Net bán giám sát).
- NADC China-VO r101545 "Solar Filament Segmentation Dataset" (403 khi tải; chỉ thấy tiêu đề).
- Lemay et al. 2022 arXiv 2202.07550; Wang et al. 2023 arXiv 2303.16296 (trích từ research06/train_recipe.md).
- RESEARCH.md S1, S9d, S9e, S17, S19, S20; F26, F41, F51, F53, F56.
