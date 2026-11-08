# Research07 · lens metric_decoding: giải mã tối ưu theo kỳ vọng PQ (06/11)

**Kết luận một dòng**: trên map OOF hiện tại (N1 + N2, giải mã final 1), **giải mã đã chạm sàn**.
Phần chênh oracle +0.05 đến +0.13 gần như toàn bộ là "khẩu vị riêng" của annotator được chấm, và
không đoán được kể cả khi biết nhãn thật của annotator khác. Không nên tốn thêm giờ Mac hay Colab cho
tầng giữ/bỏ, extent hay tách/gộp. Đòn bẩy còn lại chỉ nằm ở chất lượng map.

Script, output (chỉ đọc repo, không sửa):
`/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/metric_decoding/`
(`cand.py` → `cand.pkl`, `oracles.py` → `oracles.txt`, `annot.py` → `annot.txt`,
`learned.py` → `learned.txt`, `consensus.py` → `consensus.txt`).

Pipeline đo: `outputs/ens/n1n2_oof` (707 ảnh, 2048 px), `configs/post_ens_n1n2_hyst_orient.json`,
**chưa** áp lọc FP → base PQ **0.4312** (final 1 sau lọc FP là 0.4327). 4,397 instance, 8,199 GT.

## 1. Khung lý thuyết (nhắc lại F26, viết ở dạng dùng được)

Với dự đoán không chồng nhau, mỗi bản gán nhãn có TP + ½FP + ½FN = ½(n_pred + n_gt). Cộng dồn:

PQ = Σ_j Σ_k s_jk / ( ½ (Σ_j K_j + G) ),  với s_jk = IoU(j, GT của annotator k) nếu > 0.5, ngược lại 0.

Mẫu số **không phụ thuộc việc khớp**, chỉ phụ thuộc số instance phát ra (nhân K của ảnh). Hệ quả:
mỗi instance đóng góp độc lập, giá trị biên là s̄_j − PQ/2 (s̄_j = trung bình qua annotator của ảnh).
Do đó: giữ khi s̄_j > PQ/2; chọn extent tối đa s̄_j; tách khi s̄_a + s̄_b − s̄_ab > PQ/2. Lời giải
Bayes cần đúng một thứ: **E_annotator[s_j(e)]** cho từng instance j và extent e. Phần đo dưới đây hỏi
thông tin này có tồn tại ở đâu hay không.

14 extent ứng viên cho mỗi instance: base; cắt lại bên trong ở p > 0.5…0.9; co 1–2 px; dãn 1–3 px
vào nền; lan trắc địa tới 6 px vào p > 0.3/0.2/0.1 (không lấn instance khác).

## 2. Số đo

### 2.1 Oracle "tự chấm" (dùng chính annotator được chấm): trần lý thuyết, không đạt được

| Oracle | PQ | Δ |
|---|---|---|
| base | 0.4312 | – |
| extent toàn cục tốt nhất (in0.5) | 0.4313 | +0.0001 (mọi extent khác âm: dil1 −0.029, ero1 −0.014, in0.7 −0.008) |
| giữ/bỏ hoàn hảo | 0.4798 | +0.049 (F27 cũ: +0.058) |
| extent hoàn hảo, giữ tất cả | 0.4714 | +0.040 |
| extent + giữ/bỏ hoàn hảo | 0.5156 | +0.084 |
| như trên, nhưng oracle theo từng bản gán nhãn (K ≥ 2) | 0.5605 | +0.129 |

Nhìn qua thì có vẻ "decoding có +0.08". Câu hỏi đúng là phần nào trong đó **đoán được**.

### 2.2 Oracle chéo annotator (mới): quyết định bằng nhãn **thật** của annotator khác, chấm trên annotator còn lại

Ảnh K ≥ 2. Với mỗi bản gán nhãn k: chọn extent và giữ/bỏ theo trung bình s của các annotator
o ≠ k, rồi chấm trên k. Đây là một bộ dự đoán E_annotator[s] **tốt hơn mọi model**, vì nó có nhãn thật
của 1–2 người cùng ảnh.

| Quyết định từ người khác | K ∈ {2,3} | K = 2 (1 người khác) | K = 3 (2 người khác) |
|---|---|---|---|
| giữ/bỏ | **+0.0000** | −0.0153 | +0.0098 |
| chọn extent | **−0.0074** | −0.0059 | −0.0084 |
| extent + giữ/bỏ | −0.0039 | −0.0152 | +0.0034 |
| extent có lề δ (đổi chỉ khi hơn base ≥ δ), δ 0.05 / 0.1 / 0.2 / 0.3 | −0.0060 / −0.0045 / −0.0032 / −0.0012 | | |

Độ tin cậy giữa hai annotator trên cùng instance dự đoán: corr(s_A, s_B) = **0.43**;
P(B khớp | A khớp) 0.81, P(B khớp | A không khớp) **0.42**. Vì 0.42 × SQ ≈ 0.27 > PQ/2 ≈ 0.22, kể cả
instance bị một người "bác" vẫn đáng giữ. Đó là lý do bỏ theo người khác không có lợi.

**Diễn giải**: extent tốt nhất cho annotator A **không** chuyển sang annotator B (chọn theo A làm B
tệ đi, có lề cũng chỉ tiến về 0 từ phía âm). Toàn bộ +0.040 của oracle extent là nhiễu riêng của từng
người. Ở giữ/bỏ, trần đạt được với hai người khác chỉ khoảng +0.01. Lọc FP hiện tại đã lấy khoảng
+0.0015–0.003 trong phần đó.

### 2.3 Bộ giải mã học được (GBM, 5 fold theo ảnh, target = s̄ theo từng extent)

R² OOF của s̄ ở extent base **0.19** (F27 cũ 0.26, trần thực tế theo §2.2 vào khoảng 0.43² ≈ 0.19 cho
một annotator). Chọn λ in-sample, nên số dưới đây còn lạc quan:
giữ/bỏ **+0.0004**, extent **−0.0050**, extent + giữ/bỏ **−0.0036**. Kết quả trùng với F27
(−0.004) trên một map khác và cách làm khác.

### 2.4 Giải mã theo annotator (mới; điều kiện cần cho "decode theo khẩu vị")

- Chọn extent (14 loại) và min_area theo **từng annotator** (37 người), CV theo fold: **+0.0053**
  (in-sample +0.010). Theo **nhóm** (6 nhóm): **+0.0000**. Vậy kể cả khi biết người gán, lợi ích
  vẫn rất nhỏ.
- Đoán annotator của test từ thời điểm chụp: ảnh train gần nhất về thời gian (trung vị cách 1.5 ngày)
  chung ít nhất một annotator chỉ **6.8%**, **thấp hơn** mức ngẫu nhiên 10.9%. Không đoán được, nên
  hướng này chết, và cũng không phải đụng tới rủi ro dùng metadata.

### 2.5 Trần "đồng thuận người" (mới): 151 ảnh 3 annotator, đoán người thứ 3 từ 2 người còn lại

| Bộ dự đoán cho annotator bị giữ lại | PQ |
|---|---|
| một annotator khác | 0.318 |
| CC của hợp 2 annotator | **0.332** (tốt nhất trong 5 cách gộp) |
| CC của giao | 0.314 |
| các cặp khớp (hợp), bỏ phần lẻ | 0.326 |
| **model (final 1, chưa lọc FP)** | **0.425** |

Model đã hơn mọi cách gộp hai người **+0.09**. Map của model đã là một ước lượng của "annotator trung
bình" tốt hơn những gì hai người thật cung cấp được. Phần còn lại tới trần tự chấm chủ yếu là
**phương sai giữa các annotator**, không phải lỗi giải mã.

## 3. Ý tưởng và khuyến nghị

### Ý tưởng 1. Đóng băng tầng giải mã; chuyển toàn bộ thời gian còn lại sang chất lượng map (KILL có số đo)

- Cơ chế: theo §1, mọi quyết định giải mã chỉ cần E_annotator[s_j(e)]. §2.2 cho thấy cả nhãn thật của
  người khác cũng không cải thiện ước lượng này hơn map (extent âm, giữ/bỏ 0). §2.3 cho thấy GBM trên
  map không vượt được.
- ΔOOF kỳ vọng từ mọi biến thể giữ/bỏ, extent, λ: **0 ± 0.002**. Trần đạt được lạc quan nhất (K=3, hai
  người khác) là +0.01 cho giữ/bỏ, và lọc FP đã lấy một phần.
- Hệ quả cho PLAN: bỏ mọi thẻ hậu xử lý còn mở. Ngưỡng chấp nhận "rẻ, CI > 0" của F48 sẽ chỉ cho ra
  dương giả. Giờ Mac nên dùng cho inference, tái lập và ensemble của N9/N10.

### Ý tưởng 2. Head mô phỏng annotator để ước lượng E[s] (mới so với "style head" đã loại; ưu tiên thấp)

- Khác với thẻ "head theo annotator" bị loại (RESEARCH § 10.3, l. 247): ở đây không điều kiện theo
  annotator của test. Ta lấy trung bình qua M head như M mẫu annotator, rồi dùng s̄ của từng instance
  cho luật s̄ > PQ/2.
- Trần đo được: tương đương oracle chéo với 2 người thật trên K=3, giữ/bỏ **+0.0098**; trên K=2 thì
  âm. Head học được sẽ yếu hơn người thật. Realistic **0 đến +0.003**, tốn một lượt Colab 6.5 h.
- **Không khuyến nghị** trước freeze 09/11; chỉ ghi lại để đóng hướng này.

### Ý tưởng 3. Giải mã theo khẩu vị annotator / nhóm (KILL)

- Biết annotator thì CV +0.005, biết nhóm thì +0.000. Annotator của test không đoán được từ thời gian
  (6.8% so với 10.9% ngẫu nhiên). Rủi ro luật nếu dùng metadata tên file. Chết.

## 4. Đối chiếu với F26/F27/F52–F56

| Đã có | Bản này thêm |
|---|---|
| F26: luật s̄ > PQ/2 | đồng nhất thức mẫu số dạng pooled; áp được cho extent và tách |
| F27: oracle giữ/bỏ +0.058, học được −0.004 (map cũ, val 213) | OOF 707, map final 1: oracle +0.049; học được +0.0004 / −0.0036 |
| F52/F33: near-miss +0.055, extent "có trần" | **oracle chéo**: extent chọn theo người khác **âm** (−0.007), tức trần near-miss là nhiễu annotator |
| F53/F56: chỉnh biên toàn cục ≈ 0 | per-instance hoàn hảo +0.040 nhưng không chuyển giao giữa người (corr s 0.43) |
| F6: người–người 0.34 | gộp 2 người đoán người thứ 3 cũng chỉ 0.33; model 0.425 |

Giới hạn: oracle chéo ước lượng E_annotator bằng 1–2 mẫu, nên đánh giá thấp lời giải Bayes với vô hạn
annotator. Biên độ kéo lên với K=3 so với K=2 (+0.025 ở giữ/bỏ) gợi ý trần Bayes giữ/bỏ chỉ khoảng
+0.01–0.02, extent vẫn ≤ 0. Các extent dãn có thể chồng nhẹ với instance hàng xóm (bỏ qua; chỉ làm
oracle lạc quan hơn).
