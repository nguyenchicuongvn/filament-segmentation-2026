# research07 — lens `annotator_modeling` (06/11)

Câu hỏi: test được chấm theo những annotator mà mình không biết (inter-annotator PQ 0.34). Nếu mô
hình hoá kiểu vẽ của từng annotator, hoặc ước lượng được tỉ lệ các kiểu đó trên test, thì có ăn được
nhiều PQ không? Mình đo mọi thứ trên OOF của final 1 (N1 + N2, hysteresis, gộp theo hướng L16 r6,
bộ lọc `fpfilter_ens_n1n2_orient.pkl`, 707 ảnh, 1,154 bản gán nhãn).

**Kết luận ngắn: không.** Kể cả khi biết trước nhóm annotator hoặc số annotator của từng ảnh, phần
lợi tối đa chỉ +0.001 đến +0.002. Thang suy giảm 06/11 trên LB **khớp** với OOF khi chấm bằng điểm lọc
cross-fit; nhận định cũ "instance điểm thấp trên test đúng nhiều hơn OOF" là do điểm lọc in-sample.
Hướng này không cho đột phá. Đừng tiêu một run Colab nào cho nó.

Script và output (chỉ đọc repo, ghi ở scratchpad):
`/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/annotator_modeling/`
(`styles.py`, `dump.py`, `feats.py`, `analyze.py`, `oracle1.py`, `honest.py`, `ceiling.py`,
`ladder_noise.py`, `ladder_fit2.py`, `shift.py`).

## 1. Cấu trúc annotator trong train JSON

- id của annotator có 6 chữ số. Hai chữ số đầu là nhóm (01–06), bốn chữ số đầu là **team**.
- **Một ảnh luôn do một team gán.** 2–3 annotator của cùng một ảnh luôn chung 4 chữ số đầu, ví dụ
  `0104×3` (70 ảnh), `0501×3` (43), `0101×2` (43), `0204×2` (42). Ảnh một annotator thuộc các team
  0301, 0303, 0604, 0404, 0401, 0201...
- Vì vậy số annotator của ảnh gắn chặt với team: nhóm 01 và 05 chủ yếu ở ảnh 3 người, nhóm 03, 04, 06
  gần như toàn ảnh 1 người.
- Team **không** đoán được từ metadata: mọi team trải đều theo năm 2011–2022 và theo 6 trạm.
- Kiểu vẽ có khác nhau, nhưng không nhiều:

| Nhóm | Bản gán nhãn | Annotator | Filament / bản | Tổng diện tích | Trung vị diện tích | Tỉ lệ < 1k px | PQ model (in-sample) |
|---|---|---|---|---|---|---|---|
| 01 | 516 | 12 | 6.7 | 14.9k | 1,417 | 0.35 | 0.416 |
| 02 | 125 | 5 | 7.2 | 13.8k | 1,231 | 0.43 | 0.488 |
| 03 | 111 | 4 | 8.2 | 14.7k | 1,207 | 0.44 | 0.439 |
| 04 | 156 | 7 | 7.4 | 14.0k | 1,068 | 0.48 | 0.434 |
| 05 | 180 | 7 | 7.2 | 17.5k | 1,420 | 0.35 | 0.461 |
| 06 | 66 | 2 | 7.0 | 15.4k | 1,317 | 0.35 | 0.463 |

Giữa từng annotator, số filament trên mỗi bản dao động 4.4–10.5, còn trung vị diện tích dao động
930–2,006 px.

## 2. Đo trên OOF

### 2a. Lợi tối đa khi chọn đầu ra theo kiểu annotator (oracle)

Mình xét hai loại biến thể:
- **Extent**: dãn hoặc co mọi mask 1 px ở 2048 (kernel 3×3). Biến thể này không dùng điểm lọc nên
  số đo trung thực.
- **Ngưỡng lọc**: điểm lọc được **cross-fit** 5 fold theo ảnh. Mình train lại HGB bằng đúng
  hyperparam của `fp_filter.py`. Ở ngưỡng 0.275 được PQ 0.4325, khớp con số 0.4327 trong docs.

| Oracle (chọn biến thể tốt nhất cho mỗi đơn vị, dựa trên GT) | ΔPQ |
|---|---|
| extent {−1, 0, +1}, biết **nhóm** annotator | **0.0000** (mọi nhóm đều chọn 0) |
| extent, biết **số annotator** của ảnh | 0.0000 |
| extent, chọn riêng **từng ảnh** (gian lận: nhìn GT của ảnh) | +0.0093 |
| extent, chọn riêng **từng bản gán nhãn** (không làm được, vì một ảnh chỉ nộp một bộ dự đoán) | +0.0157 |
| ngưỡng lọc, biết **nhóm** (01: 0.35, 02: 0.175, 05: 0.40, 06: 0.15...) | **+0.0011** |
| ngưỡng lọc, biết **số annotator** (1 người: 0.175, 2: 0.325, 3: 0.425) | **+0.0016** |

Có thấy kiểu vẽ khác nhau: nhóm 02/04 vẽ mảnh hơn (co mask chỉ mất −0.016/−0.022, dãn mất
−0.086/−0.071), nhóm 05 vẽ béo hơn (co −0.062, dãn −0.039). Nhưng ở mức 1 px, mọi nhóm vẫn chọn
"không đổi". Ngưỡng lọc đúng là nên lỏng ở ảnh 1 người và chặt ở ảnh 3 người, khớp với F26: trên ảnh
3 người, nhãn "khớp ít nhất một annotator" quá rộng. Dù vậy, **biết chắc** số annotator cũng chỉ được
+0.0016. Đường cong PQ theo ngưỡng (honest) phẳng: 0.4313–0.4328 cho ngưỡng từ 0.05 đến 0.35.

### 2b. Phần trần phía người

Trên ảnh 3 annotator, mình chấm theo annotator A các phương án sau:
- mask của B: 0.318;
- đồng thuận B ∩ C (mask của B mà C cũng vẽ): 0.313;
- hợp B ∪ C: 0.330;
- model, chấm theo cả 3 người: **0.428**.

Model đã là một "annotator trung bình" tốt hơn bất kỳ người nào. Muốn tăng thì phải làm map/tách
instance tốt hơn. Chọn kiểu người không giúp.

### 2c. Thang suy giảm 06/11: LB khớp OOF khi chấm trung thực

Ladder cũ (`degrade_ladder.txt`) dùng điểm lọc của bộ lọc fit trên chính OOF, tức in-sample. Vì vậy
drop15 hiện là +0.005 (với bộ lọc orient, in-sample còn là +0.019). Khi dùng điểm cross-fit:

| | drop15 | drop30 | drop50 |
|---|---|---|---|
| OOF honest, 707 ảnh | **−0.0082** [−0.012, −0.005] | −0.0347 | −0.0929 |
| SD trên tập con 90 ảnh (cỡ public) | 0.0050 | 0.0082 | 0.0117 |
| LB (khoảng do làm tròn 0.01) | −0.02 (−0.03, −0.01) | −0.04 (−0.05, −0.03) | −0.10 (−0.11, −0.09) |
| ảnh 1 annotator / 2 / 3 | −0.013 / −0.010 / −0.002 | −0.045 / −0.041 / −0.021 | −0.107 / −0.098 / −0.076 |

Cả ba nấc đều nằm trong nhiễu cộng làm tròn. **Không có bằng chứng** rằng instance điểm thấp trên
test đúng nhiều hơn OOF. Dòng ghi trong PROGRESS/EXPERIMENTS ("trên test instance điểm thấp đúng
nhiều hơn OOF") nên sửa thành "LB khớp ladder honest". Nếu có lệch thì chỉ hơi nghiêng về ảnh
1 annotator (số của n1 khớp LB tốt nhất), nhưng theo § 2a, điều đó cũng chỉ đáng ≤ +0.0016.

Mình thử fit "số GT mỗi bản gán nhãn trên test" G từ 4 điểm LB, dùng PQ = S / ½(P + G) (F26) và giả
sử phần S mất ở mỗi nấc bằng OOF. Kết quả: G và S₀ đánh đổi cho nhau. Hậu nghiệm gần phẳng trên
6–16, so với OOF 7.1; có P(G > 7.1) ≈ 0.94, nhưng con số này phụ thuộc prior. LB làm tròn 0.01 trên
~90 ảnh **không xác định được** kiểu annotator của test, khớp với kết luận Blum–Hardt trong
`kaggle_endgame.md`.

### 2d. Lệch metadata không giải thích khoảng cách 0.06

Test có nhiều ảnh 2021–2022 hơn train (12% so với 6%), và PQ năm 2021 thấp (0.385, 28 ảnh). Nhưng
khi cân lại OOF theo phân bố năm của test, PQ chỉ còn 0.4315 (−0.001). Cân theo trạm được 0.4324.
Nhóm có PQ thấp nhất (01) vẫn được 0.416, còn annotator thấp nhất được 0.37. Vậy mọi cách trộn các
annotator **có trong train** đều không kéo được OOF xuống 0.37. Nếu H5 đúng, test phải do những
annotator/team không có trong train gán, hoặc có kiểu vẽ khác hẳn, và train không cho cách nào học
được kiểu đó.

## 3. Ý tưởng và đánh giá

### Ý 1. Model có điều kiện theo annotator/team (multi-head), rồi chọn đầu ra tối đa PQ kỳ vọng dưới hỗn hợp kiểu
- **Cơ chế**: head cho từng team học kiểu vẽ riêng. Lúc inference, chọn mask/instance tối đa PQ kỳ
  vọng theo tỉ lệ các team.
- **Đo**: biết chắc nhóm thì chỉ được +0.0011 (ngưỡng) hoặc 0 (extent). Oracle gian lận theo từng ảnh
  được +0.009. Trên test thì không biết team (S7), và team cũng không đoán được từ metadata.
- Bayes-optimal khi không biết annotator chính là "trung bình theo hỗn hợp", và target mềm hiện tại
  đã là thứ đó.
- **ΔOOF kỳ vọng**: 0 đến +0.002.
- **Quyết định**: **loại** (xác nhận lại quyết định 1.5 điểm ở RESEARCH § 10, lần này có số đo).

### Ý 2. Ước lượng tỉ lệ annotator của test từ thang LB, rồi chỉnh ngưỡng
- **Đo**: ladder honest khớp LB trong nhiễu; fit G không xác định được. Kể cả khi test toàn ảnh
  1 annotator, ngưỡng tối ưu (0.175) chỉ hơn +0.0016.
- Quy tắc PQ/2 của F26 cũng gợi ý nới ngưỡng khi PQ của test thấp hơn (0.37/2 = 0.185 thay vì 0.216).
  Nhưng đường cong ngưỡng phẳng, nên lợi ≤ 0.002.
- **ΔOOF**: ≤ +0.002, không kiểm chứng được.
- **Quyết định**: **loại**. Chỉ giữ một việc là sửa câu diễn giải ladder trong docs.

### Ý 3. Cân lại bản gán nhãn train cho giống test
- Không có tín hiệu nào về kiểu annotator của test. Cân theo năm/trạm chỉ đổi −0.001 PQ.
- **Quyết định**: **loại**.

### Ý 4. Train chống nhiễu annotator (confusion/disentangle heads), chỉ dùng head "đồng thuận"
- **Nguồn**: Tanno et al. CVPR 2019; Zhang et al. NeurIPS 2020 "Disentangling Human Error from the
  Ground Truth"; Lemay 2022 không thấy cách hợp nhãn nào thắng rõ.
- **Cơ chế**: các head riêng hấp thụ khác biệt kiểu vẽ (extent, độ bao), nhờ đó backbone học biên
  "sạch" hơn so với target trung bình.
- **Bằng chứng ngược**: F51 (map hiệu chỉnh tốt không đổi PQ); § 2b (model đã hơn mọi người
  0.10 PQ); `train_recipe.md` (target mềm ≥ lấy mẫu annotator).
- **ΔOOF**: 0 đến +0.005, chưa có số đo. Tốn 1 run Colab, tức phải bỏ EffNetV2-S hoặc 1536 px.
- **Quyết định**: **không làm trước khi đóng băng 09/11**. Chỉ nhắc trong report như hướng tương lai.

## 4. Việc nên làm (rẻ)

1. Sửa diễn giải ladder trong PROGRESS/EXPERIMENTS: dùng số honest (−0.008/−0.035/−0.093 so với LB
   −0.02/−0.04/−0.10, nằm trong nhiễu); giữ ngưỡng lọc 0.275. Ngưỡng tốt nhất theo honest là 0.325
   (+0.0003), tức phẳng.
2. Đưa vào report (4 trang): "một ảnh = một team", oracle theo nhóm/số annotator ≤ +0.002, model
   0.428 so với đồng thuận hai người 0.313. Đây là bằng chứng cho việc phần còn lại là bất đồng giữa
   người với người, không phải kiểu vẽ học được.
3. Muốn đột phá thì phải tìm ở các lens khác (chất lượng map, tách instance). Lens annotator đã cạn.
