# Research 07/11 · lens `new_paradigm`: có paradigm nào nhảy được, không chỉ nhích?

Ngày 06/11. Agent một lens trong workflow research07. Không sửa repo, chỉ viết file này. Script và số
đo nằm ở scratchpad
`/private/tmp/claude-502/-Users-dybala-Working-Learning-kaggle-filament-segmentation-2026/78f944e0-1f2d-4d7b-bf8a-e7dbcdf4ca35/scratchpad/new_paradigm/`
(`mbr_oracle.py`, `mbr_models.py`, `mbr_report.py`, `mbr_oracle_report.txt`, `mbr_models_f0_report.txt`).
Mọi số là OOF trên map `outputs/ens/n1n2_oof` (final 1, config `post_ens_n1n2_hyst_orient.json`,
**không** lọc FP). Bootstrap 1000 lần, ghép cặp theo ảnh.

## Tóm tắt (đọc phần này là đủ)

1. **Số đo mới quan trọng nhất (đề xuất mã F57)**: ngay cả khi có một **mô hình sinh annotator hoàn
   hảo**, giải mã PQ tối ưu kiểu Bayes (MBR) cũng chỉ cho **+0.0036 [−0.007, +0.014]**. Ở đây mô hình
   hoàn hảo được thay bằng chính các annotator thật còn lại của ảnh. Oracle dùng nhãn của chính
   annotator được chấm thì cho +0.166. Khoảng cách 0.16 giữa hai số đó là **nét riêng của từng
   annotator**: không ảnh nào dự đoán được, nên không model nào lấy lại được. Hệ quả là mọi paradigm
   "mô hình hoá phân phối annotator" (Probabilistic U-Net, PHiSeg, diffusion segmentation, head theo
   nhóm annotator + MBR) bị **chặn trên ở khoảng +0.01**. Hướng này coi như chết, và chết vì có số đo.
2. Foundation model (DINOv2/SAM/SAM2, đều Apache-2.0; **DINOv3 dùng license riêng của Meta, không
   phải OSI → loại**) đi ngược hai số đo có sẵn: model sống nhờ chi tiết dưới kích thước patch (F37,
   F42), và các đội khác đã thử Mask2Former/SAM đều kém (S9e). Kỳ vọng ≤ 0. Lại cần code mới, không
   kiểm chứng kịp trước 09/11. **Loại.**
3. Các phương pháp instance trực tiếp cho vật mảnh (skeleton + bề rộng, affinity/mutex watershed,
   embedding, polyline) có trần đã đo: gộp/cắt hoàn hảo +0.023 (F54), gộp nhầm +0.028 (F52). Thực tế
   được +0.003 đến +0.008, cần head mới. **Không làm trước khi đóng băng**, chỉ đưa vào report.
4. Cascade 1024 → crop 2048 và refine bằng diffusion: phần "cascade" đã có dưới dạng N1 + N2 (và N10
   1536 đang trong hàng đợi). Phần refine biên nhắm vào lỗi biên, mà lỗi biên chủ yếu là bất đồng
   giữa annotator (F31, F53). **Loại.**
5. Hướng duy nhất còn đáng một lượt GPU, nếu muốn đặt cược: **Mask R-CNN ở độ phân giải cao làm thành
   viên ensemble khác paradigm**. Code có sẵn (`filseg.maskrcnn`). Có một bằng chứng ngoài: S9a, Mask
   R-CNN đơn đạt 0.429, với kiểu lỗi khác U-Net. Kỳ vọng ensemble +0.003 đến +0.010, xác suất qua gate
   khoảng 25%. Chỉ làm nếu thay chỗ N8 (seed 2, vốn được dự báo trượt gate) và fold 0 về trước 08/11.

**Kết luận thẳng**: trong 3 ngày và tối đa 1 lượt Colab còn trống, không có paradigm nào hứa hẹn
+0.03. Phép đo MBR cho thấy vì sao: phần PQ còn thiếu (1 − 0.43) gần như toàn bộ là nét riêng của
annotator, không phải năng lực model. Các đội trung thực đạt 0.45–0.46 OOF đều nhờ cộng dồn
ensemble/hiệu chỉnh, không nhờ paradigm mới (S9c, S10, S15). Khoảng cách đến giải thưởng sẽ không lấp
được bằng kiến trúc trong tuần này.

## 1. MBR với phân phối annotator: oracle "mô hình sinh hoàn hảo" (mới, đã đo)

### Ý tưởng

Metric chấm từng annotator riêng (F3). Vì vậy dự đoán tối ưu là tập instance làm cực đại PQ **kỳ
vọng** theo p(nhãn | ảnh). Theo F26, giữ một candidate c sẽ cộng IoU·hit vào tử số và luôn cộng ½ vào
mẫu số, nên luật là: giữ c nếu E_b[IoU_b(c)·hit_b(c)] > λ ≈ PQ/2. Nếu có K mẫu nhãn từ p(nhãn | ảnh),
ví dụ từ Probabilistic U-Net, PHiSeg, diffusion segmentation hay head theo nhóm annotator, thì E[·] ước
được trực tiếp. Không cần bộ dự đoán IoU (bộ này đã thất bại với R² 0.26, F27), và cũng không cần
biết annotator của test (lý do head style bị loại ở § 10.3).

### Cách đo trần

Với mỗi ảnh có ≥ 2 annotator (296 ảnh, 743 bản gán nhãn), lần lượt giữ lại một annotator a để chấm.
Các annotator còn lại B = A∖{a} đóng vai các mẫu từ p(nhãn | ảnh). Đây đúng là thứ một mô hình sinh
hoàn hảo cho ra với |B| mẫu, vì các annotator độc lập có điều kiện theo ảnh. Có hai cách chọn
candidate:
- `keep`: giữ/bỏ từng instance của final 1;
- `tree`: chọn antichain bằng DP trên cây thành phần cc ở các ngưỡng 0.2–0.9, nên chọn được cả **hình
  dạng** của instance.

`treeO` là cùng phép DP nhưng dùng nhãn của chính a. Con số này bị rò rỉ, chỉ để thấy thang đo (giống
F27).

| Tập bản gán nhãn | base (final 1, không lọc FP) | keep (λ ≤ 0.215) | tree λ 0.215 | treeO (rò rỉ) |
|---|---|---|---|---|
| mọi ảnh ≥ 2 annotator (743) | 0.4313 | +0.0000 [−0.007, +0.007] | **+0.0036 [−0.007, +0.014]** | +0.166 |
| ảnh 2 annotator (1 mẫu) | 0.4411 | −0.0153 [−0.028, −0.004] | −0.0046 [−0.021, +0.010] | +0.161 |
| ảnh 3 annotator (2 mẫu) | 0.4249 | **+0.0098 [+0.002, +0.018]** | +0.0089 [−0.005, +0.023] | +0.169 |

Đọc bảng:
- Một mẫu thì hại: chỉ giữ những gì một người khác cũng vẽ sẽ cắt mất TP, nhiều hơn số FP bỏ được.
- Hai mẫu được +0.01. Với K → ∞ mẫu hoàn hảo, lợi ích có thể cao hơn. Tuy vậy, ngoại suy từ 1 → 2
  mẫu (−0.015 → +0.010) cho thấy trần thực tế khó vượt **+0.02**, ngay cả với một sampler hoàn hảo.
- Một sampler **học** từ khoảng 700 ảnh sẽ kém xa sampler hoàn hảo. Bằng chứng: bản chỉ dùng ảnh
  (dưới đây) đã ≈ 0, và F27 cho thấy dự đoán IoU từ map rất kém.
- Khoảng 0.166 − 0.004 ≈ **0.16 PQ không lấy lại được bằng bất kỳ thông tin nào có trong ảnh**. Phần
  này nên vào report: giới hạn của bài toán là nét riêng của annotator, không phải model.

### Bản thực hiện được, chỉ dùng ảnh (zero GPU)

Ở đây các mẫu là các thành viên ensemble: pipeline final 1 chạy riêng trên map N1, N2 và BCE của cùng
fold. Đo trên fold 0, 142 ảnh, 226 bản gán nhãn. Kết quả: `tree` **−0.014** [−0.025, −0.004];
`keep` λ 0.4 +0.0025 [−0.0003, +0.0053], và λ này được chọn in-sample sau khi xem số. Các thành viên
quá giống nhau nên không thay được annotator, giống kết luận của W-agree (−0.0004). **Không đề xuất.**

**Quyết định**: dừng mọi hướng generative/probabilistic (diffusion, Prob. U-Net, head theo nhóm
annotator + MBR). Chi phí: 1 lượt Colab và code mới. Kỳ vọng ≤ +0.005, và đó là trong trường hợp tốt
nhất.

## 2. Mask R-CNN độ phân giải cao làm thành viên khác paradigm (một phần mới so với N4)

- **Cơ chế**: lợi của một thành viên tăng theo độ khác biệt với các thành viên còn lại (khác độ phân
  giải +0.0105, khác loss +0.0018, `ensemble_diversity.md`). Detector khác U-Net ở chính cách tạo
  instance: nó có score cho từng đối tượng, ra quyết định gộp/cắt trực tiếp, và ít mảnh vụn hơn. Đây là
  mức khác biệt lớn nhất còn có thể đạt được với code đã có.
- **Khác N4 ở đâu**: N4-mac chạy @512, 8 epoch, và chỉ dùng detector để quyết cách chia (−0.006). Ở
  đây đề xuất (a) @1024 với 24 epoch, hoặc crop từ ảnh 2048 như biconcavelens (S9f: train @2048, 6
  epoch, GPU 16 GB); (b) dùng như **thành viên bỏ phiếu ở mức instance** có hiệu chỉnh isotonic. Không
  hiệu chỉnh thì mất 0.021 (S9b), nên đây là phần bắt buộc.
- **Bằng chứng ngoài**: S9a, Mask R-CNN đơn 0.429 so với U-Net của họ 0.396. Bỏ phiếu cho +0.007 đến
  +0.012, nhưng CI > 0 chỉ ở 3/5 lần chia (S9c). Bằng chứng ngược: U-Net của họ yếu hơn của mình, và
  head mask 28×28 kém với vật dài (§ 11.2).
- **Trần**: phần mà detector có thể sửa là gộp/cắt: +0.023 (F54) đến +0.028 (F52), cộng với FP mồ côi
  +0.010. Lấy được 20–30% phần này thì ra **+0.005 đến +0.010**.
- **Chi phí và luật**: torchvision là BSD-3, đạt OSI. Fold 0 @1024 mất ~1.5 h L4, 5 fold ~7 h, tức 1
  lượt Colab, phải **thay N8**. Nếu đến tối 07/11 fold 0 chưa về thì bỏ, vì không kịp 5 fold trước
  09/11. Còn phải giải chồng lấn và thêm assert mask rời nhau (luật S3).
- **Bước đầu, chạy được hôm nay**: thêm `mrcnn_s1024_f0` vào đầu queue thay N8. Gate trên fold 0:
  ensemble ở mức instance với N1 + N2 phải hơn N1 + N2 với Δ ≥ +0.008 và CI > 0, rồi mới chạy fold
  1–4.

## 3. Foundation-model encoder: DINOv2 / SAM / SAM2 (loại, có lý do)

- **License**: DINOv2, SAM và SAM2 là Apache-2.0, dùng được. DINOv3 dùng "DINOv3 License" riêng của
  Meta, không phải license OSI chuẩn → **cấm theo luật của mình**.
- **Cơ chế ngược chiều**:
  - Model phụ thuộc chi tiết mịn: làm mờ σ 1.5 px thì −0.096 (F37). Patch 14 px ở 1024 tương đương
    28 px ở 2048, trong khi filament chỉ rộng cỡ chục px.
  - Encoder không có skip stride 2 đã thua −0.020 (ConvNeXt, F42).
  - Phải thêm adapter CNN độ phân giải cao (kiểu ViT-Adapter): code mới, chưa ai kiểm chứng trên dữ
    liệu này.
- **Bằng chứng ngoài**: biconcavelens có Mask2Former ≤ 0.345 và SAM với box prompt 0.316 (S9e).
  Không có paper filament nào dùng foundation model (`filament_papers.md`). Nghiên cứu DINOv2 frozen
  trên ảnh y tế nói chính encoder frozen là giới hạn ([arXiv 2312.02366](https://arxiv.org/pdf/2312.02366)).
  Không tìm thấy benchmark DINOv2 cho mạch máu mảnh.
- **Kỳ vọng**: −0.02 đến +0.005. Tốn ≥ 1 lượt Colab cộng code mới. **Không kiểm chứng kịp trước
  09/11.**

## 4. Instance trực tiếp cho vật mảnh: skeleton + bề rộng, affinity, embedding, polyline (hoãn)

- Phần "chia instance" có trần đã đo: +0.023 (oracle gộp mảnh, F54). Có 257 lỗi gộp nằm gọn trong
  một thành phần liên thông; chỉ affinity/embedding mới tách được chúng, trần tổng của nhóm gộp nhầm
  là +0.028 (F52).
- Biểu diễn skeleton + bề rộng chỉ làm mượt biên. Biên lại nhạy dưới 1 px (dãn/co 1 px: −0.014 /
  −0.029), và spine không mang thông tin mới so với mask (F54).
- Hao & Li 2026 trên MAGFiLO: các loss skeleton/component **không** cho lợi ổn định (S22).
- Kỳ vọng +0.003 đến +0.008. Cần head mới và ~6.5 h Colab cho 5 fold, không còn slot. Đưa vào report
  như hướng tương lai (S23).

## 5. Cascade và refine bằng diffusion (loại)

- Cascade "detect ở 1024, refine ở 2048" thực chất là N1 + N2 (+0.0105, đã có trong final 1) và N10
  1536 (đang trong hàng đợi).
- Một refiner (CascadePSP/SegRefiner) nhắm vào lỗi biên, chiếm 36% của 1 − PQ. Nhưng 59% dự đoán gần
  trúng lại là TP với một annotator khác (F31), và chỉnh biên theo cường độ ảnh đều âm (F56).
- Kỳ vọng ≤ +0.003. Lại cần một codebase mới. **Loại.**

## Nguồn ngoài đã đọc

- DINOv2 frozen trên benchmark radiology: [arXiv 2312.02366](https://arxiv.org/pdf/2312.02366)
- DINOv3 License (license riêng, không phải OSI): [Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-license), [issue #31](https://github.com/facebookresearch/dinov3/issues/31)
- Segmentation xác suất với nhiều annotator: Probabilistic U-Net [arXiv 1806.05034](https://arxiv.org/pdf/1806.05034), diffusion phân loại có điều kiện [arXiv 2303.08888](https://arxiv.org/pdf/2303.08888). Cả hai đều đánh giá bằng GED/Dice; không bài nào giải mã theo PQ.
- Không tìm thấy công trình nào làm MBR cho PQ với nhiều annotator. Theo những gì tìm được, phép đo ở
  § 1 là mới.
