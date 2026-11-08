# Research 06/11 — `kaggle_endgame`: dùng 10 ngày cuối và 5 lượt nộp/ngày thế nào

Phạm vi: không GPU, không model mới. Trả lời bốn câu: (1) lượt nộp nào còn mang thông tin khi LB
chỉ hiện 2 chữ số; (2) chọn 2 final khi OOF và LB lệch nhẹ; (3) checklist hồ sơ nhận giải và các
lỗi hay gặp 48 h cuối; (4) rà lại PLAN § 5b và đề nghị sửa. Mọi số OOF/LB lấy từ
`docs/EXPERIMENTS.md`, `docs/RESEARCH.md`, `docs/PROGRESS.md`; số mô phỏng mới nằm ở § 1.2
(script inline, chạy bằng `.venv` ngày 06/11, seed 0). Trang Kaggle (overview, discussion) không
đọc được bằng WebFetch (chỉ trả về title, cần JS/đăng nhập), nên luật của host lấy từ
`PRIZE_RULES.md` (đọc 19/10) và RESEARCH S6/S6b/S8/S19/S21 (đọc 29/10–05/11). CLI `kaggle`
(read-only) xác nhận 8 bài gần nhất đều `COMPLETE`, public 0.36–0.37, cột private trống.

## 0. Kết luận ngắn

1. **LB public 90 ảnh, 2 chữ số, không phân biệt được bất kỳ thay đổi nào còn lại** (mọi đòn bẩy
   còn lại ≤ +0.005 OOF). Mô phỏng ghép cặp (§ 1.2): một thay đổi thật +0.005 chỉ hiện +0.01 trên LB
   với xác suất 0.50; +0.002 → 0.26; **không đổi gì (null) → 0.19**. Tức "LB nhích 0.01" gần như
   không có nghĩa. 9 bài gần nhất đều 0.37 là đúng kỳ vọng, không phải tín hiệu xấu.
2. Chỉ **ba loại bài nộp còn giá trị**: (a) kiểm tra tái lập (fp16, notebook Kaggle) — nhưng giá
   trị nằm ở so sánh **CSV/PQ cục bộ**, LB chỉ bắt được lỗi thô; (b) bảng "model đơn so với ensemble
   trên test" (3 fold đơn) — thứ duy nhất OOF không đo được, giá trị yếu nhưng miễn phí; (c) 2 bài
   final. Mọi bài thăm dò hậu xử lý (min_area, seed_thresh, group_radius, trọng số N1:N2, ngưỡng
   P/R) **nên bỏ**: đã đo được trên OOF (≈ 0 hoặc âm), và LB không thể thấy.
3. **Chọn final theo OOF, không có "bất đồng" thật giữa OOF và LB**: Spearman(val, LB) 0.92 trên 12
   bài (F9); hai final đều 0.37 vì LB không có độ phân giải, không phải vì LB nói ngược. Văn liệu
   (Roelofs et al. NeurIPS 2019; Blum & Hardt 2015; Kaggle Book) đều ủng hộ tin CV, và đặc biệt
   cảnh báo đúng trường hợp **test public 90 mẫu** là ngoại lệ "pathological". Kaggle lấy điểm private
   **cao hơn** trong 2 bài được chọn, nên final 2 phải là bài **khác nhiều nhất** mà vẫn OOF cao,
   không phải tập con của final 1 (N1 đơn hiện là tập con của N1 + N2, giá trị phòng hộ ≈ 0).
4. **Rủi ro lớn nhất 10 ngày tới không nằm ở PQ mà ở hồ sơ**: notebook end-to-end **chưa tồn tại**
   (`kaggle/` chỉ có `make_kernel.py` để train); 10 checkpoint final nằm trong `outputs/colab/*/`
   (**git-ignored**, fp32 82 MB mỗi file); bộ lọc FP là pickle sklearn 1.9.1 (vỡ khi lệch version);
   `requirements.txt` ghim torch 2.14.1 / pandas 3.1.0 / numpy 2.5.4 khó cài trên kernel Kaggle; repo
   private đến khi đóng nên notebook không thể `git clone` lúc test. Tất cả phải xong **trước 09/11**,
   không phải 10–11/11 như PLAN § 5b.

## 1. Câu 1 — Bài nộp nào còn mang thông tin ở độ phân giải 2 chữ số

### 1.1 Cơ sở lý thuyết

- **Blum & Hardt, "The Ladder: A Reliable Leaderboard for Machine Learning Competitions" (ICML
  2015, [arXiv 1502.04585](https://arxiv.org/abs/1502.04585))**: cơ chế Ladder chỉ cập nhật điểm khi
  bài mới hơn điểm cũ quá bước η và làm tròn điểm theo bội của η; sai số LB của Kaggle cổ điển tăng
  theo √k (k = số bài nộp), còn Ladder chỉ tăng theo (log k / n)^{1/3}. Thí nghiệm boosting của họ:
  public n = 4,000, Kaggle làm tròn 10⁻⁵, kẻ tấn công nộp vector ngẫu nhiên rồi bỏ phiếu đa số →
  điểm public bị thổi lên trong khi điểm trên 8,000 mẫu còn lại vẫn ≈ 1/2. **Hệ quả cho mình**: LB
  làm tròn 0.01 trên n = 90 chính là một Ladder với bước **rất lớn** (η = 0.01 so với SD lấy mẫu
  0.014); nó gần như không rò thông tin về nhãn test, nên (i) không thể "dò" phân bố annotator của
  test, (ii) cũng không có rủi ro overfit LB với ~50 lượt nộp còn lại.
- **Roelofs et al., "A Meta-Analysis of Overfitting in Machine Learning" (NeurIPS 2019,
  [paper](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning))**:
  120 cuộc thi Kaggle; khoảng cách public–private đối xứng quanh 0 trong ±2%, "little evidence of
  substantial overfitting"; **ngoại lệ gần như toàn là cuộc thi có test nhỏ hoặc split không i.i.d.**,
  ví dụ competition 12681 với **public test 90 mẫu**, 12349/5903 với private 209/100 mẫu; họ đề nghị
  ≥ 10,000 mẫu test để chống overfit thích nghi. Cuộc thi này: public ≈ 90 ảnh, private ≈ 90 ảnh →
  đúng loại "pathological". Nguồn nhiễu chính là **lấy mẫu**, không phải overfit.
- **Kaggle Book (Banachewicz & Massaron), chương validation** ([tóm tắt](https://alibakh62.github.io/pages/content/book_summary/the_kaggle_book/designing_good_validation.html)):
  "Probing ... works particularly well if the private test set is similar to the public one";
  "There are no general method for probing". Với làm tròn 0.01 và 5 lượt/ngày, probing kiểu
  mlcourse.ai ("nộp các bài khác nhau một giá trị rồi giải phương trình") là bất khả.

### 1.2 Mô phỏng độ phân giải LB cho các thay đổi còn lại (số mới, 06/11)

Giả định: điểm public của bài gốc có SD lấy mẫu 0.0138 trên 90 ảnh (F8, bootstrap); Δ ghép cặp trên
90 ảnh có SD = (CI trên 707 ảnh) × √(707/90); hai điểm được làm tròn 2 chữ số độc lập; 400k lần.

| Thay đổi (Δ OOF thật, CI 707 ảnh) | SD ghép cặp 90 ảnh | P(LB tăng ≥ 0.01) | P(LB không đổi) | P(LB giảm ≥ 0.01) |
|---|---|---|---|---|
| G4: N1 + N2 so với N1 (+0.0105 [+0.0072, +0.0140]) | 0.0049 | **0.83** | 0.16 | 0.00 |
| UP2048 cho final 2 (+0.0050 [+0.0029, +0.0069]) | 0.0029 | 0.50 | 0.50 | 0.00 |
| BCE 15 so với 10 model (+0.0018 [−0.0008, +0.0045]) | 0.0038 | 0.26 | 0.67 | 0.08 |
| **Không đổi gì (null)** | 0.0049 | **0.19** | 0.62 | 0.19 |

- Đọc: G4 là thay đổi **duy nhất** lớn đến mức LB "nên" thấy (0.83), và LB **đã không thấy**
  (N1 0.37 → N1 + N2 0.37; rơi vào 16%). Mọi thứ sau đó (≤ +0.005) có P(thấy) 0.26–0.50 so với
  nền 0.19 → một bài nộp gần như vô nghĩa.
- "Xu hướng qua nhiều điểm" (PLAN § 5b mục 1) cũng không cứu được: với 3 biến thể cùng +0.005, xác
  suất **ít nhất một** bài hiện +0.01 là 0.73, nhưng dưới null xác suất ít nhất một trong 3 hiện
  +0.01 cũng ≈ 0.47. Không tách được.
- Hai bài **duy nhất** từng tách được trên LB là các bước ≥ +0.01 val (ví dụ R34 → EffNet-B4 +0.013
  val → LB 0.34 → 0.36; 5 fold → +0.01). Đây đúng là F9 (LB ≈ 0.84 × val, Spearman 0.92 trên 12
  bài): LB **có** độ nhạy cho bước lớn, và mình đã có đủ "thang đo" từ 0.21 đến 0.37 cho report. Không
  cần nộp biến thể cố ý làm kém để đo độ nhạy nữa.

### 1.3 Phân loại bài nộp theo giá trị thông tin

| Loại | Giá trị | Lý do | Khuyến nghị |
|---|---|---|---|
| **Final 1, final 2** | bắt buộc | | nộp sớm (đã có), nộp lại khi weights/notebook đổi |
| **Tái lập fp16 / notebook Kaggle** | cao, nhưng **không nằm ở LB** | LB bằng 0.37 là điều kiện cần, không đủ: một regression −0.005 vẫn cho 0.37 (bảng trên). Giá trị thật ở so sánh cục bộ: CSV trùng từng byte hoặc PQ(mask fp16 chấm theo mask fp32) ≥ 0.995 | nộp mỗi bản 1 lần; định nghĩa đạt bằng số cục bộ (§ 3.2) |
| **Model đơn fold 1–3 (bảng "đơn so với ensemble" trên test)** | thấp–trung bình | là đại lượng OOF **không thể** đo (mỗi ảnh chỉ có 1 fold OOF); đã có f0 0.36 so với 5 fold 0.37. 3 điểm nữa cho report một câu thật: "trên test, 4/4 fold đơn 0.36, ensemble 0.37" (hoặc không) | giữ, nhưng chỉ khi còn lượt trống; ưu tiên thấp |
| **Ablation hậu xử lý trên LB** (r = 0, không lọc FP, cc, ngưỡng) | ≈ 0 | đã nộp 05/11: tất cả 0.37, đúng dự báo. Bảng ablation cho report phải là **OOF có CI**; cột LB chỉ để ghi "LB không phân giải" | bỏ các bài còn lại (cc mode, ...); một câu trong report là đủ |
| **Thăm dò P/R, lệch annotator** (t 0.3/0.4/0.5, min_area, seed, radius) | 0 | Blum–Hardt: làm tròn 0.01 trên 90 mẫu không rò thông tin; 05/11 ba ngưỡng đều 0.37; H5 không kiểm chứng được (S7) | **bỏ hẳn** |
| **Trọng số N1:N2, 15 model** | 0 | đã đo trên 707 OOF: 2:1 +0.0004, 1:2 −0.0018, BCE +0.0018 CI chạm 0; LB không thể thấy | bỏ (EXPERIMENTS đã ghi "bỏ 2 bài 06/11") |
| **Biến thể cố ý làm kém để đo độ nhạy LB** | 0 | đã có thang 0.21 → 0.37 với val tương ứng (F9) | không làm |

## 2. Câu 2 — Chọn 2 final khi OOF và LB lệch nhẹ

### 2.1 Văn liệu và thực hành Kaggle

- Kaggle Book: "One common error in Kaggle competition is to rely on public leaderboard ranking";
  "public standings usually do hold in the unveiled private leaderboard"; chọn final: **một theo LB,
  một theo CV**; nếu không tin LB (test nhỏ, không i.i.d.) thì **"picking two very different models
  or ensembles"** với CV cao. Hệ thống khác biệt có sai số ít tương quan → max của hai bài có kỳ vọng
  cao hơn.
- Slide "Becoming a Kaggle Master" lec 3 ([pdf](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf)):
  "Trust your CV ... CV is better 90%+ of the time"; đội giữ bài theo CV ở Santander có biến động hạng
  cuối thấp hơn hẳn. Thứ tự thực hành: trust CV + LB > trust CV > trust LB.
- Roelofs 2019 (§ 1.1): với test nhỏ, khoảng cách public–private là **nhiễu lấy mẫu**, không phải
  tín hiệu để chọn.
- Cơ chế Kaggle: chọn tối đa 2 bài; điểm cuối là **điểm private cao hơn** trong 2 bài (DataCamp:
  "the final result is taken from the best score out of these two final submissions"). Nếu không
  chọn, Kaggle lấy các bài có public cao nhất (quy tắc này không đọc được trực tiếp từ docs Kaggle
  hôm nay — trang chỉ trả title; coi là **chưa xác minh** nhưng là lý do phải chọn tay: 9 bài đang
  **hoà 0.37**, auto-chọn sẽ tuỳ ý).

### 2.2 Áp vào dữ liệu của mình

- OOF và LB **không mâu thuẫn**: F9 Spearman 0.92; mọi bài ≥ 0.42 OOF đều 0.37 LB; không có bài nào
  OOF thấp mà LB cao. "Lệch nhẹ" chỉ là làm tròn. Vậy quy tắc "một theo LB, một theo CV" **thoái hoá**
  (LB không chọn được gì) → dùng quy tắc "hai bài khác nhau nhiều, cùng OOF cao".
- **Final 1** = ứng viên OOF cao nhất đã qua gate: N1 + N2 (0.4284 cross-tune), thêm N8 nếu qua gate
  CI > 0 trên 707 ảnh (thẻ N8). Không đổi.
- **Final 2 hiện tại (N1 5 fold UP2048, OOF 0.4229)** là **tập con** của final 1: nó thua final 1 ở
  **5/5 fold** (G4) và map của nó là một nửa map final 1, nên sai số trên private gần như đồng biến
  với final 1. Giá trị phòng hộ ≈ 0; nó chỉ "an toàn" theo nghĩa đơn giản hơn (S6b), nhưng rubric chấm
  **bài được chọn + repo**, không chấm độ đơn giản của bài dự phòng.
- **Đề nghị final 2** = bài có OOF ≥ final 1 − 0.002 và **khác final 1 theo một trục OOF không phân
  xử được**. Ứng viên theo thứ tự:
  1. **N1 + N2 + BCE (15 model, OOF 0.4302, +0.0018 [−0.0008, +0.0045], P>0 0.91)**: kỳ vọng OOF
     ≥ final 1, cơ chế khác (map BCE hiệu chỉnh p ≈ tỉ lệ annotator, FP −100). Nếu test private có
     nhiều ảnh 1 annotator hơn (F16: FP thật tập trung ở ảnh 1 annotator) thì bài này có lợi hơn.
     Không qua gate "vào final 1" nhưng **gate không áp cho slot phòng hộ**: slot này tối đa hoá
     E[max(private₁, private₂)], tức cần OOF gần bằng và tương quan thấp, không cần CI > 0.
  2. Nếu N8 về và qua gate: final 1 = N1 + N2 + N8; final 2 = N1 + N2 (+ BCE).
  3. Chỉ dùng N1 đơn làm final 2 nếu muốn **bài dự phòng khi notebook ensemble hỏng** — nhưng điều đó
     thuộc checklist hồ sơ, không thuộc chọn final (notebook có thể demo cả hai cấu hình).
- Kỳ vọng thật: Δ private giữa hai cách chọn final 2 nằm trong ±0.003; đây là quyết định đúng về
  nguyên tắc hơn là đòn bẩy điểm. Khoảng cách tới các đội 0.40 (≈ 0.03 trên cùng ảnh test) là thật
  (SD ghép cặp giữa hai đội ~0.005–0.008), không do chọn final.
- **Thủ tục**: chọn 2 final trên Kaggle **chậm nhất 13/11**, chụp màn hình vào `docs/logs/`, kiểm tra
  lại sau mỗi lần nộp thêm (nộp thêm không đổi lựa chọn, nhưng hãy nhìn lại), và ghi vào PROGRESS.

## 3. Câu 3 — Checklist hồ sơ nhận giải và lỗi hay gặp 48 h cuối

Luật (PRIZE_RULES.md, S6, S19): repo public **ngay khi đóng** (15/11 06:00 UTC = 13:00 VN), giữ
public đến khi công bố (30/11); `requirements.txt` ghim version; **một notebook** chạy toàn bộ
pipeline, load weights **có sẵn trong repo**; không cần file nào ngoài repo; report 4 trang theo
template Overleaf; Google form trước **15/11 11:59 pm CDT** (= 16/11 12:59 VN); người nộp phải giải
thích được mọi thành phần. Rubric 70% định lượng (PQ, Dice, phân bố, 1:n / n:1, **hiệu năng
end-to-end**), 30% định tính (mô tả pipeline, hình mask, chất lượng code).

### 3.1 Tình trạng repo hôm nay (đo trực tiếp 06/11)

| Mục | Thực tế | Rủi ro |
|---|---|---|
| Notebook end-to-end | **chưa có** (`kaggle/make_kernel.py` chỉ dựng kernel train; `colab/train_unet.ipynb` là train) | mục bắt buộc duy nhất chưa bắt đầu |
| Weights final | 10 file `outputs/colab/<run>/<run>.pt`, fp32 **81.7 MB** mỗi file; `outputs/` **git-ignored** | clone repo sạch → không có weights → notebook chết. fp16 ≈ 41 MB/file × 10 = ~410 MB (+ BCE 5 file nếu final 2 dùng) → dưới 1 GB "ideal" của GitHub, mỗi file < 100 MiB (GitHub chặn ≥ 100 MiB, cảnh báo ≥ 50 MiB; [docs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)) |
| Bộ lọc FP | `configs/fpfilter_ens_n1n2.pkl` = pickle sklearn GBM, sklearn ghim 1.9.1 | pickle sklearn **không tương thích giữa version**; kernel Kaggle có sklearn khác → `pip install scikit-learn==1.9.1` phải chạy được; nếu không, lọc FP chết. Lợi ích của lọc FP trên ensemble chỉ +0.0018 in-sample (0.4294 → 0.4312; LB không đổi khi bỏ) → **ứng viên bỏ khỏi final** để giảm độ phức tạp (S6b) |
| `requirements.txt` | torch 2.14.1, torchvision 0.29.1, numpy 2.5.4, pandas 3.1.0, smp 0.5.0, pycocotools 2.0.11 | kernel Kaggle có torch/numpy cài sẵn khác; `pip install torch==2.14.1` tải ~2 GB và có thể lệch CUDA. Notebook phải chạy được với torch có sẵn (chỉ cài smp + pycocotools) **và** ghi rõ version thật dùng khi tạo CSV |
| Tải weights ImageNet | `infer_probs.py` dựng model với `pretrained=False` → không cần internet | OK; giữ nguyên trong notebook (một lỗi cổ điển là `encoder_weights="imagenet"` kéo tải lúc infer) |
| Map xác suất | lưu **uint8** (`round(p·255)`) rồi hậu xử lý trên uint8 | **điểm lợi lớn cho tái lập**: fp16/GPU khác chỉ đổi pixel khi p vượt biên 1/255; CSV gần như trùng. Nhưng không đảm bảo trùng từng byte (cuDNN/MPS khác nhau, docs PyTorch: không bảo đảm reproducible giữa CPU/GPU/platform) → tiêu chí phải là **dung sai** (§ 3.2) |
| Repo private | notebook không thể `git clone` lúc test trước 15/11 | test bằng **dataset Kaggle riêng tư** chứa snapshot repo (cùng layout); notebook có cờ `REPO_SRC = "dataset" | "git"`; sau khi public, chạy lại một lần bằng `git clone` |
| Thời gian infer | chưa đo trên Kaggle (PROGRESS "đang chờ") | 10 model × TTA 8 × 180 ảnh, N2 ở 2048 (crop) nặng gấp ~4 N1; ước ~30–60 phút T4. Host: chỉ loại "quá chậm hoặc rò bộ nhớ" (S8) → chấp nhận được nhưng phải **ghi số** vào report; cân nhắc cell "lite" (N1 5 fold) có timing riêng |
| Mask chồng nhau | mask dựng từ một ảnh nhãn nên rời nhau theo cách dựng; **chưa có assert** (RESEARCH § 8) | thêm assert trong `make_submission.py` + cell kiểm tra trong notebook (host báo lỗi khi nộp mask chồng, S21) |
| Hạng/ID | `filament_id = <stem>_<k>`, k theo thứ tự label; RLE pycocotools | thứ tự instance phụ thuộc `label()` → nếu notebook dùng scikit-image version khác, k có thể đổi thứ tự → CSV khác byte nhưng PQ bằng. Tiêu chí tái lập phải so **tập mask**, không so byte |

### 3.2 Tiêu chí tái lập (thay cho "LB phải bằng")

Viết `scripts/compare_submissions.py` (CPU, vài phút): nhận hai CSV, ghép theo ảnh, báo
(a) số instance mỗi ảnh; (b) PQ của CSV B chấm theo CSV A làm GT (và ngược lại); (c) số pixel khác.
**Đạt** nếu PQ(B|A) ≥ 0.995 và số instance lệch ≤ 0.5 %; **trùng byte** là lý tưởng nhưng không bắt
buộc. Áp cho: fp16 so với fp32 (Mac), notebook Kaggle so với Mac, `git clone` sạch so với working
tree. LB chỉ cần "0.37" cho mỗi bản, một lần.

Bằng chứng gián tiếp về dung sai fp16: model card ACR-instance-segmentation (HF, 2026) báo mask mIoU
fp16 thấp hơn fp32 0.0008, F1 +0.0022, "within measurement variation"; kết hợp map uint8 của mình,
kỳ vọng PQ(fp16|fp32) > 0.995 (chưa đo, đo ở mục 06/11 bên dưới).

### 3.3 Checklist theo thứ tự (hoàn tất trước 09/11, không phải 10–11/11)

1. **Weights**: `weights/<run>.fp16.pt` (state_dict + meta json), bỏ `outputs/` khỏi đường dẫn load;
   cập nhật `.gitignore`; kiểm tra `git clone --depth 1` thời gian tải và dung lượng (< 1 GB).
2. **fp16 == fp32**: infer lại test từ weights fp16 trên Mac → CSV → `compare_submissions.py` so với
   `submission_ens_n1n2.csv`. Nộp 1 lần.
3. **Notebook `notebooks/final_inference.ipynb`**: cell 0 cấu hình (`REPO_SRC`, đường dẫn test), cell
   cài `segmentation-models-pytorch==0.5.0 pycocotools==2.0.11` (không cài torch), load 10 weights
   fp16, infer TTA 8 ở độ phân giải đúng, ensemble, hậu xử lý ở 2048 (`post_ens_n1n2_hyst.json`),
   assert mask rời nhau, ghi CSV, in thời gian từng bước và tổng. Chạy trên **kernel Kaggle mới**
   với dataset riêng tư snapshot repo + dữ liệu cuộc thi; so CSV bằng tiêu chí § 3.2. Nộp 1 lần.
4. **Lọc FP**: quyết định giữ/bỏ theo kết quả cài sklearn trên Kaggle; nếu giữ, lưu thêm phiên bản
   không pickle (xuất cây sang JSON hoặc `skops`) hoặc chấp nhận ghim version và ghi rõ trong README.
5. **`requirements.txt`**: giữ bản ghim đầy đủ (môi trường Mac tạo CSV) + `requirements-kaggle.txt`
   tối thiểu; README ghi version torch thật trên kernel Kaggle đã chạy.
6. **Rubric N5** trên 2 final (Dice torchmetrics, phân bố IoU, 1:n / n:1) + thời gian infer → bảng
   trong report.
7. **Report 4 trang** (bản 1 ngày 10/11, bản cuối 13/11): pipeline, phân rã lỗi F52, inter-annotator
   0.34 (F6/S4), ablation OOF có CI, đoạn **"LB không phân giải"** (F8 + mô phỏng § 1.2 + Roelofs
   90 mẫu), giới hạn và hướng tương lai (S23). Mọi thành phần phải giải thích được (luật AI).
8. **Chọn 2 final trên Kaggle** (13/11), chụp màn hình; kiểm lại 14/11 và sáng 15/11.
9. **15/11 13:00 VN**: repo → public (Settings → Visibility); kiểm tra link ẩn danh (trình duyệt
   không đăng nhập) tới README, notebook, weights; nộp Google form **cùng lúc** (hạn form muộn hơn
   nhưng không có lý do chờ). Lưu bản PDF report vào repo (`docs/report/`).

### 3.4 Lỗi hay gặp 48 h cuối (từ forum Kaggle/fast.ai và từ chính repo)

- Notebook chạy được trên máy mình nhưng không trên kernel mới: thiếu file git-ignored (weights,
  `configs/*.pkl`), version lệch (`AttributeError` kiểu smp/timm), tải pretrained lúc infer cần
  internet, đường dẫn tuyệt đối `/Users/...`. Phòng: chạy từ `git clone` sạch, không từ working tree.
- Timeout/RAM: kernel Kaggle ~12 h CPU / 9 h GPU; 2048² × 10 model × TTA 8 giữ float32 trong RAM có
  thể vượt 13 GB RAM nếu giữ cả 180 ảnh → stream từng ảnh, cộng dồn uint8.
- Hết lượt nộp đúng ngày cần nộp lại: 15/11 reset 00:00 UTC rồi đóng 06:00 UTC → chỉ 6 giờ. Mọi bài
  "phải nộp" xong trước 14/11; 15/11 chỉ cho sự cố.
- Quên chọn final hoặc chọn nhầm bản cũ cùng tên file (nhiều `submission_ens_n1n2*.csv` 0.37).
- Repo public nhưng thiếu commit cuối (weights trong commit chưa push), hoặc GitHub từ chối push vì
  một file ≥ 100 MiB (fp32 81.7 MB qua được, nhưng BCE/N8 fp32 cũng 82 MB; dùng fp16 cho tất).
- Report: con số không khớp code/CSV (OOF in-sample so với cross-tune); dùng chữ "0.43 OOF" khi CSV
  nộp là 0.4284 cross-tune. Dùng đúng một bảng nguồn (`docs/EXPERIMENTS.md` sổ cái).

## 4. Câu 4 — Rà PLAN § 5b và đề nghị sửa

### 4.1 Nhận xét chung

- Ba mục đích của "5 lượt/ngày" trong § 5b: (1) hiệu ứng chỉ có trên test, (2) tái lập, (3) ablation
  cho report. Theo § 1: (1) gần như không đo được trừ bước ≥ +0.01 (duy nhất còn lại: "model đơn so
  với ensemble"); (3) phải làm trên OOF, cột LB chỉ nói "không phân giải"; (2) là việc thật nhưng xếp
  quá muộn (10–11/11) và tiêu chí "LB bằng" sai (cần, không đủ).
- Nguyên tắc "mỗi bài một giả thuyết" tốt, giữ. Thêm: **mỗi giả thuyết phải có P(LB thấy) > 0.5 theo
  bảng § 1.2**, nếu không thì không nộp.

### 4.2 Sửa từng ngày

| Ngày | PLAN § 5b hiện tại | Đề nghị | Lý do |
|---|---|---|---|
| 06/11 | 1. final 2 N1 UP2048 · 2–3. N1:N2 2:1, 1:2 · 4–5. min_area 120/300 | 1. final 2 N1 UP2048 (giữ, điểm LB cho report) · 2. **fp16 final 1** (sau khi đổi weights, § 3.3 mục 1–2) · 3–5. **không nộp** | 2:1/1:2 đã đo OOF ≈ 0/âm (ENS-W); min_area là knob hậu xử lý đã cạn (F56) và LB không thấy (P = 0.19 dưới null) |
| 07/11 | 1–3. trọng số N1:N2 · 4. seed 0.9 · 5. radius 6 | 1. **CSV từ notebook Kaggle (bản dataset)** · 2. 15 model BCE **chưa cần nộp lại** (đã có 0.37) · 3–5. trống | notebook phải chạy trước freeze để còn 2 ngày sửa; seed/radius là knob hậu xử lý |
| 08/11 | 1. ens 4 thành viên · 2. + TTA · 3. r = 0 · 4. cc · 5. không lọc FP | 1. ens có N8 (nếu N8 về và qua gate) · 2. ens có N8 **từ notebook** · 3. **final 1 không lọc FP** nếu quyết định bỏ lọc (§ 3.3 mục 4) · 4–5. trống | "+ TTA" vô nghĩa (map đã TTA 8); r = 0 và không lọc đã nộp 05/11 (0.37); cc không cần |
| 09/11 | 1. final 1 · 2. final 2 · 3–5. fold đơn 1, 2, 3 | giữ nguyên; final 2 = ứng viên § 2.2 (15 model hoặc N1 + N2 nếu final 1 có N8); fold đơn giữ (giá trị yếu, miễn phí) | |
| 10/11 | fp16 ×2, N2 một mình, BCE một mình, dự phòng | 1–2. **chỉ nếu** weights/notebook đổi sau 09/11 · N2/BCE một mình: **bỏ** (LB sẽ 0.36–0.37, không nói gì; OOF đã có 0.4129) · còn lại trống | đã làm 06–08/11 |
| 11/11 | notebook ×2, dự phòng | **`git clone` sạch → notebook → CSV** (lần cuối), 1 bài | đổi từ "lần đầu" thành "lần cuối" |
| 12–14/11 | ablation còn thiếu; chỉ nộp khi phát hiện lỗi | giữ; **13/11 chọn 2 final + chụp màn hình** | |
| 15/11 | – | 5 lượt chỉ cho sự cố; 13:00 VN repo public + form | reset 00:00 UTC, đóng 06:00 UTC |

Tổng số bài cần nộp từ 06/11: ~10–12, so với ~45 lượt còn lại. Lượt dư **không cần dùng**; "dùng hết
5 lượt/ngày" không phải mục tiêu.

### 4.3 Việc nên thêm vào PLAN (không tốn GPU)

1. `scripts/compare_submissions.py` (tiêu chí § 3.2) — nửa ngày Mac.
2. `scripts/export_fp16.py` + thư mục `weights/` tracked — 1 giờ; đo PQ(fp16|fp32) trên 707 OOF **và**
   trên test CSV.
3. `notebooks/final_inference.ipynb` + dataset Kaggle snapshot — 1 ngày; chạy thật trên kernel.
4. Quyết định lọc FP: giữ nếu cài được sklearn 1.9.1 trên kernel trong < 5 phút và PQ không đổi; bỏ
   nếu không (OOF −0.0018 in-sample, LB không đổi, bớt một pickle).
5. Đoạn report "LB không phân giải": F8, bảng § 1.2, Roelofs 90 mẫu, 9 bài 0.37. Biến điểm yếu
   (0.37) thành phân tích có bằng chứng — rubric 30% định tính.

## 5. Hướng đã kiểm và thấy không đáng làm

| Hướng | Bằng chứng | Kết luận |
|---|---|---|
| Dò phân bố annotator / lệch train–test qua LB | Blum–Hardt: làm tròn lớn ≈ Ladder, không rò; 3 ngưỡng P/R 05/11 đều 0.37; S7 host từ chối nói | chết |
| Xu hướng qua nhiều điểm trong một ngày | § 1.2: 3 biến thể +0.005 → P(≥ 1 hiện +0.01) 0.73 so với 0.47 dưới null | không tách được |
| Biến thể cố ý làm kém để đo độ nhạy LB | F9: đã có thang 0.21–0.37 với val tương ứng, Spearman 0.92 | đã có, không nộp thêm |
| Ablation hậu xử lý trên LB cho report | 05/11: r0, không lọc, t0.3/0.5 đều 0.37 | làm trên OOF |
| "LB bằng" làm tiêu chí tái lập | bảng § 1.2: regression −0.005 vẫn 0.37 với P ≈ 0.5 | thay bằng PQ(B|A) ≥ 0.995 |
| N1 đơn làm final 2 để "phòng hộ" | thua final 1 ở 5/5 fold, map là tập con → tương quan cao, E[max] ≈ final 1 | thay bằng bài khác nhiều hơn với OOF ≥ final 1 − 0.002 |
| Tin LB hơn OOF khi chọn final | Roelofs 2019 (public 90 mẫu là ngoại lệ pathological), Kaggle Book, F8 (±0.027), F9 | không |
| Chờ đến 10–11/11 mới làm weights/notebook | notebook chưa tồn tại, weights git-ignored, pickle sklearn, pin torch | dời lên 06–08/11 |

## 6. Nguồn

- Blum, Hardt. The Ladder: A Reliable Leaderboard for Machine Learning Competitions. ICML 2015.
  https://arxiv.org/abs/1502.04585 (đọc bản ar5iv 06/11).
- Roelofs, Shankar, Recht, Fridovich-Keil, Hardt, Miller, Schmidt. A Meta-Analysis of Overfitting in
  Machine Learning. NeurIPS 2019. https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning
- Banachewicz, Massaron. The Kaggle Book, ch. "Designing good validation" (tóm tắt):
  https://alibakh62.github.io/pages/content/book_summary/the_kaggle_book/designing_good_validation.html
- Becoming a Kaggle Master, lec 3 ("Trust your CV"):
  https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf
- Kaggle competition strategy (game-theory mô phỏng shake-up, ρ public–private):
  https://r-heller.github.io/equilibria/tutorials/public-apis-and-datasets/kaggle-competition-strategic/
- DataCamp, Winning a Kaggle Competition in Python (quy tắc 2 final, lấy điểm tốt hơn):
  https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/modeling?ex=12
- GitHub, About large files: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- PyTorch, Reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- HF model card ACR-instance-segmentation-RF-FP16 (dung sai fp16 cho mask):
  https://huggingface.co/cmes-deepvision/ACR-instance-segmentation-RF-FP16-v1.0.0
- Repo: `docs/RESEARCH.md` (F8, F9, F16, F52–F56, S6, S6b, S7, S8, S19, S21), `docs/EXPERIMENTS.md`
  (G4, UP2048 OOF, ENS-W, ENS-n1n2bce OOF), `docs/PROGRESS.md` (lịch sử submission), `PRIZE_RULES.md`,
  `docs/logs/lb_noise_effb4_hyst.txt`, `scripts/infer_probs.py`, `scripts/make_submission.py`,
  `kaggle/make_kernel.py`, `.gitignore`, `requirements.txt`.
