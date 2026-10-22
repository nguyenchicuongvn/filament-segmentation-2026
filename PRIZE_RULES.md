# Solar Filament Segmentation Challenge 2026 — Điều kiện nhận giải

Nguồn: https://www.kaggle.com/competitions/filament-segmentation-2026 (Overview + Rules, đọc ngày 2026-10-19)

## Tóm tắt nhanh
- Host: ESAIR Lab; Sponsor: NSF National Solar Observatory (NSO/AURA). Community competition, **không có points/medals**.
- Tổng giải: **tối đa $3,000**, chia cho top-3 theo công thức `prize_i = 3000 * score_i^2 / sum(score_j^2)` (score = điểm rubric của ban tổ chức, không phải chỉ LB).
- Deadline Kaggle submission: **2026-11-15 06:00 UTC** (13:00 giờ VN). Deadline nộp report + code (Google form): **Nov 15, 11:59 pm CDT** — nên nộp cùng lúc cho an toàn.
- Công bố: Nov 30, 2026; trao tại IEEE BigData 2026 (Dec 14–17, Phoenix — tham dự là optional).
- Tài khoản đã join competition (`userHasEntered = True`).

## Xếp hạng KHÔNG chỉ dựa vào leaderboard
Rubric chấm cuối cùng:
- **Định lượng (70%)**: Panoptic Quality (metric LB), phân phối Dice/IoU, phân phối quan hệ one-to-many / many-to-one (phân mảnh / gộp nhầm), hiệu năng tính toán end-to-end.
- **Định tính (30%)**: mô tả pipeline chi tiết, hình thái mask dự đoán trên ảnh H-Alpha, **chất lượng code** (modularity, documentation).
- Host lưu ý: nhiều người "hack" LB; PQ > 0.35 đã có giá trị với họ.

## Bắt buộc để được xét giải (Final Submission qua Google form)
1. **Technical report 4 trang** (PDF, dùng template Overleaf của host): preprocessing, feature engineering, kiến trúc, training, post-processing… đủ để tái lập.
2. **Git repo public** (GitHub/GitLab/Bitbucket), không cần xin quyền truy cập:
   - `requirements.txt` có đủ package + version.
   - Một **Jupyter notebook** minh hoạ toàn bộ pipeline.
   - Không cần file nào ngoài repo để reproduce prediction.
   - Repo phải public **ngay khi competition đóng** và giữ public ít nhất đến khi công bố winners.
3. Thông tin form phải chính xác; thiếu/sai khiến ban tổ chức không truy cập được → bị loại.

## Điều kiện khi thắng
- Giao code đầy đủ (training + inference + mô tả môi trường) chạy ra đúng submission.
- License: winning code dưới **OSI-approved license không hạn chế thương mại** (vd MIT/Apache-2.0); cấp cho organizers license vĩnh viễn, non-exclusive để dùng cho nghiên cứu/benchmark.
- Ký giấy tờ nhận giải trong **2 tuần**; phản hồi email thông báo trong **1 tuần**.
- Thuế: nộp **W-8BEN** (người ngoài Mỹ); thuế do người thắng chịu.
- **Phải nhận được chuyển khoản từ ngân hàng Mỹ** và hoàn tất vendor onboarding với AURA, nếu không → mất giải.
- Team: giải chia đều cho các thành viên đã verify.

## Eligibility
- ≥ 18 tuổi, 1 tài khoản Kaggle duy nhất; không cư trú tại Crimea/DNR/LNR, Cuba, Iran, North Korea; không bị US sanctions. (Việt Nam: OK.)
- Không đủ điều kiện nhận giải nếu được ban tổ chức (Ahmadzadeh, Kempton, Qin Li, Pevtsov) hướng dẫn/giám sát hoặc có conflict of interest — vẫn được thi, nhưng giải chuyển cho người kế tiếp.

## Ràng buộc kỹ thuật / luật chơi
- Team tối đa 5; 5 submissions/ngày; chọn 2 final submissions.
- External data/pretrained models được phép nếu public, miễn phí/chi phí hợp lý. Nhưng: **inference chỉ dùng ảnh H-alpha trong thư mục test**, và **không dùng ground-truth metadata khác** (bbox, spine, class label của MAGFiLO) để train.
- Không hand-label / dự đoán thủ công dữ liệu test.
- Open source code dùng trong model phải có license OSI cho phép thương mại.
- Không chia sẻ code riêng ngoài team; chia sẻ công khai thì phải đăng trên Kaggle.
- Data chỉ dùng cho mục đích phi thương mại / nghiên cứu.
- **Dùng AI/LLM để viết code được phép**, nhưng người thắng phải hiểu, giải thích, bảo vệ và tái lập mọi thành phần chính; dựa quá nhiều vào code AI mà không giải thích được → có thể bị loại.

## Submission format
- CSV: `filament_id,segmentation_rle` — mỗi dòng 1 filament, id dạng `<image_id>_<k>`.
- RLE counts kiểu COCO (pycocotools), ảnh 2048×2048, không cần lưu size, không có dấu nháy.
- Self-eval notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
