# Đối chiếu rubric và trạng thái bàn giao core

Nguyễn Nhân Sâm · 2A202602672 · thí nghiệm 07/10/2026.

**Số liệu đã kiểm tra; verdict FAILED được phân tích đúng. Chưa đủ mọi mục rubric: 3.4 chưa có hai ca FT thua; 4.4 cần học viên đọc và xác nhận phản tư.** Không cam kết tổng điểm hoặc kết quả chấm.

| Mục | Trạng thái bằng chứng | File hoặc nhận xét |
|---|---|---|
| 1.1 Mask | Đủ | Hai assert true, supervised fraction 39/94 < 0,95 |
| 1.2 Template | Đủ | Template giữ trace; report phân biệt với corpus JSON không có trace |
| 1.3 p95 | Có giải thích | p95=98, suggested=256, max=101; giữ tier cap=1024, không truncate |
| 1.4 Adapter correct | Đủ | Hai file adapter; CSV loss/VRAM; header, hash và finite đã kiểm tra |
| 2.1 Ngân sách params | Đủ | Attn_only rank=283; lệch 0,025233%, dưới 5% |
| 2.2 Step | Đủ cấu hình và log | Bốn max_steps=30; chưa thống kê skipped updates fp16 |
| 2.3 Một yếu tố | Có thiết kế và phân tích | Placement kèm rank/alpha bù budget; LR ÷10; base 4-bit; mask đối chứng đối chiếu source/log |
| 2.4 Placement/rank | Đủ phân tích | Correct và attn_only đều 0,97; không xếp hạng bằng loss |
| 3.1 Baseline | Đủ bằng chứng nhất quán | b=0,765 > a=0; đo trước training; baseline bất biến qua bốn checkpoint |
| 3.2 Bốn nhóm | Đủ aggregate | Target, regression, format, latency; NB5 chưa lưu raw FT regression |
| 3.3 Verdict | Đủ | FAILED vì regression giảm 0,268889 vượt tolerance 0,02; gate giữ nguyên |
| 3.4 Định tính | CHƯA ĐỦ | Bảy ví dụ thật; 33 thắng, 17 hòa, 0 thua; không bịa hai ca thua |
| 4.1 Cấu trúc | Đã biên tập | Model/dataset, lý do, mask, baseline, đối chứng, verdict, phản tư |
| 4.2 Kết luận | Đủ độ dài | 386 từ theo whitespace; có lập luận và giới hạn |
| 4.3 Số liệu | Đã kiểm tra và review | Raw baseline/cặp target re-score; bảng và bảy ví dụ khớp; aggregate regression đối chiếu log |
| 4.4 Phản tư | Học viên cần xác nhận | Soạn từ thao tác và kết quả thật; AI hỗ trợ được khai báo |
| B1 merge/hot-swap | Chuẩn bị mã, chưa chạy GPU | Helper và notebook bổ sung; không yêu cầu điểm thưởng |
| B2 dataset riêng | Có bản nháp synthetic riêng | Chưa xác nhận chất lượng thủ công; không tự nhận hoàn thành |
| B3 traces | Chưa thực hiện | Cần dữ liệu trace và hai training run riêng; core JSON không đủ |
| B4 rank sweep | Chuẩn bị mã, chưa chạy GPU | r8/r16/r64 cùng recipe; chưa có bảng measured |
| B5 Hub | Chuẩn bị local, chưa upload | Cần tài khoản đích và login; xem HF_PREPARATION.md |

Verify local: 26 đạt, 1 cảnh báo, 0 lỗi, exit 0; cảnh báo model FAILED. Colab smoke: 119 original tests đạt. Local: 116 original tests đạt, 3 skip do chưa có PyTorch; 59 student tests đạt. Tests gốc, scorer, config và corpus không đổi. Verify không chấm đủ rubric 3.4.

Gói Option A chứa submission/REPORT.md, READINESS.md, toàn bộ results, hai file adapter correct và notebooks/*.py. Gói tối giản không cam kết tự chạy verify độc lập; checkout đầy đủ giữ scripts/src/data để kiểm tra. Không có full weights, .env, .venv, cache hoặc adapter đối chứng. Adapter correct F32 có dung lượng thô 129,93 MB; ZIP khoảng 120 MB, lớn hơn ví dụ 5–15 MB trong rubric.

Nếu bổ sung ví dụ regression, dùng notebook local output/Lab21_Regression_Followup.ipynb, upload ZIP after_correct gốc và tải regression_followup.json. Notebook này chưa chạy GPU; đây là phép đo bổ sung. Không thay baseline/verdict core hoặc gọi regression loss thành target loss. Cần giảng viên xác nhận cách áp dụng 3.4 khi tập target thực có 0 ca thua.

Học viên trực tiếp nộp LMS hoặc đường dẫn theo yêu cầu lớp. Gói local và remote branch không đồng nghĩa đã nộp bài.

Notebook bổ sung: https://colab.research.google.com/github/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab/blob/feature/lab21-finetuning/colab/Lab21_Deadline_Followup.ipynb

Đã nhận ba ZIP bổ sung ngày07/10 lúc23:49. Cả ba status=incomplete do bắt đầu sau
mốc dừng23:20; chỉ có log/status, chưa thực hiện regression/B1/B4. Đã kiểm tra và lưu
nguyên trạng trong results/followup_*. Core không đổi; rubric3.4, bonus và phản tư còn
giới hạn đã công bố. Không tuyên bố ZIP được tạo nghĩa là phép đo đã hoàn thành.
