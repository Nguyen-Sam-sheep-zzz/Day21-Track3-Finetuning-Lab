# Tiến độ Lab 21 — Nguyễn Nhân Sâm · 2A202602672

Cập nhật theo bốn checkpoint Colab ngày 07/10/2026. Commit thực nghiệm: `90679f91842e9f42c9192b9c5eab316935705e9c`.

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| Fork và chuẩn bị | Hoàn thành | Nhánh feature/lab21-finetuning; 38 file mã runtime khớp commit |
| NB1 | Hoàn thành trên Colab | Mask 39/94; hai bool true; p95=98, max=101; split 225/25 |
| NB2 | Hoàn thành | Full 50/15; b=0,765 > a=0; baseline giống hệt bốn checkpoint; trước training chưa có adapter |
| NB3 | Hoàn thành | Correct: 32.464.896 tham số; 30 step; loss 0,6256; 8,78 GB; 416,3 giây |
| NB4 | Hoàn thành | Ba đối chứng đủ; cùng max_steps=30; attn_only rank=283, lệch tham số 0,025233% |
| NB5 | Hoàn thành | Target 0,97; regression 0,522222; format 1; latency 1355,8 ms; model FAILED |
| Report | Đã cập nhật core và regression repeat | 10ví dụ thật gồm3regression thua; kết luận386từ; học viên cần xác nhận phản tư |
| Verify | Hoàn thành | 26 đạt, 1 cảnh báo, 0 lỗi; 59 student tests đạt; original CPU 116 đạt, 3 skip |
| Rubric 3.4 | Có ví dụ bổ sung đúng nguồn | 3ca regression repeat thua; target vẫn33thắng17hòa0thua; giảng viên chấm |
| Gói core | Đã kiểm tra | output/lab21_2A202602672_submission_core.zip, kèm regression repeat; hash/CRC, cấu trúc và loại trừ đều đạt |
| B1/B4 | Mã và notebook bổ sung đang kiểm tra | Chưa có measurement GPU; không nhận điểm |
| B2 | Soạn draft synthetic riêng | Chất lượng người đọc chưa xác nhận |
| B3 | Chưa chạy | Cần trace và hai run riêng |
| B5 | Đã chuẩn bị model card + adapter local | Chưa upload vì thiếu account/login |
| Nộp LMS | Chưa xác nhận | Local ZIP và GitHub không chứng minh đã nộp chính thức |

Runtime cũ đã ngắt. Bốn adapter còn ở adapters/{correct,attn_only,wrong_lr,qlora}; không cần train lại để đọc core. ZIP gốc giữ nguyên trong Downloads; SHA256 và thứ tự checkpoint nằm ở results/received_checkpoints.json.

Model FAILED là kết quả hợp lệ; không đổi gate, eval hoặc prompt để chuyển thành PASS. Trọng số đã lưu đều finite, nhưng grad_norm=nan trong log cần được công bố: chưa có thống kê skipped updates fp16. Model/tokenizer revision null nên không khẳng định đã pin phiên bản lịch sử trên Hub.

## Tiến độ theo yêu cầu trước 23:30

Cập nhật UTC: 2026-10-07T15:36:18.182144+00:00. Notebook bổ sung khôi phục đúng ZIP core, giữ immutable baseline và xuất ZIP nhỏ từng bước.
Công cụ điều khiển Colab lỗi khởi tạo; GPU cục bộ GTX1650 4GB không phù hợp recipe T4 4B.
Đang cần thao tác mở/chạy notebook T4 của học viên để có các số đo mới.
Gói core đã có và được bảo toàn; không sửa gate để biến FAILED thành PASS.

Kiểm tra mã bổ sung cuối: 201 passed, 3 skipped (results/supplementary_tests_cpu.txt). Verify UTF-8: 26 đạt, 1 cảnh báo, 0 lỗi. Không có kết quả GPU bổ sung tại thời điểm ghi này.

Đã nhận ba ZIP bổ sung ngày07/10 lúc23:49. Cả ba status=incomplete do bắt đầu sau
mốc dừng23:20; chỉ có log/status, chưa thực hiện regression/B1/B4. Đã kiểm tra và lưu
nguyên trạng trong results/followup_*. Core không đổi; rubric3.4, bonus và phản tư còn
giới hạn đã công bố. Không tuyên bố ZIP được tạo nghĩa là phép đo đã hoàn thành.

Cập nhật08/10: notebook supplementary đã sửa mốc dừng cố định thành cửa sổ2giờ và kiểm tra file thực tế trước khi báo HOÀN TẤT. 211 CPUtests đạt,3skip; chưa có measurement mới. Chạy theo HUONG_DAN_HOAN_THIEN.md.

Cập nhật sau checkpoint01:21ngày08/10: regression complete đủ15câu; re-score7loss/1win/7tie, metric0,522222 khớp core. Report đã thêm3ví dụ thua, tổng10ví dụ. B1/B4chờ đo; phản tư4.4vànộpLMS cần học viên thực hiện. Các đoạn incomplete/phục hồi phía trên là lịch sử trước checkpoint mới.
