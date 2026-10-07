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
| Report | Đã biên tập và review số liệu | Bảy ví dụ thật, kết luận 386 từ; học viên cần xác nhận phản tư |
| Verify | Hoàn thành | 26 đạt, 1 cảnh báo, 0 lỗi; 59 student tests đạt; original CPU 116 đạt, 3 skip |
| Rubric 3.4 | Chưa đủ bằng chứng | Target: 33 thắng, 17 hòa, 0 thua; chưa có hai ca thua định tính |
| Gói core | Đã kiểm tra | output/lab21_2A202602672_submission_core.zip, 120,17 MB; hash/CRC, cấu trúc và loại trừ đều đạt |
| B1–B5 | Chưa làm | Không yêu cầu điểm thưởng khi chưa có measurement |
| Nộp LMS | Chưa xác nhận | Local ZIP và GitHub không chứng minh đã nộp chính thức |

Runtime cũ đã ngắt. Bốn adapter còn ở adapters/{correct,attn_only,wrong_lr,qlora}; không cần train lại để đọc core. ZIP gốc giữ nguyên trong Downloads; SHA256 và thứ tự checkpoint nằm ở results/received_checkpoints.json.

Model FAILED là kết quả hợp lệ; không đổi gate, eval hoặc prompt để chuyển thành PASS. Trọng số đã lưu đều finite, nhưng grad_norm=nan trong log cần được công bố: chưa có thống kê skipped updates fp16. Model/tokenizer revision null nên không khẳng định đã pin phiên bản lịch sử trên Hub.
