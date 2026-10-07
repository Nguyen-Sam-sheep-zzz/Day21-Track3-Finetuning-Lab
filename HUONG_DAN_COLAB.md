> **Cập nhật08/10:** notebook followup đã bỏ mốc23:20 cố định và dùng cửa sổ2giờ. Xem HUONG_DAN_HOAN_THIEN.md; phần mô tả hạn07/10 dưới đây là lịch sử.

> Core NB1–NB5 đã chạy ngày 07/10/2026 và kết quả được giữ trong repo. Hướng dẫn bên dưới mô tả lần chạy gốc; không chạy lại NB2 trên thí nghiệm đã train. Thí nghiệm gốc dùng commit90679f9. Report và READINESS hiện ghi đầy đủ trạng thái/thiếu hụt.

# Chạy Lab 21 trên Colab — Nguyễn Nhân Sâm · 2A202602672

Repo: https://github.com/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab
Nhánh làm bài: feature/lab21-finetuning.

Mở notebook:
https://colab.research.google.com/github/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab/blob/feature/lab21-finetuning/colab/Lab21_NguyenNhanSam.ipynb

1. Đăng nhập Google trong trình duyệt của bạn; không gửi mật khẩu hoặc token cho trợ lý.
2. Runtime → Change runtime type → T4 GPU. Chỉ giữ một phiên GPU free hoạt động.
3. Chạy ô Setup và ô hàm chạy/smoke. Setup lấy code từ fork cá nhân, không dùng repo gốc cho các bổ sung evidence.
4. Chạy NB1 rồi NB2. NB2 phải có full eval 50 target + 15 regression, baseline b>a, raw prediction và manifest. Tải ZIP before_training được notebook tạo.
5. Chạy NB3; tải ZIP after_correct để giữ adapter.
6. Chạy NB4; kiểm tra bốn run cùng max_steps và ngân sách attn_only lệch <5%; tải checkpoint.
7. Chạy NB5; PASS/FAIL đều được giữ nguyên. Tải ZIP core_results về C:\Users\SAM\IdeaProjects\VinUniAIThucChien\Lap 21. Giữ ZIP gốc để đối chiếu.
8. Báo lại tên file ZIP cho trợ lý để đọc kết quả, viết report theo số thực và kiểm tra toàn bộ rubric. ZIP hiện là checkpoint, chưa phải bài nộp hoàn chỉnh.

Cấu hình chính thức: T4, unsloth/Qwen3.5-4B, assistant-only, EPOCHS=2, không EVAL_LIMIT. Không đổi cấu hình giữa các run. Đừng chạy lại NB2 sau training; nếu baseline cần đo lại, phải là một thí nghiệm riêng hoặc iteration trước train được ghi rõ.

Nếu Colab báo lỗi, dừng ở ô đó; notebook giữ log tại results/colab_<stage>.txt và tạo checkpoint có nhãn failed. Tải checkpoint/log để chẩn đoán. Không chạy tiếp các bước phụ thuộc sau ô lỗi. Không reconnect rồi đổi code giữa thí nghiệm đã frozen.

Nếu NB4 đứt giữa chừng mà runtime còn dữ liệu: chạy lại ô NB4, các adapter đã xong sẽ được skip. Nếu runtime đã mất: cần phục hồi checkpoint, đúng commit/config, và chạy lại NB1 để tái tạo split; nhờ trợ lý hướng dẫn theo file đã lưu, không lấy adapter từ smoke hay thí nghiệm khác.

B1 merge/hot-swap có ô tùy chọn, mặc định chưa chạy. B2/B3/B4/B5 sẽ làm trong các chặng riêng sau core. Full merged weights không nằm trong ZIP checkpoint; adapter chính và đối chứng có trong ZIP để phục hồi.

## Phục hồi để lưu thêm output regression

Không cần huấn luyện lại. Mở Colab → File → Upload notebook, chọn file local output/Lab21_Regression_Followup.ipynb rồi chọn T4. Khi ô upload yêu cầu, chọn đúng lab21_2A202602672_after_correct_20261007T112307Z.zip trong Downloads (~120MB). Notebook kiểm tra SHA256, phục hồi đúng source commit cũ và adapter, sinh thêm15câu regression cùng recipe và tải regression_followup.json. Đây là lần đo bổ sung chưa chạy GPU; kết quả không thay thế raw output gốc của NB5. Gửi fileJSON này để phân tích tiếp. Không dùng nó để tuyên bố có hai ca thua trong50ticket target hoặc bảo đảm đạt3.4 khi giảng viên chưa xácnhận.

## Notebook bổ sung cho hạn 23:30 ngày 07/10/2026

[Mở notebook khôi phục và đo bổ sung](https://colab.research.google.com/github/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab/blob/feature/lab21-finetuning/colab/Lab21_Deadline_Followup.ipynb).

1. Nếu phiên cũ đang Run all, dùng Runtime → Interrupt execution; giữ các ZIP đã tải.
2. Mở notebook bổ sung trên T4; chạy các ô setup/restore theo thứ tự.
3. Khi được yêu cầu, chọn **lab21_2A202602672_core_results_20261007T124407Z.zip** trong Downloads (~478 MB).
4. Chạy regression trước; tải ZIP nhỏ sau bước đó. Sau đó chạy B1 và B4 nếu còn giờ.
5. Mỗi bước tải ZIP bằng chứng riêng. Giữ tất cả và gửi lại để kiểm tra trước khi sửa report.

Nguồn core giữ commit90679f9; helper được pin commit và hash riêng. Không chạy lại NB2/NB3/NB4 core.
Notebook chỉ kiểm tra thời gian trước từng bước; mốc23:20 dành cho tổng hợp, không ngắt CUDA giữa chừng.
Kết quả bổ sung chưa được khẳng định trước khi chạy GPU. Lỗi hoặc partial giữ nguyên log/status.
Không bấm lại bước đã tạo bằng chứng; dùng một thư mục/attempt mới nếu thực sự cần repeat.
