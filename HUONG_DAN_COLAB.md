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
