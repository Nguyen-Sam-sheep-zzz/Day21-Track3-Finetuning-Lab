# Lab 21 — Đánh giá fine-tuning ticket CSKH tiếng Việt

**Họ tên:** Nguyễn Nhân Sâm
**MSSV:** 2A202602672
**Ngày chuẩn bị:** 07/10/2026
**Repo:** https://github.com/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab
**Nhánh:** feature/lab21-finetuning

> Trạng thái: đã hoàn thành kiểm tra CPU và NB1 bằng tokenizer model T4. Chưa chạy NB2–NB5 trên Colab; chưa có kết quả chất lượng hay training cá nhân. Đây là report đang thực hiện, chưa phải bản nộp cuối.

## 1. Bài toán và lựa chọn thí nghiệm

Bài toán là phân loại ticket chăm sóc khách hàng tiếng Việt thành JSON với bốn trường intent, urgency, product và sentiment. Dùng corpus mặc định để hoàn thành một phép so sánh rõ ràng trước khi mở rộng dữ liệu miền riêng. Corpus gồm 250 mẫu ban đầu, chia 225 train và 25 validation bằng seed 42; target eval có 50 ticket và regression eval có 15 instruction phổ thông.

Base được chọn là unsloth/Qwen3.5-4B trong tier T4, phù hợp đường chạy Colab Free T4 mà lab cung cấp. Baseline và adapter sẽ dùng cùng base. Cấu hình chính thức dự kiến là assistant-only, epochs=2, r=16, alpha=32, LR=1e-4 cho correct; batch hiệu dụng 16. GPU, precision, số step thực, loss, VRAM và thời gian sẽ được ghi từ lần chạy Colab, không lấy số demo trong tài liệu làm kết quả cá nhân.

## 2. Bằng chứng mask và template đã đo cục bộ

Nguồn: results/mask_proof.json, results/template_check.json, results/token_stats.json và results/nb1_local.txt.

| Nội dung | Kết quả |
|---|---:|
| Tổng token trong mẫu proof | 94 |
| Token supervised | 39 |
| Supervised fraction | 0.4149 |
| Answer nằm trong loss | true |
| Question được che khỏi loss | true |
| p95 toàn corpus | 98 |
| Token dài nhất | 101 |
| Suggested max length | 256 |
| Giới hạn tier được giữ | 1024 |

Tokenizer test bảo toàn trace thử nghiệm trong khối think. Đoạn supervised của mẫu proof có phần đóng think, JSON đáp án và EOS; không chứa câu hỏi. Đối chứng everything tính loss trên 100% token, gồm prompt, cho thấy vì sao không dùng chế độ đó cho run chính.

Giữ max_length=1024 là giữ giới hạn trên của cấu hình tier cho tất cả run. P95 gợi ý 256 và mọi mẫu đo được đều ngắn hơn cả hai giới hạn, nên lựa chọn 1024 không cắt mất answer. Đây là lựa chọn có giải thích theo rubric; cần chạy lại NB1 trong môi trường Colab và đối chiếu số thực trước khi chốt report.

Chẩn đoán check_mask_agreement cho thấy template không có generation markers; mask assistant do tokenizer trả về rỗng. Pipeline của lab tránh đường này bằng cách tạo input_ids và labels từ labkit.data theo mask đã proof, thay vì bật assistant_only_loss để thư viện suy diễn mask. Chẩn đoán đó không được báo thành một lần test PASS. Môi trường CPU chưa cài TRL hoặc PyTorch; test thực của SFTTrainer vẫn chờ GPU stack.

## 3. Baseline đóng băng — chưa đo

| Phương án | Target | Regression | Format | Latency |
|---|---|---|---|---|
| (a) Base + prompt đơn giản | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| (b) Base + prompt tối ưu | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| (c) LoRA + prompt ngắn | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

NB2 sẽ chạy trước training, giữ raw prediction từng mẫu và manifest. Không có số baseline để đưa ra kết luận tại thời điểm này.

## 4. Thí nghiệm đối chứng — chưa chạy

Correct là run tham chiếu. Attn_only đổi vị trí, rank được matched để ngân sách tham số lệch dưới 5%. Wrong_lr giảm learning rate 10 lần. QLoRA dùng base 4-bit. Bốn run phải có cùng max_steps. Bảng xếp hạng sẽ dùng target trong autopsy.json, kèm training loss để phân tích khả năng sai khác giữa chỉ số huấn luyện và năng lực tác vụ.

## 5. Phán quyết và ví dụ định tính — chờ NB5

Chưa có verdict. Chưa có ca FT thắng hoặc thua đã đo. Sau NB5 sẽ chọn ít nhất năm ví dụ từ output thật, trong đó ít nhất hai ca FT thua baseline b; mỗi ví dụ có ticket, nhãn, hai output và điểm hai bên. Không suy ra thua baseline chỉ vì FT trả lời sai.

## 6. Kết luận cuối và phản tư — chưa hoàn thành

Kết luận ít nhất 150 từ và phản tư cá nhân sẽ viết sau khi có bằng chứng NB2–NB5. Hiện chỉ kết luận được rằng mask của mẫu NB1 loại câu hỏi khỏi loss và giữ câu trả lời; chưa thể khẳng định fine-tuning hiệu quả, không quên năng lực chung, hay đáng triển khai.

## 7. Kiểm tra và điểm thưởng

Smoke gốc: 116 passed, 3 skipped, 0 failures; nguồn results/setup_smoke.txt. Bộ gốc và student_tests sau bổ sung evidence: 175 passed, 3 skipped; nguồn results/prepared_tests.txt. Các skip thuộc môi trường CPU chưa có PyTorch, không chứng minh pipeline GPU đã chạy. B1–B5 chưa hoàn thành. Chưa chạy verify nộp bài đầy đủ, chưa có ZIP nộp cuối hoặc xác nhận nộp LMS.
