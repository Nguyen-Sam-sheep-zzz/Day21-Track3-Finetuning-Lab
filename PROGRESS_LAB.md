# Tiến độ Lab 21 — Nguyễn Nhân Sâm · 2A202602672

Ngày cập nhật: 07/10/2026 (Asia/Saigon).

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| Fork + clone | Hoàn thành | origin là Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab; upstream là VinUni-AI20k; branch feature/lab21-finetuning |
| Môi trường CPU | Hoàn thành | Python 3.12, requirements-cpu.txt; results/environment_cpu.txt |
| Smoke | Hoàn thành | 116 passed, 3 skipped; results/setup_smoke.txt |
| NB1 tokenizer T4 trên CPU | Hoàn thành cục bộ; chạy lại trên Colab | results/mask_proof.json, template_check.json, token_stats.json, nb1_local.txt |
| Chuẩn bị evidence NB2/NB5 | Code đã qua review; chờ đo GPU | raw baseline + manifest; paired qualitative; kiểm tra base và metadata training của adapter |
| Notebook Colab cá nhân | Đã tạo; chưa chạy GPU | colab/Lab21_NguyenNhanSam.ipynb, full eval, epochs=2 |
| NB2 baseline | Chưa chạy GPU | Không có baselines_frozen.json chính thức |
| NB3 correct | Chưa chạy GPU | Chưa có adapter/loss/VRAM |
| NB4 đối chứng | Chưa chạy GPU | Chưa có ba run |
| NB5 verdict | Chưa chạy GPU | Chưa có metric/verdict cá nhân |
| Report và gói nộp | Mới có phần NB1 | Chờ số liệu NB2–NB5 và phản tư cá nhân |

## NB1 đã đo

- Model tokenizer: unsloth/Qwen3.5-4B, tier T4, không tải full weights trên Windows.
- Corpus: 250 mẫu; train=225, val=25, seed=42.
- Mask sample: 39/94 token supervised (0.4149), answer_is_supervised=true, question_is_masked=true.
- Token lengths: p95=98, max=101, suggested_max_length=256; giữ giới hạn tier=1024, không cắt answer và dùng cùng giới hạn mọi run.
- Template test giữ reasoning trace thử nghiệm; corpus huấn luyện mặc định chỉ có answer JSON.
- check_mask_agreement.py báo tokenizer-level assistant mask rỗng vì template không có generation markers. Đây là lý do pipeline dùng labels đã build bởi labkit, không bật assistant_only_loss để TRL tự tạo mask. Không báo chẩn đoán này là PASS.

## Bước kế tiếp

Mở notebook Colab của fork, chọn T4, chạy Setup → smoke → NB1 → NB2 trước NB3. Giữ checkpoint baseline trước training. Sau NB5 tải checkpoint core_results về thư mục Lab 21 để hoàn thiện report và nghiệm thu.

Browser automation của phiên Codex hiện không khởi tạo được. Chưa có kết nối tới GPU Colab hoặc trạng thái tài khoản Google; người dùng cần chạy các ô GPU nếu công cụ vẫn lỗi. Không cần cung cấp mật khẩu/API key cho pipeline mặc định.

Kiểm tra bổ sung: bộ gốc + student_tests sau sửa review có 175 passed, 3 skipped; điểm provenance đã được reviewer xác nhận ADDRESSED. JSONL đã được khôi phục byte-exact từ Git blob upstream và .gitattributes giữ LF, tất cả checksum gốc khớp. Chưa đổi nội dung dataset/checksum/scorer/test gốc.
