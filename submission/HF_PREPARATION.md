# Chuẩn bị B5 — chưa upload

Thư mục local `output/huggingface_adapter/` có model card và hai file adapter correct đã kiểm tra hash.
Model card công bố verdict FAILED, mức regression suy giảm và giới hạn tái lập.
Không có token trong thư mục này. B5 chỉ hoàn thành khi Hub công khai và link được kiểm chứng.

Cần username/tài khoản Hugging Face của Sâm và đăng nhập trên máy bằng `hf auth login`.
Không gửi token hoặc mật khẩu qua chat. Sau khi đăng nhập, upload riêng ba file trong
thư mục này vào model repository của chính tài khoản đã xác nhận; không upload toàn bộ repo lab.
Trợ lý chưa thực hiện upload vì chưa có tài khoản đích và credential tại thời điểm chuẩn bị.
