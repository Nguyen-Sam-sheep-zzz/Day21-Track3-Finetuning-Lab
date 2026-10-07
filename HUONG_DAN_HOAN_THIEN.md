# Hoàn thiện Lab21 sau các checkpoint đã có

Core1–7 đã có kết quả thật. Ba ZIP supplementary23:49 ngày07/10 bị guard23:20
bỏ qua; ZIP có status/log nhưng chưa có kết quả. Notebook đã sửa lỗi này.

1. Mở lại notebook từ GitHub (bản Drive cũ không tự cập nhật):
https://colab.research.google.com/github/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab/blob/feature/lab21-finetuning/colab/Lab21_Deadline_Followup.ipynb
2. Chọn T4, Save a copy in Drive nếu muốn; chạy năm ô code chuẩn bị từ trên xuống.
3. Ô upload chọn ZIP core_results_20261007T124407Z gốc trong Downloads (~478MB).
4. Chạy run_stage('regression'), đợi dòng HOÀN TẤT và tải ZIP. Gửi ZIP ngay để chọn
   ca FT thua regression nếu có; ghi supplementary repeat, không gọi thành target loss.
5. Chạy run_stage('b1'), tải ZIP; yêu cầu đủ50trước/sau merge, passed=true và cùngbase
   có hai adapter. Merge measured mà gatefail không được gọi B1đạt.
6. Chạy run_stage('b4'), tải ZIP; yêu cầu ba rank8/16/64 đều đủ50outputs, statuscomplete.
7. Gửi ba ZIP mới để audit/report/package. Sâm đọc và chỉnh phần phản tư theo trải nghiệm
   thật. Không bịa hai ca thua khi dữ liệu không có, không đổi frozen baseline/core.

Cửa sổ mặc định2giờ từ ô khai báo hàm, không theo hạn ngày07/10. Đổi RUN_WINDOW_HOURS
trước khi khai báo nếu cần. Mỗi setup tạo attempt mới; không chạy lại setup khi stage
đang hoạt động. Nếu ô đỏ, giữ ZIP và dừng bước phụ thuộc; không coi tảiZIP là thành công.

Core rubric còn3.4(≥2caseFTthua cóoutput) và học viên xác nhận phản tư4.4.
Regression có thể cung cấp ví dụ thuộc nhómregression; việc áp dụng3.4 cần diễn giải
đúng rubric, không cam kết điểm khi chưa có output và chấm của giảng viên.

Các bonus khác: B2đã có240train60eval vàCUSTOM_DATASET.md nhưng cần ràsoát chấtlượng;
rubricB2không bắt buộctrainriêng. B3cần dữliệutrace,2runmask vàvalid_trace_rate; chưa có
notebook hoàn chỉnh. B5cầnupload3file trongoutput/huggingface_adapter lênmodelrepoPublic
vàgửiURL; không gửi token/password. Notebook này không chứng minh hoàn thành cảB1–B5.
