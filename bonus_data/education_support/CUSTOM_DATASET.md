# Bộ dữ liệu CSKH giáo dục AI giả lập — bản chuẩn bị B2

**Trạng thái:** đã tạo và kiểm tra cơ học dữ liệu; **chưa xác nhận hoàn thành B2**. Miền dữ liệu chưa được người dùng phê duyệt, chất lượng chưa được người dùng/chuyên gia thẩm định độc lập và chưa chạy thí nghiệm GPU trên bộ này.

## Nguồn và cách thu thập

Dữ liệu được AI hỗ trợ biên soạn mới cho bài Lab 21 ngày 07/10/2026, từ danh sách tình huống dịch vụ học AI giả lập trong `build_dataset.py`. Không cào web, không nhập ticket khách hàng thật, không sử dụng dữ liệu cá nhân. Đây **không phải 200 mẫu do con người thu thập và gán nhãn độc lập**.

Miền gồm đăng ký khóa học AI, đổi gói học, học phí, phòng lab trực tuyến, hỗ trợ mentor và giao học liệu vật lý. Các tên Khóa MâyTre AI Cơ Bản, Gói SenĐá AI Thực Hành, Lộ Trình GióLúa AI, Gói ĐomĐóm AI Mentor cùng bốn sản phẩm eval đều hư cấu. Những phát biểu về quyền học, phí, gia hạn, dùng thử hoặc điều kiện hủy chỉ là tình huống giả định của ticket; không phải chính sách thật của VinUni, VLearn hay tổ chức nào.

Tác giả AI viết 75 nhóm tình huống, mỗi nhóm có bốn chi tiết khác nhau. Bộ tạo chỉ ghép những tình huống đã viết với tên sản phẩm và dấu hiệu nhãn rõ ràng; không gọi mô hình hoặc mạng khi chạy. Tên sản phẩm được xoay vòng theo nhóm để tránh một tên sản phẩm quyết định hoàn toàn urgency/sentiment.

## Cấu trúc và chia tập

| Tập | Số ticket | Nhóm tình huống | Ticket mỗi intent |
|---|---:|---:|---:|
| `train.jsonl` | 240 | 60 | 48 |
| `eval_target.jsonl` | 60 | 15 | 12 |

Mỗi dòng có `instruction`, `input`, `output` và `label`, giống định dạng core. `output` là chuỗi JSON; `label` là object cùng nội dung và **đúng bốn khóa** `intent, urgency, product, sentiment`. Instruction được lấy nguyên văn từ train seed; không sửa core.

Train và eval tách theo nhóm tình huống, không chia ngẫu nhiên các biến thể cùng nhóm. Eval dùng các nhóm riêng: đổi mục tiêu/nhịp học/môi trường thực hành; thanh toán vượt biên nhận; lỗi trợ năng/máy chấm/thông báo; kiện học liệu bị giữ/tủ nhận/tuyến hoàn; hỏi quyền chia sẻ đồ án/học nhóm/ngoại tuyến. Bốn tên sản phẩm eval cũng khác train.

## Quy tắc gán nhãn

| Intent | Ý nghĩa trong miền giả lập |
|---|---|
| `doi_tra` | Đổi/trả gói để nhận gói học thay thế; không yêu cầu hoàn tiền. |
| `hoan_tien` | Yêu cầu lấy lại tiền hoặc xử lý khoản hoàn tiền. |
| `san_pham_loi` | Sự cố cụ thể trên nền tảng: phát video, nộp bài, đăng nhập, notebook, trợ năng... |
| `van_chuyen` | Giao **vật phẩm vật lý**: sách giấy, vở bài tập, thẻ in, bộ linh kiện. Không gán cho liên kết kích hoạt hoặc tải tệp. |
| `hoi_thong_tin` | Tìm hiểu nội dung, đầu vào, lịch, giá, mentor hoặc quy định học tập. |

Urgency có cả cụm thời gian và mức rõ ràng: xử lý ngay/lập tức → `cao`; phản hồi trong hai ngày → `trung_binh`; “Khi nào tiện”, “không vội” → `thap`. Không suy mức khẩn cấp từ sentiment.

Sentiment có câu diễn đạt rõ: không hài lòng → `tieu_cuc`; thái độ trung lập → `trung_tinh`; hài lòng/đánh giá tích cực cách đội hỗ trợ tiếp nhận → `tich_cuc`. Sự hài lòng với tiếp nhận hỗ trợ có thể tồn tại khi một yêu cầu còn chờ giải quyết; ticket không nói tính năng lỗi đang hoạt động tốt.

Product phải là toàn bộ tên sản phẩm xuất hiện **nguyên văn** trong input. Không có mã hồ sơ cá nhân, điện thoại, email hoặc địa chỉ thật.

## Khử nhiễm và kiểm tra đã chạy

`quality_audit.json` ghi kết quả cùng SHA-256 các tập. Kiểm tra parse JSONL, schema, output khớp label, enum, product span, số dòng, trùng input, nhóm tình huống, UTF-8 không BOM và LF đã đạt.

- Trùng input chuẩn hóa trong mỗi tập: **0**.
- Trùng input chuẩn hóa train–eval: **0**.
- Nhóm tình huống chung train–eval: **0**.
- Trùng prompt chuẩn hóa với core train seed 250 dòng, eval target 50 dòng, eval regression 15 dòng: **0**. Prompt core được lấy từ `input` nếu không rỗng; nếu rỗng hoặc chỉ có khoảng trắng thì lấy `instruction`, đúng với định dạng regression. Không đọc hidden holdout.
- Fixture tổng hợp của bộ kiểm tra overlap đạt **4/4**: trùng instruction-only, fallback input chỉ có khoảng trắng, ưu tiên input có nội dung và mẫu không trùng. Fixture không được ghi vào core hoặc các tập custom.
- Chuẩn hóa dùng Unicode NFKC, casefold, bỏ dấu câu và gộp khoảng trắng; không bỏ dấu tiếng Việt.
- Kiểm tra từ vựng sau khi bỏ tên sản phẩm/câu nhãn chung: Jaccard cao nhất giữa một cặp train–eval là **0,35294**, ở nhóm ngày xuất kho và tủ nhận hàng. Chỉ là bộ lọc từ vựng, không chứng minh loại hết mọi tương đồng ngữ nghĩa.
- Tác giả đã đọc mẫu lắp ghép thực tế của cả năm intent trên train và eval, cùng các câu positive/neutral. Đây là rà soát của tác giả AI, không thay thế thẩm định độc lập.

Tokenizer `unsloth/Qwen3.5-4B` được tải từ snapshot local `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`, không tải mạng. Đo bằng `labkit.data.to_messages` mặc định và chat template của tokenizer, sau đó encode bản render không thêm special token:

| Tập | Token min–max | Trung bình | Vượt 512 / 1024 |
|---|---|---:|---|
| Train | 116–146 | 129,50 | 0 / 0 |
| Eval | 132–162 | 144,45 | 0 / 0 |

Đây là độ dài dữ liệu, không phải kết quả inference, loss hoặc accuracy.

## Vì sao giả thuyết phân phối mới có thể hợp lý

Ticket dùng tên gói hư cấu mới, tình huống học AI và ánh xạ năm intent thương mại sang hỗ trợ giáo dục. Những kết hợp cụ thể này khác ví dụ bán thiết bị trong core, nên có thể tạo dịch chuyển miền đối với mô hình base.

**Không thể chứng minh base chưa thấy những mẫu hoặc ý tương tự khi pretrain.** Không biết corpus pretrain của base và không có phép đo novelty. Các ý như hoàn tiền và giao sách là phổ biến; tên sản phẩm mới không đủ chứng minh phân phối hoàn toàn mới. Đây là giả thuyết cần so sánh baseline và fine-tune bằng đánh giá thật.

## Giới hạn và việc cần làm tiếp

Dữ liệu có câu mở đầu/câu nhãn lặp và tín hiệu nhãn rất rõ, nên dễ hơn ticket tự nhiên. Một nhóm có bốn biến thể không tương đương bốn quan sát thực độc lập. Train có 50% urgency thấp và 50% sentiment trung tính; các mẫu positive mô tả thái độ với hỗ trợ thay vì sản phẩm. Chưa bao phủ ticket thiếu tên sản phẩm, nhiều intent hoặc ngôn ngữ mơ hồ.

Cần người dùng xác nhận miền, đọc và sửa mẫu; thêm câu tự nhiên và ca khó; khóa eval trước thí nghiệm; rồi đưa bộ này vào **checkout/thí nghiệm cô lập** với prompt, model, step budget và scorer nhất quán. Không chép đè `data/train_seed.jsonl`, eval core hoặc baseline core trong checkout đang nộp. Chưa chạy GPU trên bộ này, chưa có kết quả chất lượng mô hình hoặc bằng chứng đạt bonus.

## Tái tạo và kiểm tra

Chạy từ thư mục gốc repo, dùng Python có transformers và cache tokenizer local nếu muốn đo token:

```powershell
.venv/Scripts/python.exe bonus_data/education_support/build_dataset.py
.venv/Scripts/python.exe bonus_data/education_support/build_dataset.py --check-only --tokenizer-dir "C:/Users/SAM/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B/snapshots/3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"
```

Lệnh đầu tái tạo JSONL/audit. Lệnh sau đọc các JSONL đã lưu, so khớp bộ tạo và kiểm tra lại. Bộ tạo không cài thư viện; khi thiếu tokenizer local, audit ghi unavailable thay vì bịa số token. Kiểm tra cơ học có lỗi thì trả exit code khác 0.
