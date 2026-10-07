# CUSTOM_DATASET — hồ sơ B2 đang chuẩn bị

Hồ sơ nguồn, cách tạo, khử nhiễm, giả thuyết phân phối mới và giới hạn nằm tại [bonus_data/education_support/CUSTOM_DATASET.md](../bonus_data/education_support/CUSTOM_DATASET.md).

Dữ liệu tổng hợp do AI hỗ trợ biên soạn: **240 train + 60 eval**, tách 75 nhóm tình huống và giữ schema bốn nhãn gốc. JSONL và audit nằm ở `bonus_data/education_support/`, ngoài tập core.

Đã kiểm tra cơ học; **chưa khẳng định đạt B2**. Miền chưa được người dùng phê duyệt, chất lượng chưa được người dùng/chuyên gia thẩm định độc lập, chưa huấn luyện/đánh giá GPU trên bộ này. Cần đưa vào thí nghiệm cô lập trước khi training; không ghi đè train/eval/baseline core. Đây không phải 200 mẫu con người gán nhãn và không thể chứng minh base chưa thấy ý tương tự khi pretrain.
