"""Deterministic AI-assisted synthetic dataset builder; no model inference or downloads."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FAMILIES = json.loads(r'''[
  [
    "train",
    "doi_tra",
    "switch_level",
    "Tôi muốn đổi gói khóa học vì {d}; xin chuyển sang cấp phù hợp, không yêu cầu hoàn tiền.",
    [
      "bài Python cơ bản quá dễ, tôi cần lớp nâng cao",
      "phần tensor nâng cao vượt khả năng, tôi cần lớp nhập môn",
      "tôi đã biết SQL và muốn đổi sang gói xử lý ảnh",
      "tôi chưa học xác suất nên muốn đổi sang gói nền tảng"
    ]
  ],
  [
    "train",
    "doi_tra",
    "switch_schedule",
    "Xin đổi suất học sang lịch khác vì {d}; tôi vẫn muốn tiếp tục học, không xin hoàn tiền.",
    [
      "ca tối thứ Hai trùng lịch trực của tôi",
      "ca sáng thứ Tư trùng giờ thực tập",
      "nhóm cuối tuần chuyển sang tối thứ Sáu sẽ phù hợp hơn",
      "lịch tháng Mười cần chuyển sang tháng Mười Một"
    ]
  ],
  [
    "train",
    "doi_tra",
    "switch_format",
    "Tôi yêu cầu đổi hình thức gói học: {d}. Đây là yêu cầu đổi gói, không phải hoàn tiền.",
    [
      "từ lớp trực tiếp sang lớp trực tuyến vì đổi nơi ở",
      "từ video tự học sang lớp có mentor vì cần phản hồi",
      "từ lớp học cùng nhóm sang tự học vì lịch làm việc thay đổi",
      "từ phiên mentor cá nhân sang nhóm nhỏ để học cùng bạn"
    ]
  ],
  [
    "train",
    "doi_tra",
    "wrong_package_choice",
    "Tôi chọn nhầm gói khi đặt mua: {d}; xin đổi sang đúng gói và không hoàn tiền.",
    [
      "đã chọn lộ trình NLP nhưng cần lộ trình thị giác máy tính",
      "đã chọn gói dành cho quản lý nhưng cần gói cho lập trình viên",
      "đã chọn quyền học sáu tháng nhưng cần quyền học ba tháng",
      "đã chọn gói luyện phỏng vấn nhưng cần gói làm đồ án"
    ]
  ],
  [
    "train",
    "doi_tra",
    "exchange_language",
    "Tôi muốn đổi phiên bản ngôn ngữ của gói học: {d}; xin xử lý đổi gói, không hoàn tiền.",
    [
      "đang có bản tiếng Anh và muốn chuyển sang bản tiếng Việt",
      "đang có bản tiếng Việt và muốn luyện bằng bản tiếng Anh",
      "đã chọn lớp thảo luận tiếng Anh nhưng cần lớp thảo luận tiếng Việt",
      "đã chọn bộ phụ đề song ngữ nhưng muốn đổi sang phụ đề tiếng Việt"
    ]
  ],
  [
    "train",
    "doi_tra",
    "exchange_duration",
    "Tôi đề nghị đổi thời lượng gói học vì {d}; không yêu cầu trả lại tiền.",
    [
      "gói mười hai tuần quá dài, tôi muốn sang gói sáu tuần",
      "gói bốn tuần quá ngắn, tôi muốn sang gói tám tuần",
      "tôi chỉ còn rảnh hai tháng nên muốn đổi gói theo tháng",
      "tôi muốn chuyển quyền học một quý sang quyền học cả năm"
    ]
  ],
  [
    "train",
    "doi_tra",
    "exchange_specialization",
    "Xin đổi chuyên đề trong gói vì {d}; tôi muốn nhận gói thay thế, không nhận hoàn tiền.",
    [
      "dự án mới cần RAG thay vì tạo ảnh",
      "nhóm tôi cần học đánh giá mô hình thay vì triển khai web",
      "tôi đổi hướng sang học tác tử thay cho phân tích bảng",
      "tôi muốn học nhận dạng giọng nói thay cho phân loại ảnh"
    ]
  ],
  [
    "train",
    "doi_tra",
    "exchange_mentor_tier",
    "Tôi muốn đổi cấp dịch vụ mentor: {d}; xin đổi gói học chứ không hoàn tiền.",
    [
      "gói chỉ có phản hồi chữ cần chuyển sang gói gọi video",
      "gói bốn lượt góp ý cần chuyển sang gói tám lượt",
      "gói mentor nhóm cần đổi thành mentor cá nhân",
      "gói mentor chuyên lập trình cần đổi sang mentor làm sản phẩm"
    ]
  ],
  [
    "train",
    "doi_tra",
    "exchange_team_license",
    "Tôi cần đổi loại giấy phép học: {d}; xin thay gói, không hoàn tiền.",
    [
      "gói một người cần đổi sang gói nhóm ba người",
      "gói nhóm năm người cần đổi sang gói một người",
      "gói học cá nhân cần đổi sang gói cho câu lạc bộ",
      "gói dành cho nhóm cần đổi thành ba suất học độc lập"
    ]
  ],
  [
    "train",
    "doi_tra",
    "return_for_credit",
    "Tôi muốn trả lại gói chưa mở bài và nhận một gói học khác: {d}; không muốn nhận tiền hoàn.",
    [
      "trả gói phân tích dữ liệu để lấy gói Python",
      "trả gói tạo ảnh để lấy gói đạo đức AI",
      "trả gói tự học để lấy gói có hướng dẫn",
      "trả gói luyện đề để lấy gói dự án thực hành"
    ]
  ],
  [
    "train",
    "doi_tra",
    "exchange_exam_path",
    "Tôi xin đổi gói chuẩn bị đánh giá: {d}; đây là đổi gói và không hoàn tiền.",
    [
      "từ luyện bài viết sang luyện bài lập trình",
      "từ đề cơ bản sang đề nâng cao",
      "từ kiểm tra kiến thức sang bảo vệ đồ án",
      "từ luyện cá nhân sang buổi luyện theo nhóm"
    ]
  ],
  [
    "train",
    "doi_tra",
    "exchange_accessibility",
    "Xin đổi gói sang phiên bản phù hợp hơn: {d}; tôi tiếp tục học và không yêu cầu hoàn tiền.",
    [
      "cần bản có phụ đề lớn thay vì bản video thông thường",
      "cần bản có mô tả âm thanh thay vì bản chỉ có hình",
      "cần lớp tương tác bằng văn bản thay vì trao đổi bằng giọng nói",
      "cần bộ học liệu ít hiệu ứng chuyển động thay vì bản tiêu chuẩn"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "duplicate_charge",
    "Tôi đề nghị hoàn tiền khoản thanh toán bị lặp: {d}. Tôi không muốn đổi gói.",
    [
      "một đơn đăng ký bị thu tiền hai lần trong cùng phút",
      "tôi bấm trả tiền lại sau khi trang đứng và có hai giao dịch",
      "ví học tập trừ hai khoản cho cùng một kỳ gia hạn",
      "cùng một mã mua hàng xuất hiện hai lần trên biên nhận"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "cancel_before_start",
    "Xin hủy đăng ký và hoàn tiền vì {d}; không chuyển sang khóa khác.",
    [
      "tôi chưa mở bài nào và không còn thời gian học",
      "tôi vừa mua hôm qua nhưng lịch công tác kéo dài",
      "tôi chưa sử dụng quyền học và đã chọn phương án học khác",
      "lớp chưa khai giảng nhưng tôi phải dừng kế hoạch học"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "unexpected_renewal",
    "Tôi yêu cầu hoàn tiền lần gia hạn vì {d}; không cần đổi gói.",
    [
      "tôi đã tắt tự gia hạn trước ngày thu tiền",
      "tôi không chấp thuận kỳ học tiếp theo nhưng vẫn bị thu",
      "quyền học hết hạn hôm qua mà hệ thống tự trừ tiền",
      "tôi đã hủy gia hạn trong trang tài khoản nhưng hóa đơn mới vẫn phát sinh"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "class_cancelled",
    "Xin hoàn tiền học phí vì {d}; tôi không chọn lớp thay thế.",
    [
      "lớp đã bị đơn vị giả lập thông báo hủy trước khai giảng",
      "giảng viên không thể mở lớp nên khóa bị dừng",
      "lịch học bị hủy toàn bộ và không có ngày mở lại",
      "buổi nhập học được xác nhận hủy cùng cả chương trình"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "refund_unused_addon",
    "Tôi muốn hoàn tiền phần mua thêm chưa sử dụng: {d}; không đổi sang tiện ích khác.",
    [
      "gói mentor hai buổi vẫn còn nguyên lượt",
      "gói nhận xét đồ án chưa gửi bài lần nào",
      "gói thi thử chưa mở một lượt làm bài",
      "gói tài nguyên thực hành chưa tải tệp nào"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "discount_difference",
    "Xin hoàn tiền phần chênh lệch bị thu sai: {d}; không thay gói học.",
    [
      "mã giảm giá đã được chấp nhận nhưng hóa đơn thu đủ giá",
      "biên nhận ghi ưu đãi học viên cũ nhưng số tiền không giảm",
      "tôi thanh toán giá ba tháng nhưng hóa đơn tính sáu tháng",
      "gói đã hiển thị giảm mười phần trăm nhưng vẫn thu giá gốc"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "cancel_gift",
    "Tôi yêu cầu hoàn tiền suất học làm quà vì {d}; không đổi tên hoặc đổi gói.",
    [
      "người nhận chưa kích hoạt và từ chối món quà",
      "suất tặng chưa được gửi vì tôi hủy kế hoạch",
      "tôi mua hai suất làm quà nhưng chỉ cần một suất chưa dùng",
      "người được tặng không còn muốn học và chưa truy cập"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "partial_module_refund",
    "Tôi xin hoàn tiền phần nội dung bị bỏ khỏi chương trình: {d}; không yêu cầu đổi khóa.",
    [
      "chuyên đề triển khai bị cắt sau khi tôi trả tiền",
      "hai buổi mentor đã bị hủy và không tổ chức bù",
      "phần dự án cuối khóa được thông báo không mở",
      "bốn bài thực hành đã bị rút khỏi gói mua"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "charged_after_trial",
    "Tôi muốn hoàn tiền khoản thu sau dùng thử vì {d}; không tiếp tục bằng gói khác.",
    [
      "tôi đã kết thúc dùng thử trước thời hạn",
      "tôi chỉ đồng ý dùng thử miễn phí mà không chọn gói trả phí",
      "tài khoản thử đã đóng nhưng vẫn phát sinh hóa đơn",
      "tôi từ chối nâng cấp trên màn hình nhưng bị trừ tiền"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "unintended_purchase",
    "Xin hoàn tiền đơn mua nhầm vì {d}; tôi muốn hủy chứ không đổi.",
    [
      "tôi bấm xác nhận nhầm khi xem giá và chưa học",
      "tôi chọn nhầm số lượng ba suất thay vì một suất",
      "tôi mua lại gói đã sở hữu và chưa dùng đơn mới",
      "tôi tưởng nút là xem chi tiết nhưng đã hoàn tất mua"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "excess_deposit",
    "Tôi yêu cầu hoàn lại tiền đặt cọc dư: {d}; không dùng tiền đó để đổi gói.",
    [
      "khoản cọc được thu thêm một lần sau khi đã thanh toán đủ",
      "tôi đã trả toàn bộ học phí nhưng tiền giữ chỗ chưa được trả",
      "biên nhận có khoản bảo đảm phòng lab dù tôi không đăng ký phòng lab",
      "hai khoản giữ chỗ được ghi cho một suất học"
    ]
  ],
  [
    "train",
    "hoan_tien",
    "refund_processing_status",
    "Tôi cần xử lý khoản hoàn tiền đã được chấp thuận vì {d}; yêu cầu của tôi vẫn là hoàn tiền.",
    [
      "trạng thái đã duyệt mười ngày mà chưa có tiền về",
      "biên nhận hoàn tiền chưa được phát hành sau khi duyệt",
      "khoản hoàn chỉ được trả một phần dù quyết định ghi trả đủ",
      "mã tham chiếu hoàn tiền được báo không tồn tại"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "video_playback",
    "Nền tảng có lỗi cụ thể: {d}. Xin sửa lỗi để tôi học tiếp; không đổi gói hay hoàn tiền.",
    [
      "video bài đầu chỉ hiện màn hình đen trong Chrome",
      "video dừng ở giây thứ ba dù mạng ổn định",
      "bấm phát bài hai thì báo mã lỗi VID-31",
      "trình phát không có tiếng dù âm lượng bật"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "quiz_submit",
    "Tính năng nộp bài kiểm tra đang hỏng: {d}; xin khắc phục lỗi nền tảng.",
    [
      "nút nộp bị vô hiệu dù đã trả lời đủ câu",
      "bấm nộp hiện lỗi 500 và mất đáp án",
      "hệ thống quay vòng mãi sau khi nộp bài",
      "bài nộp thành công nhưng trang kết quả báo chưa làm"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "login_failure",
    "Tôi không đăng nhập được do lỗi: {d}; cần sửa hệ thống, không đổi hoặc hoàn tiền.",
    [
      "nhập đúng mật khẩu vẫn báo máy chủ 503",
      "mã xác thực hợp lệ bị từ chối ngay khi nhập",
      "đăng nhập xong bị đẩy lại trang đăng nhập liên tục",
      "nút đăng nhập không phản hồi trên Firefox"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "notebook_runtime",
    "Phòng notebook gặp lỗi: {d}; tôi yêu cầu sửa chức năng học.",
    [
      "khởi tạo môi trường báo KERNEL-07",
      "ô mã đơn giản chạy mãi không trả kết quả",
      "kernel tự ngắt sau mỗi lần nhập thư viện",
      "tệp notebook vừa lưu biến thành tệp rỗng"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "progress_sync",
    "Tiến độ học bị lỗi: {d}; xin sửa dữ liệu tiến độ trên nền tảng.",
    [
      "đã hoàn tất bài bốn nhưng thanh tiến độ vẫn bằng không",
      "tiến độ trên điện thoại không khớp với máy tính",
      "bài đã đánh dấu xong bị mở lại thành chưa học",
      "thanh hoàn thành vượt một trăm phần trăm"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "assignment_upload",
    "Chức năng tải bài tập bị hỏng: {d}; cần xử lý lỗi kỹ thuật.",
    [
      "tệp PDF hợp lệ bị báo định dạng không hỗ trợ",
      "tệp dưới giới hạn dung lượng vẫn bị từ chối",
      "tải xong trang hiển thị một tệp cũ khác",
      "nút chọn tệp biến mất ở trang bài cuối"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "certificate_render",
    "Chứng nhận số gặp lỗi: {d}; xin sửa tính năng trên nền tảng.",
    [
      "tải PDF chỉ nhận được trang trắng",
      "nút tải chứng nhận báo CERT-09",
      "mã xác minh trên chứng nhận mở ra lỗi 404",
      "bản xem trước bị cắt mất tên khóa học"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "subtitle_bug",
    "Phụ đề video bị lỗi: {d}; tôi cần sửa phụ đề trên hệ thống.",
    [
      "phụ đề chạy chậm hơn tiếng nói ba mươi giây",
      "chọn tiếng Việt nhưng hiện ký tự vuông",
      "nút bật phụ đề tự tắt mỗi lần đổi bài",
      "phụ đề bài năm lại hiển thị nội dung bài một"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "forum_bug",
    "Diễn đàn học gặp sự cố: {d}; xin khắc phục tính năng.",
    [
      "gửi câu hỏi hiện lỗi FORUM-12",
      "bài viết đã gửi không xuất hiện dù báo thành công",
      "khung soạn thảo xóa nội dung khi bấm xem trước",
      "bấm trả lời thì mở nhầm chủ đề"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "mentor_booking_bug",
    "Trang đặt lịch mentor bị lỗi: {d}; xin sửa nền tảng.",
    [
      "chọn giờ còn trống lại báo đã kín",
      "lưu lịch xong giờ bị đổi sang nửa đêm",
      "nút xác nhận không hoạt động dù đủ trường bắt buộc",
      "hủy lịch rồi nhưng lượt đặt vẫn bị khóa"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "resource_download",
    "Tệp học liệu số bị lỗi: {d}; đây là lỗi nền tảng, không phải giao hàng vật lý.",
    [
      "liên kết tải dữ liệu mẫu trả về 404",
      "tệp ZIP tải xong không giải nén được",
      "bấm tải slide nhận tệp HTML thay vì PDF",
      "tệp CSV mẫu chỉ chứa dòng tiêu đề"
    ]
  ],
  [
    "train",
    "san_pham_loi",
    "search_bug",
    "Tìm kiếm bài học không hoạt động: {d}; xin sửa lỗi ứng dụng.",
    [
      "gõ tên bài có sẵn vẫn báo không có kết quả",
      "bộ lọc cấp cơ bản trả về bài nâng cao",
      "bấm kết quả tìm kiếm mở sang bài không liên quan",
      "tìm kiếm gây treo trang trên máy tính bảng"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "physical_book_late",
    "Tôi cần hỗ trợ vận chuyển sách giấy của gói học vì {d}; không yêu cầu đổi hay hoàn tiền.",
    [
      "đơn sách chưa giao sau ngày dự kiến ba ngày",
      "mã vận đơn đứng ở kho trung chuyển suốt một tuần",
      "sách được hẹn giao hôm qua nhưng chưa có người giao",
      "đơn sách báo đang giao từ sáng mà chưa tới"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "physical_kit_tracking",
    "Xin kiểm tra hành trình bộ linh kiện thực hành gửi bằng bưu kiện: {d}.",
    [
      "mã vận đơn không tra được trên trang hãng giao",
      "bưu kiện không cập nhật sau khi rời kho",
      "trang theo dõi hiện hai ngày giao khác nhau",
      "đơn bộ linh kiện báo lấy hàng nhưng chưa có mã vận chuyển"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "printed_workbook_address",
    "Tôi muốn sửa địa chỉ giao vở bài tập in vì {d}; đây là yêu cầu giao hàng vật lý.",
    [
      "đơn chưa gửi nhưng thiếu tên tòa nhà giả lập",
      "tôi chuyển chỗ ở trước khi bưu kiện rời kho",
      "địa chỉ nhận cần bổ sung tầng của văn phòng giả lập",
      "điểm nhận ghi nhầm cơ sở học giả lập A thành B"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "delivery_time_slot",
    "Xin điều chỉnh thời gian giao tài liệu giấy vì {d}.",
    [
      "tôi chỉ có thể nhận sách vào buổi chiều",
      "đơn vị giao hẹn sáng nhưng tôi cần ca tối",
      "tôi muốn nhận bộ thẻ học vào thứ Bảy",
      "bưu kiện cần giao sau kỳ nghỉ tại điểm nhận giả lập"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "failed_delivery",
    "Đơn giao sách giấy bị giao không thành công: {d}; xin tổ chức giao lại.",
    [
      "nhân viên tới ngoài khung giờ đã hẹn",
      "điểm nhận đóng cửa vào lần giao đầu",
      "hãng giao ghi không liên hệ được dù chưa gọi",
      "bưu kiện bị hoàn về kho sau một lần giao"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "physical_material_split",
    "Tôi hỏi về vận chuyển bộ học liệu giấy bị tách kiện: {d}.",
    [
      "đã nhận sách nhưng chưa nhận vở bài tập cùng đơn",
      "hai kiện được tạo mã nhưng mới một kiện tới",
      "bộ thẻ học gửi riêng chưa có lịch giao",
      "đơn có sách và poster mà poster còn ở kho"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "signed_not_received",
    "Tôi cần kiểm tra bưu kiện sách ghi đã giao nhưng chưa nhận: {d}.",
    [
      "trạng thái ghi ký nhận nhưng quầy lễ tân chưa có sách",
      "ảnh giao hàng là một cửa nhà không phải điểm nhận giả lập",
      "mã vận đơn báo phát thành công nhưng kiện chưa tới",
      "nhật ký ghi nhận tại cơ sở A trong khi chọn cơ sở B"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "pickup_point",
    "Xin chuyển điểm nhận bưu kiện tài liệu giấy vì {d}.",
    [
      "tôi muốn lấy sách tại bưu cục thay vì chờ giao",
      "điểm nhận sách hiện tại đóng cửa nên cần điểm khác",
      "tôi muốn nhận bộ flashcard tại kho tự lấy giả lập",
      "bộ bài tập in cần chuyển đến tủ nhận hàng giả lập"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "physical_shipping_fee",
    "Tôi cần giải thích phí giao bộ học liệu vật lý: {d}; đang hỏi về vận chuyển.",
    [
      "phí giao sách cao hơn mức hiện lúc đặt đơn",
      "đơn bộ linh kiện có thêm phí giao vùng giả lập",
      "hai kiện cùng địa chỉ bị tính phí giao hai lần",
      "phí giao nhanh chưa được ghi trên vận đơn"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "reroute_parcel",
    "Tôi yêu cầu đổi tuyến giao kiện sách đang vận chuyển vì {d}.",
    [
      "điểm nhận giả lập A ngừng hoạt động nên chuyển sang B",
      "tôi cần chuyển bưu kiện từ văn phòng sang điểm học giả lập",
      "người nhận chuyển nơi học nên kiện cần đổi tuyến",
      "đơn đang ở kho khu vực nhưng cần chuyển sang kho khác"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "delivery_instructions",
    "Xin cập nhật hướng dẫn giao sách giấy: {d}.",
    [
      "gửi bưu kiện tại quầy tiếp nhận tầng một của điểm giả lập",
      "không đặt kiện trước cửa vì tôi cần ký nhận trực tiếp",
      "gọi qua kênh hỗ trợ trước khi mang bộ linh kiện tới",
      "dán nhãn giao cho bộ phận học liệu tại điểm giả lập"
    ]
  ],
  [
    "train",
    "van_chuyen",
    "physical_dispatch_date",
    "Tôi muốn biết ngày xuất kho tài liệu in vì {d}; hỏi về giao hàng vật lý.",
    [
      "đơn sách đã đóng gói nhưng chưa bàn giao hãng giao",
      "bộ bài tập in đã thanh toán mà chưa có ngày gửi",
      "bộ thẻ ôn tập còn trạng thái chờ lấy hàng",
      "sách bổ trợ được thông báo sẵn kho nhưng chưa xuất"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "course_prerequisites",
    "Tôi muốn biết yêu cầu đầu vào của gói học: {d}. Chỉ cần thông tin.",
    [
      "có cần biết Python trước khi bắt đầu không",
      "có yêu cầu học đại số tuyến tính trước không",
      "người chưa dùng Git có theo học được không",
      "có cần kiến thức thống kê ở mức đại học không"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "course_syllabus",
    "Xin cung cấp thông tin nội dung khóa: {d}; tôi đang tìm hiểu trước khi đăng ký.",
    [
      "có phần xây dựng RAG với dữ liệu riêng không",
      "có hướng dẫn đánh giá độ đúng câu trả lời không",
      "có bài về tác tử sử dụng công cụ không",
      "có học cách triển khai API mô hình không"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "course_duration",
    "Tôi hỏi thời lượng của khóa: {d}; chưa có yêu cầu thay đổi đơn.",
    [
      "mỗi tuần cần học bao nhiêu giờ",
      "toàn bộ lộ trình kéo dài mấy tuần",
      "mỗi buổi thực hành dài bao lâu",
      "quyền xem lại video được giữ trong bao lâu"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "course_price",
    "Xin cho tôi thông tin giá của gói học: {d}; tôi chưa thanh toán.",
    [
      "học phí niêm yết của gói một người là bao nhiêu",
      "gói nhóm ba người có mức giá nào",
      "giá đã bao gồm các buổi mentor chưa",
      "phí kiểm tra cuối khóa có nằm trong học phí không"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "course_schedule",
    "Tôi cần biết lịch mở lớp: {d}; chỉ hỏi thông tin trước khi mua.",
    [
      "đợt tiếp theo bắt đầu ngày nào",
      "lớp tối có học vào cuối tuần không",
      "có ca dành cho người đi làm không",
      "ngày tổ chức buổi giới thiệu là khi nào"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "mentor_information",
    "Tôi muốn tìm hiểu cách hỗ trợ mentor: {d}.",
    [
      "học viên được hỏi đáp qua kênh nào",
      "mỗi tháng có bao nhiêu lượt góp ý",
      "mentor có xem và nhận xét mã nguồn không",
      "buổi hỗ trợ được tổ chức cá nhân hay theo nhóm"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "assessment_rules",
    "Xin giải thích cách đánh giá trong khóa: {d}.",
    [
      "điểm đồ án chiếm bao nhiêu trong kết quả",
      "có bao nhiêu bài kiểm tra giữa khóa",
      "có được làm lại bài kiểm tra không",
      "cần nộp những thành phần nào cho đồ án"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "hardware_requirements",
    "Tôi cần thông tin cấu hình để học: {d}.",
    [
      "máy có tám GB RAM có dùng phòng lab được không",
      "có bắt buộc có GPU trên máy cá nhân không",
      "máy dùng Linux có được hỗ trợ không",
      "trình duyệt nào phù hợp để chạy notebook"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "certificate_information",
    "Tôi muốn biết thông tin chứng nhận của gói: {d}.",
    [
      "điều kiện nhận chứng nhận là gì",
      "chứng nhận có ghi chuyên đề đã hoàn thành không",
      "có bản PDF của chứng nhận không",
      "chứng nhận có mã xác minh trực tuyến không"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "teaching_language",
    "Xin thông tin về ngôn ngữ giảng dạy: {d}.",
    [
      "video chính dùng tiếng Việt hay tiếng Anh",
      "có phụ đề tiếng Việt cho bài nâng cao không",
      "tài liệu bài tập được viết bằng ngôn ngữ nào",
      "buổi trao đổi với mentor dùng tiếng Việt được không"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "learning_resources",
    "Tôi muốn biết các tài nguyên đi kèm: {d}; không có sự cố tải tệp.",
    [
      "có cung cấp dữ liệu mẫu cho đồ án không",
      "có slide sau mỗi bài không",
      "có notebook mẫu để tham khảo không",
      "có bộ câu hỏi ôn tập theo từng chương không"
    ]
  ],
  [
    "train",
    "hoi_thong_tin",
    "registration_process",
    "Tôi đang tìm hiểu thủ tục đăng ký: {d}.",
    [
      "cần làm những bước nào để tạo suất học",
      "có thể giữ chỗ trước khi thanh toán không",
      "đăng ký nhóm cần cung cấp thông tin gì",
      "có buổi học trải nghiệm trước khi mua không"
    ]
  ],
  [
    "eval",
    "doi_tra",
    "switch_learning_goal",
    "Tôi đã đổi mục tiêu học tập và cần thay gói đang sở hữu: {d}. Hãy đổi gói giúp tôi; tôi không muốn nhận lại tiền.",
    [
      "bây giờ cần học kiểm định công bằng thay cho tăng tốc mô hình",
      "công việc mới cần học OCR thay cho tối ưu lời nhắc",
      "đồ án đổi sang dự báo chuỗi thời gian thay cho chatbot",
      "tôi cần học bảo mật tác tử thay cho tạo nội dung"
    ]
  ],
  [
    "eval",
    "doi_tra",
    "swap_practice_environment",
    "Nhờ thay gói thực hành hiện tại bằng phiên bản khác: {d}. Tôi muốn đổi quyền học, không xin hoàn tiền.",
    [
      "chuyển gói dùng công cụ kéo thả sang gói viết mã",
      "đổi gói thực hành cloud sang gói chạy cục bộ",
      "thay gói phòng lab notebook bằng gói phòng lab dòng lệnh",
      "đổi gói thực hành trên bảng tính sang gói thực hành cơ sở dữ liệu"
    ]
  ],
  [
    "eval",
    "doi_tra",
    "swap_learning_pace",
    "Tôi cần đổi gói học do nhịp học không phù hợp: {d}. Tôi vẫn tiếp tục học bằng gói thay thế.",
    [
      "đổi lớp tăng tốc sang lớp học giãn cách",
      "thay gói có hạn chót cố định bằng gói học linh hoạt",
      "đổi lớp theo tuần sang lộ trình chia theo từng chặng",
      "chuyển gói bài ngắn hằng ngày sang gói học theo buổi dài"
    ]
  ],
  [
    "eval",
    "hoan_tien",
    "bank_settlement_mismatch",
    "Tôi cần nhận lại khoản tiền thu thừa, {d}. Hãy hoàn tiền, không chuyển thành tín dụng học tập.",
    [
      "ngân hàng xác nhận trừ nhiều hơn tổng trên biên nhận",
      "hệ thống cộng thêm tiền thuế hai lần cho cùng một hóa đơn",
      "số tiền thanh toán vượt giá đã xác nhận của đúng gói",
      "biên nhận chỉ có một suất mà số tiền tính thành hai suất"
    ]
  ],
  [
    "eval",
    "hoan_tien",
    "withdraw_closed_lab",
    "Tôi muốn lấy lại học phí vì {d}. Tôi rút đăng ký và yêu cầu hoàn tiền, không đổi gói.",
    [
      "phòng lab bắt buộc đã đóng trước ngày bắt đầu",
      "nhà tổ chức giả lập dừng nhận học viên sau khi tôi trả tiền",
      "đợt học chuyển sang kỳ không xác định nên tôi không tham gia",
      "buổi thực hành cốt lõi không thể tổ chức trong đợt đã mua"
    ]
  ],
  [
    "eval",
    "hoan_tien",
    "refund_residual_balance",
    "Nhờ hoàn trả phần tiền còn lại cho tôi: {d}. Tôi không mua thêm hoặc đổi sang sản phẩm khác.",
    [
      "tôi đã đóng ví học tập và cần rút số dư chưa dùng",
      "đơn đã hủy nhưng khoản ứng trước còn bị giữ",
      "đợt học kết thúc và tiền đặt chỗ dư chưa được trả",
      "suất học đã được hủy đúng điều kiện giả lập nhưng tiền tạm thu chưa trả"
    ]
  ],
  [
    "eval",
    "san_pham_loi",
    "accessibility_control_bug",
    "Tôi gặp lỗi điều khiển trợ năng ở nền tảng: {d}. Nhờ sửa tính năng để học tiếp.",
    [
      "nhấn Tab không đưa tiêu điểm tới nút phát bài",
      "trình đọc màn hình đọc sai thứ tự mục bài học",
      "nút tăng cỡ chữ không thay đổi cỡ chữ thực tế",
      "chế độ giảm chuyển động vẫn nhấp nháy liên tục"
    ]
  ],
  [
    "eval",
    "san_pham_loi",
    "lab_grading_bug",
    "Máy chấm bài thực hành hoạt động sai: {d}. Tôi cần khắc phục lỗi kỹ thuật trên nền tảng.",
    [
      "bài đã có kết quả được chấm thành tệp trống",
      "trang chấm ghi lỗi GRADER-14 với cả bài mẫu được cung cấp",
      "hệ thống lấy kết quả của lần nộp cũ thay cho lần mới",
      "bộ chấm trả hai điểm khác nhau cho cùng một lần nộp"
    ]
  ],
  [
    "eval",
    "san_pham_loi",
    "discussion_notification_bug",
    "Tính năng thông báo học tập bị lỗi: {d}. Xin sửa hệ thống thông báo.",
    [
      "thông báo câu trả lời của mentor dẫn tới trang không tồn tại",
      "tắt nhắc lịch rồi nhưng bảng nhắc vẫn bật lại mỗi giờ",
      "số thông báo chưa đọc không giảm dù đã mở hết",
      "chuông báo hiện một thảo luận mà tài khoản không tham gia"
    ]
  ],
  [
    "eval",
    "van_chuyen",
    "physical_customs_hold",
    "Tôi cần xử lý kiện sách giấy đang qua điểm kiểm tra vận chuyển giả lập: {d}. Đây là bưu kiện vật lý.",
    [
      "kiện bị giữ tại trạm kiểm tra và cần xác nhận danh mục sách",
      "hãng giao yêu cầu bổ sung mô tả cho bộ thẻ học in",
      "bưu kiện bộ bài tập bị dừng vì thiếu nhãn phân loại",
      "trạm trung chuyển yêu cầu xác nhận trọng lượng kiện sách"
    ]
  ],
  [
    "eval",
    "van_chuyen",
    "physical_locker_access",
    "Bưu kiện tài liệu in đã tới tủ nhận hàng nhưng {d}. Nhờ hỗ trợ nhận kiện vật lý.",
    [
      "mã mở tủ do hãng giao gửi không hoạt động",
      "tủ ghi ô chứa khác với thông báo giao sách",
      "thời hạn lấy sách bị ghi kết thúc trước giờ giao",
      "mã tủ bị hết hạn khi tôi vừa tới điểm nhận"
    ]
  ],
  [
    "eval",
    "van_chuyen",
    "physical_delivery_return_route",
    "Tôi hỏi về tuyến hoàn của bưu kiện sách giấy chưa nhận được: {d}. Tôi muốn nhận lại kiện, không xin hoàn tiền.",
    [
      "kiện bị chuyển về kho gốc vì điểm nhận tạm đóng",
      "bưu kiện đi ngược tuyến sau khi hãng đổi kho phân phối",
      "sách được trả về người gửi do lỗi phân loại tuyến",
      "đơn vị giao ghi hoàn kiện dù chưa có lượt giao nào"
    ]
  ],
  [
    "eval",
    "hoi_thong_tin",
    "portfolio_usage_rules",
    "Trước khi mua, tôi muốn biết quy định dùng sản phẩm học tập: {d}. Chỉ cần giải thích thông tin.",
    [
      "đồ án có được đưa vào hồ sơ năng lực cá nhân không",
      "mã nguồn bài thực hành có được đăng ở kho công khai không",
      "dữ liệu mẫu có được dùng cho bài thuyết trình không",
      "có được chia sẻ ảnh kết quả đồ án trong hồ sơ xin việc không"
    ]
  ],
  [
    "eval",
    "hoi_thong_tin",
    "collaboration_information",
    "Tôi muốn hỏi cách học cùng người khác trong gói: {d}; chưa đăng ký.",
    [
      "có hoạt động ghép nhóm làm đồ án không",
      "nhóm tự lập có được nộp chung một đồ án không",
      "có diễn đàn theo chuyên đề để tìm bạn học không",
      "có buổi trao đổi giữa các nhóm vào cuối đợt không"
    ]
  ],
  [
    "eval",
    "hoi_thong_tin",
    "offline_access_information",
    "Xin cho tôi thông tin cách học khi kết nối không liên tục: {d}. Tôi đang tìm hiểu chứ chưa gặp lỗi.",
    [
      "có thể tải video để xem ngoại tuyến không",
      "bài tập có bản hướng dẫn dùng khi không có mạng không",
      "có thể lưu tiến độ trên máy rồi đồng bộ sau không",
      "slide có sẵn bản nhẹ để tải qua mạng chậm không"
    ]
  ]
]''')
INTENTS = {"doi_tra", "hoan_tien", "san_pham_loi", "van_chuyen", "hoi_thong_tin"}
URGENCIES = {"cao", "trung_binh", "thap"}
SENTIMENTS = {"tieu_cuc", "trung_tinh", "tich_cuc"}
KEYS = {"intent", "urgency", "product", "sentiment"}
PRODUCTS = {
    "train": ["Khóa MâyTre AI Cơ Bản", "Gói SenĐá AI Thực Hành", "Lộ Trình GióLúa AI", "Gói ĐomĐóm AI Mentor"],
    "eval": ["Khóa SuốiNgọc AI", "Gói TrăngBạc AI Lab", "Lộ Trình ĐồiMơ AI", "Gói NắngBiếc AI Studio"],
}
# Explicit urgency and sentiment eliminate ambiguous label assignments.
SIGNALS = {
    "train": [
        ("cao", "tieu_cuc", "Tôi rất không hài lòng về việc này. Mức khẩn cấp cao: cần xử lý ngay hôm nay."),
        ("trung_binh", "trung_tinh", "Tôi giữ thái độ trung lập về yêu cầu này. Mức khẩn cấp trung bình: cần phản hồi trong hai ngày tới."),
        ("thap", "tich_cuc", "Tôi hài lòng với cách đội hỗ trợ tiếp nhận yêu cầu, dù việc này còn cần xử lý. Mức khẩn cấp thấp: Khi nào tiện thì hỗ trợ giúp tôi."),
        ("thap", "trung_tinh", "Tôi trao đổi với thái độ trung lập. Mức khẩn cấp thấp: không vội, khi nào tiện thì phản hồi."),
    ],
    "eval": [
        ("thap", "tieu_cuc", "Cảm nhận hiện tại của tôi là không hài lòng. Tôi xác nhận mức khẩn cấp thấp; Khi nào tiện hãy hỗ trợ."),
        ("cao", "trung_tinh", "Tôi không có đánh giá tích cực hay tiêu cực, thái độ trung lập. Cần xử lý lập tức, mức khẩn cấp cao."),
        ("trung_binh", "tich_cuc", "Tôi đánh giá tích cực cách tiếp nhận của đội hỗ trợ và hài lòng, ngay cả khi yêu cầu còn chờ xử lý. Mong phản hồi trong hai ngày, mức khẩn cấp trung bình."),
        ("thap", "trung_tinh", "Thái độ của tôi trung lập. Yêu cầu này không gấp; mức khẩn cấp thấp, khi nào tiện xử lý cũng được."),
    ],
}

def normal(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

def ticket_text(row):
    """Core regression prompts live in instruction when input is blank."""
    value = row.get("input")
    return str(value).strip() if value and str(value).strip() else str(row.get("instruction") or "").strip()

def exact_overlap(custom_prompts, original_rows):
    return {normal(text) for text in custom_prompts} & {normal(ticket_text(row)) for row in original_rows}

def check_overlap_fixture():
    """Synthetic fixture; never add it to original or training/evaluation data."""
    prompt = "Thủ đô của Việt Nam là thành phố nào?"
    fixture = {"input": "", "instruction": prompt}
    assert exact_overlap([prompt], [fixture]) == {normal(prompt)}, "instruction-only regression match missed"
    assert ticket_text({"input": "  ", "instruction": prompt}) == prompt, "whitespace-only input fallback missed"
    assert ticket_text({"input": "Ticket thực tế", "instruction": prompt}) == "Ticket thực tế", "nonempty input must take priority"
    assert not exact_overlap(["Nội dung khác hoàn toàn"], [fixture]), "unrelated fixture falsely matched"
    return {"status": "passed", "cases": 4, "scope": "synthetic instruction-only/whitespace fallback, input priority, nonmatch; no original files altered"}

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def compose(instruction):
    sets = {"train": [], "eval": []}
    groups = {"train": [], "eval": []}
    for family_index, (split, intent, family, pattern, details) in enumerate(FAMILIES):
        for index, detail in enumerate(details):
            product = PRODUCTS[split][(index + family_index) % len(PRODUCTS[split])]
            urgency, sentiment, signal = SIGNALS[split][index]
            body = pattern.replace("{d}", detail)
            if split == "train":
                message = f"Về sản phẩm {product}: {body} {signal}"
            else:
                message = f"{body} Yêu cầu này liên quan đến sản phẩm {product}. {signal}"
            label = {"intent": intent, "urgency": urgency, "product": product, "sentiment": sentiment}
            sets[split].append({
                "instruction": instruction,
                "input": message,
                "output": json.dumps(label, ensure_ascii=False),
                "label": label,
            })
            groups[split].append(f"{intent}/{family}")
    return sets, groups

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

def token_audit(sets, tokenizer_dir=None):
    """Use only cached/local tokenizer, never network or automatic package installation."""
    if not importlib.util.find_spec("transformers"):
        return {"status": "unavailable", "reason": "transformers not installed; no package installation attempted"}
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "src"))
    from labkit.data import to_messages
    sources = [str(tokenizer_dir)] if tokenizer_dir else ["Qwen/Qwen3.5-0.8B", "unsloth/Qwen3.5-4B"]
    errors = []
    for source in sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
            results = {}
            for split, rows in sets.items():
                rendered = [tokenizer.apply_chat_template(to_messages(row), tokenize=False,
                    add_generation_prompt=False) for row in rows]
                counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in rendered]
                results[split] = {
                    "n": len(counts), "min": min(counts), "max": max(counts),
                    "mean": statistics.mean(counts), "median": statistics.median(counts),
                    "over_512": sum(x > 512 for x in counts), "over_1024": sum(x > 1024 for x in counts),
                }
            return {"status": "measured_local", "source": source, "format": "labkit.data.to_messages default NAIVE_PROMPT / tokenizer default chat template", "stats": results}
        except Exception as error:
            errors.append({"source": source, "error_type": type(error).__name__})
    return {"status": "unavailable", "reason": "No listed tokenizer source could load locally; no network request", "attempts": errors}

def audit(sets, groups, tokenizer_dir=None):
    errors = []
    by_split = {}
    for split, rows in sets.items():
        normalized = [normal(row["input"]) for row in rows]
        for i, row in enumerate(rows, 1):
            lab = row["label"]
            if set(row) != {"instruction", "input", "output", "label"}:
                errors.append(f"{split}:{i}: row keys")
            if set(lab) != KEYS or json.loads(row["output"]) != lab:
                errors.append(f"{split}:{i}: label/output schema")
            if lab["intent"] not in INTENTS or lab["urgency"] not in URGENCIES or lab["sentiment"] not in SENTIMENTS:
                errors.append(f"{split}:{i}: invalid labels")
            if lab["product"] not in row["input"]:
                errors.append(f"{split}:{i}: product span")
            if len(row["input"]) < 80:
                errors.append(f"{split}:{i}: short message")
        dup = len(normalized) - len(set(normalized))
        if dup:
            errors.append(f"{split}: {dup} normalized duplicate inputs")
        by_split[split] = {
            "rows": len(rows), "scenario_families": len(set(groups[split])),
            "normalized_duplicate_inputs": dup, "intent_counts": dict(Counter(r["label"]["intent"] for r in rows)),
            "urgency_counts": dict(Counter(r["label"]["urgency"] for r in rows)),
            "sentiment_counts": dict(Counter(r["label"]["sentiment"] for r in rows)),
            "character_lengths": {"min": min(map(lambda r: len(r["input"]), rows)), "max": max(map(lambda r: len(r["input"]), rows))},
        }
    if len(sets["train"]) < 240 or len(sets["eval"]) < 50:
        errors.append("minimum split counts")
    exact = set(map(lambda r: normal(r["input"]), sets["train"])) & set(map(lambda r: normal(r["input"]), sets["eval"]))
    group_overlap = set(groups["train"]) & set(groups["eval"])
    if exact or group_overlap:
        errors.append("train/eval contamination")
    core_comparisons = {}
    all_custom = [ticket_text(row) for rows in sets.values() for row in rows]
    for filename in ["train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl"]:
        path = ROOT / "data" / filename
        original = read_jsonl(path)
        overlap = exact_overlap(all_custom, original)
        core_comparisons[filename] = {
            "rows_checked": len(original), "normalized_exact_input_overlap": len(overlap),
            "prompt_extraction": "nonempty input, else instruction",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if overlap:
            errors.append(f"original {filename} overlap")
    # Remove common label cues/product and compare content words across different splits.
    # These are screening statistics, not proof that all paraphrase leakage is absent.
    def words(row):
        body = row["input"]
        for product_list in PRODUCTS.values():
            for product in product_list:
                body = body.replace(product, "")
        for signals in SIGNALS.values():
            for _, _, signal in signals:
                body = body.replace(signal, "")
        stop = {"tôi", "xin", "gói", "vì", "cần", "không", "và", "có", "là", "của", "thì", "cho", "để", "này", "được", "một", "về", "sản", "phẩm", "yêu", "cầu", "học", "tiền", "hoàn", "đổi", "liên", "quan", "đến", "hãy", "giúp", "vẫn"}
        return set(normal(body).split()) - stop
    max_pair = {"jaccard": 0.0}
    for ti, tr in enumerate(sets["train"]):
        tw = words(tr)
        for ei, er in enumerate(sets["eval"]):
            ew = words(er)
            score = len(tw & ew) / max(1, len(tw | ew))
            if score > max_pair["jaccard"]:
                max_pair = {"jaccard": score, "train_row": ti + 1, "eval_row": ei + 1, "train_family": groups["train"][ti], "eval_family": groups["eval"][ei]}
    report = {
        "status": "mechanical_checks_passed" if not errors else "failed",
        "authorship": "AI-assisted synthetic scenarios; no independent human annotation",
        "quality_scope": "Mechanical QA plus author semantic sample review; not user-approved domain quality",
        "sets": by_split, "schema_errors": errors,
        "train_eval_normalized_exact_overlap": len(exact), "train_eval_family_overlap": sorted(group_overlap),
        "original_core_comparisons": core_comparisons,
        "overlap_checker_regression_fixture": check_overlap_fixture(),
        "near_duplicate_screen": {"method": "max content-word Jaccard after removing product/common signals; lexical screening only", "highest_cross_split_pair": max_pair},
        "tokenizer": token_audit(sets, tokenizer_dir),
        "semantic_review": {
            "method": "Author read scenario family source and representative assembled rows",
            "sample_rows": {"train": [1, 49, 97, 145, 193], "eval": [1, 13, 25, 37, 49]},
            "families_reviewed": [
                "switch_level: explicit replacement, no cash refund",
                "duplicate_charge: refund of repeated payment",
                "video_playback: concrete playback failure",
                "physical_book_late: physical printed book delivery",
                "course_prerequisites: course entry information",
                "switch_learning_goal: different learning objective, package swap",
                "bank_settlement_mismatch: excess payment refund",
                "accessibility_control_bug: concrete platform accessibility malfunction",
                "physical_customs_hold: physical parcel checkpoint",
                "portfolio_usage_rules: information about project sharing",
            ],
            "sentiment_rule": "Positive refers explicitly to current reception/support attitude, even with unresolved request; no assertion that the faulty feature itself is satisfactory",
            "limitation": "AI author review is not independent human validation; repetitive phrasing and explicit label cues limit naturalness",
        },
        "limitations": [
            "No guarantee examples are unseen in foundation-model pretraining.",
            "Invented product names and local scenarios support a distribution-novelty hypothesis, not measured novelty.",
            "No GPU training, model evaluation, or B2 result claim.",
            "Label cues are explicit and synthetic; field evaluation may be substantially harder.",
            "Different products and families are held out; lexical screening cannot exclude all semantic similarity.",
        ],
    }
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    instruction = read_jsonl(ROOT / "data" / "train_seed.jsonl")[0]["instruction"]
    sets, groups = compose(instruction)
    files = {"train": HERE / "train.jsonl", "eval": HERE / "eval_target.jsonl"}
    if args.check_only:
        for split, path in files.items():
            actual = read_jsonl(path)
            if actual != sets[split]:
                raise SystemExit(f"Saved {split} does not match reproducible generation")
            sets[split] = actual
    else:
        for split, path in files.items():
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sets[split]), encoding="utf-8", newline="\n")
    report = audit(sets, groups, args.tokenizer_dir)
    report["saved_files"] = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "utf8_bom": p.read_bytes().startswith(b"\xef\xbb\xbf"), "crlf_count": p.read_bytes().count(b"\r\n")} for p in files.values()}
    write_json(HERE / "quality_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(report["schema_errors"]))

if __name__ == "__main__":
    main()
