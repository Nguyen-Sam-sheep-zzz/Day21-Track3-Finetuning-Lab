# Lab 21 — Fine-tuning phân loại ticket CSKH tiếng Việt

**Họ tên:** Nguyễn Nhân Sâm · **MSSV:** 2A202602672

**Ngày thí nghiệm:** 07/10/2026 · **Thiết bị:** Tesla T4 trên Colab Free, precision fp16

**Repo:** https://github.com/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab

**Nhánh:** feature/lab21-finetuning · **Commit chạy GPU:** `90679f91842e9f42c9192b9c5eab316935705e9c`

> Core NB1–NB5 đã chạy đầy đủ. Adapter correct tăng target từ 0,765 lên 0,970 nhưng regression giảm từ 0,791111 xuống 0,522222, nên cổng hồi quy kết luận **FAILED**. Đây là kết quả thí nghiệm, không phải lỗi chạy notebook. Tập target có 33 ca FT thắng, 17 hòa, **0 ca thua**: yêu cầu ≥2 ca thua của rubric 3.4 chưa được chứng minh. Không thay eval hoặc tạo ví dụ để lấp thiếu hụt này.

## 1. Bài toán, model và dữ liệu

Mục tiêu là biến một ticket CSKH tiếng Việt thành JSON gồm intent, urgency, product và sentiment. Chọn `unsloth/Qwen3.5-4B` theo cấu hình T4 của lab để thực hiện cùng base cho baseline và fine-tuning. Tên namespace không có nghĩa đã dùng thư viện Unsloth: manifest ghi thư viện này không được cài; pipeline thực tế dùng Transformers, PEFT và TRL.

Giữ corpus mặc định `data/train_seed.jsonl` với 250 mẫu, split seed 42 thành 225 train và 25 validation. Full evaluation gồm 50 ticket target và 15 instruction regression; không đặt EVAL_LIMIT, không đổi prompt hay scorer sau khi xem kết quả. Các checksum gốc và hash split khớp manifest. Validation split đã tạo nhưng không được sử dụng như một phép đo validation riêng trong các notebook train; final_loss là trung bình loss huấn luyện do trainer trả về.

Target là trung bình độ chính xác **bốn trường** trên mỗi ticket. Vì vậy 0,970 không đồng nghĩa 97% ticket hoàn toàn đúng: correct đúng toàn bộ bốn trường ở **44/50 ticket (88%)**, tương ứng 194/200 trường. Regression là trung bình keyword recall trên 15 instruction, không phải accuracy tổng quát toàn diện.

## 2. Mask, template và độ dài

Nguồn: `results/mask_proof.json`, `template_check.json`, `token_stats.json`, `colab_nb1.txt`.

| Kiểm tra NB1 | Số đo |
|---|---:|
| Token supervised / tổng token trong proof | 39 / 94 |
| Supervised fraction | 0,4149 |
| Answer nằm trong loss | true |
| Question được mask | true |
| Độ dài p95 / dài nhất của 250 mẫu | 98 / 101 |
| Suggested max length | 256 |
| Max length dùng cho tất cả run | 1024 |

Đoạn supervised gồm phần đóng think, JSON đáp án và EOS, không chứa câu hỏi. Template test giữ được tag và nội dung reasoning trong ví dụ có trace. Corpus mặc định lại chỉ chứa đáp án JSON; khả năng giữ trace của template không chứng minh dữ liệu train có reasoning trace.

Giữ giới hạn tier 1024 để các run dùng cùng recipe. Đây là giới hạn trên, không phải kết quả suy ra từ p95: p95 gợi ý 256 và mẫu dài nhất 101 đều thấp hơn cả hai giới hạn, nên không cắt answer. Nếu tối ưu tài nguyên trong thí nghiệm mới, có thể chốt 256 trước khi đo baseline; không đổi giữa thí nghiệm này.

Chẩn đoán CPU `mask_agreement_local.txt` trả FAIL vì chat template không có generation markers và tokenizer-level assistant mask rỗng. Pipeline dùng labels đã tạo từ labkit và pretokenized dataset, không bật assistant_only_loss để TRL tự suy diễn mask. Không báo chẩn đoán này thành PASS; NB1 trên Colab đã chứng minh mask được dùng.

## 3. Baseline đóng băng và khả năng truy vết

Manifest được ghi lúc **2026-10-07T11:00:22.268812+00:00**. NB3 bắt đầu lúc **11:04:40 UTC**, sau lần ghi manifest và checkpoint before_training lúc 11:04:31 UTC. Checkpoint before_training không có runs.csv hoặc adapter; after_correct có một run; after_contrasts và core_results có đủ bốn run. Frozen baseline, raw prediction, manifest và environment.txt có byte/hash giống nhau qua cả bốn checkpoint.

`experiment_manifest.json` ghi model, seed, tier, epochs=2, max_length=1024, full eval, prompt SHA256, hash corpus/split, phiên bản thư viện và hash của artifacts đi kèm. Prompt (b) SHA256 bắt đầu bằng `719e74d3b6232053`, khớp prompt gốc. Raw output baseline target và regression nằm trong `baseline_predictions.json`; mỗi output target được ghép đúng thứ tự, ticket và nhãn ở `paired_qualitative.json`. Kiểm tra nhập checkpoint xác nhận 38 file source khớp commit GPU.

Đây là bằng chứng nhất quán từ file và log, không phải chứng thực thực thi có chữ ký. Model/tokenizer revision trong manifest là null, nên không khẳng định đã pin chính xác historical Hub revision. `received_checkpoints.json` giữ SHA256 bốn ZIP gốc; `core_evidence_audit.json` ghi kiểm tra độc lập bằng scorer gốc.

Môi trường GPU ghi nhận Python 3.13.15, torch 2.11.0+cu130, transformers 5.18.0, TRL 1.14.2, PEFT 0.21.1, accelerate 1.15.0, datasets 5.1.0, bitsandbytes 0.50.2. Bản pip freeze của GPU nằm trong environment_t4.txt; không ghi đè environment.txt đã được manifest hash.

## 4. Thiết kế bốn run

Tất cả dùng cùng corpus/split, assistant-only mask, hai epoch, batch hiệu dụng 1×16=16 và **max_steps=30**. Baseline được đo trước training. Trong NB4, trường mask_mode của ba dòng CSV để trống, nhưng source và log `colab_nb4.txt` xác nhận dùng cùng MASK_MODE=assistant-only; không sửa CSV để bổ sung một giá trị chưa được ghi lúc chạy.

| Run | Placement | Rank / alpha | LR | Base 4-bit | Params trainable | Final train loss | Train s | Peak VRAM GB | Target NB5 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| correct | text-linear | 16 / 32 | 0.0001 | False | 32,464,896 | 0.6256 | 416.3 | 8.78 | 0.970 |
| attn_only | attn-only | 283 / 566 | 0.0001 | False | 32,456,704 | 0.5373 | 266.8 | 8.79 | 0.970 |
| wrong_lr | text-linear | 16 / 32 | 1e-05 | False | 32,464,896 | 1.5702 | 397.0 | 8.78 | 0.000 |
| qlora | text-linear | 16 / 32 | 0.0001 | True | 32,464,896 | 0.7058 | 468.1 | 3.86 | 0.940 |

Nguồn bảng: runs.csv và autopsy.json. Precision fp16 nói về base/compute; header của cả bốn safetensors cho thấy adapter được lưu ở F32. Adapter correct có 32.464.896 tham số, không phải toàn bộ trọng số model. Train seconds chỉ đo trainer.train(), không gồm tải model; NB3 toàn stage mất khoảng 478 giây.

Correct gắn vào text-linear, loại vision tower. Attn_only chỉ gắn q,v; matched_rank nâng r lên 283 và alpha lên 566 để giữ alpha/r=2, bù số lượng vị trí ít hơn. Sai lệch ngân sách so với correct chỉ **0,025233%**, thấp hơn 5%. Rank và alpha là điều chỉnh để giữ ngân sách, không phải một sweep rank độc lập. Wrong_lr giữ placement/rank nhưng giảm LR 10 lần. QLoRA giữ cấu hình adapter và LR, chuyển base sang NF4 4-bit; notebook chấm adapter này trên base 4-bit đúng như lúc train.

### Placement, rank và learning rate

Xếp hạng theo target: **correct = attn_only (0,970) > qlora (0,940) > wrong_lr (0,000)**. Không có bằng chứng text-linear thắng attention-only trên tác vụ và ngân sách này. Attn_only có loss 0,5373 thấp hơn correct 0,6256 nhưng target chỉ hòa; đây là ví dụ cụ thể cho việc loss thấp hơn không tự chứng minh năng lực tốt hơn. Muốn kết luận rank là đòn bẩy, cần sweep r trên cùng placement và recipe, chưa có trong core này.

Wrong_lr cho loss 1,5702, target 0 và format 0, khác rõ correct dưới cùng ngân sách step đã cấu hình. Điều này phù hợp với LR quá nhỏ cho LoRA trong recipe hai epoch này. Chưa chứng minh LR 1e-4 tối ưu toàn cục; target=0 dưới scorer JSON không có nghĩa model hoàn toàn không hiểu ngôn ngữ.

### Chi phí của QLoRA

QLoRA giảm peak VRAM từ 8,78 xuống 3,86 GB (khoảng 56,0%), đổi lại target giảm 3 điểm phần trăm và train time tăng từ 416,3 lên 468,1 giây. Đối chứng này minh họa tradeoff bộ nhớ/chất lượng trên lần chạy hiện tại, không kết luận QLoRA luôn kém. Các contrast không được chạy regression đầy đủ tại NB5, nên chưa biết forgetting của từng contrast.

Các log train có một số grad_norm=nan. Loss cuối đều finite; kiểm tra toàn bộ giá trị trong bốn adapter đã lưu không có NaN hoặc infinity, theo adapter_finiteness.json. Không thể từ đó khẳng định mọi update fp16 đều đã thực hiện: GradScaler có thể bỏ qua update không hữu hạn và log không lưu tổng số skipped updates. Vì vậy 30 là ngân sách max_steps và số step được trainer log, không phải bằng chứng độc lập rằng cả bốn có cùng số update hữu hiệu. Giữ nguyên cảnh báo thay vì xóa log hay tự coi train hoàn toàn ổn định.

## 5. Bốn nhóm đánh giá và phán quyết

| Phương án | Target | Regression | Format | Latency ms/item |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3295.3 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1024.6 |
| (c) LoRA fine-tune | 0.9700 | 0.5222 | 1.0000 | 1355.8 |

Nguồn: baselines_frozen.json và verdict.json. Số hiển thị ở comparison được làm tròn; baseline regression chính xác là 0,7911111111 và FT regression suy từ delta đã lưu là 0,5222222222. Baseline (b) vượt (a) trên target trước training. Output (a) được lưu cho thấy model trả lời diễn giải dạng văn bản thay vì JSON bốn trường; vì vậy điểm target/format bằng 0 không chứng minh base thiếu hiểu biết về nội dung ticket. Hai baseline đo regression với cùng instruction và không có system prompt nên đạt cùng điểm regression.

**Verdict: FAILED.** Target tăng tuyệt đối 0,205, format giữ 1,0, nhưng regression giảm 0,268888889, vượt tolerance 0,020 của gate. Correct phải đạt đồng thời điều kiện tác vụ và giới hạn suy giảm năng lực chung; lợi ích phân loại không bù được vi phạm hồi quy theo luật đã đóng băng. Cổng được tính lại từ metric đã ghi, không thay tolerance để chuyển FAIL thành PASS. Latency tăng khoảng 32,3% so với (b), dù prompt của FT ngắn hơn; đây là quan sát của lần đo, chưa đủ để quy toàn bộ chênh lệch cho adapter vì có nhiễu GPU và không có repeated latency runs. Regression chỉ gồm 15 câu chấm bằng keyword recall, nên kết quả là dấu hiệu suy giảm trong bộ kiểm tra này, không khái quát thành model mất mọi năng lực chung. Cần đọc raw completions và thí nghiệm replay riêng trước khi quyết định triển khai. NB5 đã đo regression FT nhưng không lưu từng completion, vì vậy chưa thể minh họa các ca thua regression ở mức từng câu từ bộ artifact hiện có. Không tự tạo output đó từ điểm trung bình.

## 6. Ví dụ định tính trên cùng ticket và nhãn

Toàn bộ 50 cặp: 33 thắng, 17 hòa và **0 thua** theo delta target. Phần dưới có bảy ví dụ, gồm cả ba ticket nằm trong nhóm FT kém nhất. Một FT output sai chưa phải ca thua baseline nếu baseline cũng sai hoặc sai nhiều hơn. Yêu cầu ≥2 ca FT thua của rubric 3.4 chưa được chứng minh; không gọi các ca hòa sai bên dưới là thua. Nếu bổ sung output regression bằng một lần đo khác, phải ghi rõ nguồn/lần đo và không coi chúng là ca thua trên 50 ticket target này.

### Ticket index 0 — thắng

**Ticket:** Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.

**Nhãn:**
```json
{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
```

**Baseline (b):**
```json
{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
```

**FT correct:**
```json
{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
```

Điểm (b)=0.75; FT=1.00; delta=+0.25. FT sửa intent từ hoan_tien sang doi_tra; ba trường còn lại đều đúng.

### Ticket index 1 — thắng

**Ticket:** Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình.

**Nhãn:**
```json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}
```

**Baseline (b):**
```json
{"intent": "hoan_tien", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}
```

**FT correct:**
```json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}
```

Điểm (b)=0.75; FT=1.00; delta=+0.25. FT sửa urgency cao sang trung_binh cho cụm Sớm nhé.

### Ticket index 3 — hòa

**Ticket:** Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn:**
```json
{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```

**Baseline (b):**
```json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```

**FT correct:**
```json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```

Điểm (b)=0.75; FT=0.75; delta=+0.00. Cả hai dự đoán trung_binh thay vì thap cho Khi nào tiện. Đây là lỗi còn tồn tại nhưng là ca hòa, không phải FT thua.

### Ticket index 5 — thắng

**Ticket:** Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.

**Nhãn:**
```json
{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
```

**Baseline (b):**
```json
{"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
```

**FT correct:**
```json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
```

Điểm (b)=0.50; FT=0.75; delta=+0.25. FT sửa intent san_pham_loi nhưng vẫn sai urgency. FT chưa hoàn toàn đúng song vẫn thắng baseline từ 0,50 lên 0,75.

### Ticket index 12 — hòa

**Ticket:** Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn:**
```json
{"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}
```

**Baseline (b):**
```json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
```

**FT correct:**
```json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
```

Điểm (b)=0.75; FT=0.75; delta=+0.00. Cả hai sai urgency giống nhau; điểm 0,75 và delta=0. Ca này không được dùng để đáp ứng hai ca thua.

### Ticket index 47 — thắng

**Ticket:** Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt.

**Nhãn:**
```json
{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}
```

**Baseline (b):**
```json
{"intent": "hoi_thong_tin", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}
```

**FT correct:**
```json
{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}
```

Điểm (b)=0.75; FT=1.00; delta=+0.25. FT phân biệt van_chuyen với hoi_thong_tin khi có Shipper không gọi.

### Ticket index 49 — thắng

**Ticket:** Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé. Shop xem giúp.

**Nhãn:**
```json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}
```

**Baseline (b):**
```json
{"intent": "san_pham_loi", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}
```

**FT correct:**
```json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}
```

Điểm (b)=0.50; FT=1.00; delta=+0.50. FT sửa urgency và sentiment, tăng từ hai lên bốn trường đúng.

## 7. Kết luận

Thí nghiệm cho thấy fine-tuning đã học được quy ước phân loại ticket tốt hơn base chỉ dùng prompt tối ưu trong phạm vi tập target của lab. Điểm trung bình bốn trường tăng từ 0,765 lên 0,970, trong khi format đạt 1,0 ở cả hai phương án. Việc so cùng ticket và nhãn giúp xác định cải thiện nằm ở đâu: các ví dụ sửa intent hoặc urgency cho thấy lợi ích cụ thể; các ca Khi nào tiện vẫn sai cho thấy hành vi chưa hoàn hảo. Tuy nhiên, không nên chuyển trực tiếp từ kết quả này sang quyết định triển khai. Trên 15 instruction regression, điểm keyword recall giảm từ khoảng 0,7911 xuống 0,5222, vượt xa mức suy giảm được gate cho phép. Dữ liệu train chỉ hướng tới JSON CSKH là một giải thích phù hợp với hiện tượng quên năng lực chung, nhưng thí nghiệm chưa cô lập cơ chế nội tại hoặc loại hết ảnh hưởng của giới hạn sinh và scorer. Vì vậy kết luận đúng theo giao thức đã chốt là FAILED, kể cả khi target cao và adapter đã lưu thành công.

Đối chứng cùng ngân sách cũng làm giới hạn suy luận rõ hơn. Attention-only dùng rank cao để khớp số tham số nhưng chỉ hòa correct trên target; không thể lấy loss thấp hơn làm bằng chứng cấu hình đó tốt hơn, cũng chưa thể khẳng định placement text-linear luôn vượt trội. Giảm LR mười lần làm target và format về 0 trong recipe hiện tại, cho thấy LR đáng được kiểm tra trước khi chỉ tăng rank. QLoRA giảm VRAM rõ rệt nhưng đổi một phần chất lượng và thời gian. Bước nghiên cứu tiếp theo nên là một thí nghiệm riêng với 1–5% replay dữ liệu phổ thông, cùng đánh giá target/regression đã đóng băng và lưu raw completion từng câu. Cần thêm repeated runs để kiểm tra độ bền kết quả, ghi revisions và skipped updates để tăng khả năng tái lập. Không sửa baseline, scorer hay eval của core này để tạo verdict đẹp hơn. Thiếu hai ca thua target cũng cần được công bố đúng, thay vì gọi lỗi còn tồn tại là ca thua hoặc chọn dữ liệu mới sau kết quả rồi trộn vào lần đo cũ.

## 8. Điều học được và cách thực hiện

Trong bài này tôi trực tiếp chạy các mục 1–7 trên Colab và tải checkpoint trước training, sau correct, sau contrasts và cuối core. Việc giữ checkpoint trước training giúp đối chiếu mốc baseline ngay cả khi runtime đã ngắt; các adapter còn trong ZIP giúp tránh phải huấn luyện lại chỉ để đọc kết quả.

Điều có thể rút ra từ số thực là cần tách thành công của pipeline khỏi hiệu quả của model. NB3 lưu adapter và NB5 chạy xong không đồng nghĩa model vượt gate. Cũng cần tách thang đo: 97% trường đúng chỉ tương ứng 88% ticket đúng trọn vẹn; loss 0,5373 của attn_only không chuyển thành target cao hơn correct. Khi viết ví dụ, tôi phải kiểm tra delta với baseline thay vì chỉ nhìn FT sai; ticket index 3 và 12 chứng minh vì sao gọi chúng là ca thua sẽ sai.

Tôi dùng trợ lý AI để lập kế hoạch, bổ sung mã lưu evidence, kiểm tra checkpoint và biên tập report. Việc chạy GPU và tải checkpoint do tôi thực hiện; các số liệu lấy từ artifact, không dùng số demo của repo. Trong lần làm sau, tôi sẽ chốt việc lưu raw regression ngay trước khi chạy, giữ revisions cụ thể và theo dõi update bị bỏ qua. Phần phản tư này được soạn từ các thao tác và kết quả đã có; học viên cần đọc lại, chỉnh cách diễn đạt và xác nhận nó phản ánh đúng điều mình hiểu trước khi nộp.

## 9. Kiểm tra, hạn chế và định dạng bàn giao

Colab smoke ghi **119 original tests passed**, không có skip. Kiểm thử cục bộ trong CPU venv gồm bộ gốc và student tests trước bàn giao: **175 passed, 3 skipped** (ba test cần PyTorch), theo prepared_tests.txt. Verify đầy đủ sau khi hoàn thiện report trả **26 passed, 1 warning, 0 failures**, exit 0; output lưu ở verify.txt. Bộ student tests cuối có **59 passed**, theo student_tests_final.txt. Cảnh báo duy nhất của verify là verdict model FAILED, được giải thích như trên. Verify không kiểm tra đủ yêu cầu hai ca thua của rubric 3.4; không suy ra đủ toàn bộ rubric chỉ từ exit 0.

Các kiểm tra độc lập đã re-score raw baseline, từng cặp target và summary; kiểm tra full eval, step budget, ngân sách params, adapter provenance, safetensors header và toàn bộ giá trị trọng số. Raw FT regression không có nên không tuyên bố đã re-score metric đó từ completion; mới đối chiếu aggregate trong log/verdict. Những hạn chế revision=null, grad_norm=nan và thiếu hai ca target thua được giữ công khai.

Gói core chọn Option A: report, toàn bộ results, hai file adapter correct và notebook nguồn không chứa output. Adapter correct khoảng 129,93 MB thô (F32), nên ZIP lớn hơn ví dụ 5–15 MB trong rubric; đó là dung lượng thực, không phải full merged weights. Các adapter đối chứng giữ ngoài gói core nhưng có hash và bằng chứng trong results. Không đưa .env, token, .venv, model cache hoặc merged model vào ZIP. B1–B5 chưa có bằng chứng hoàn thành và không yêu cầu điểm thưởng. Gói local và nhánh GitHub không tự chứng minh đã nộp LMS.

## 10. Chuẩn bị bổ sung trước hạn 23:30

Mã bổ sung và notebook phục hồi đã được kiểm tra/review, nhưng chưa chạy GPU. Bộ CPU gồm
**201 passed, 3 skipped**, lưu tại results/supplementary_tests_cpu.txt; các skip cần PyTorch.
Helper đo 15 regression, full50 trước/sau merge và hai adapter trên cùng base; rank sweep
r8/r16/r64 giữ recipe riêng. Baseline, adapter và verdict core không bị ghi đè.

[Notebook bổ sung](https://colab.research.google.com/github/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab/blob/feature/lab21-finetuning/colab/Lab21_Deadline_Followup.ipynb). Kết quả mới, nếu có, phải được kiểm tra và ghi rõ là supplementary repeat.

Bản nháp dataset CSKH giáo dục giả lập nằm ở bonus_data/education_support, với pointer
data/CUSTOM_DATASET.md: **240 train + 60 eval**, tách **60/15 nhóm tình huống**, không có
trùng input chuẩn hóa nội bộ, giữa split hoặc với core đã kiểm tra. quality_audit.json lưu
schema, hash, kiểm tra từ vựng và đo token local. Đây là dữ liệu synthetic có câu nhãn lặp,
chưa có thẩm định chất lượng độc lập hoặc phép đo GPU trên miền này; không tự nhận B2 đã đạt.

Model card và adapter correct để chuẩn bị Hub đã tạo ở output/huggingface_adapter;
receipt nằm trong submission/hf_preparation_receipt.json. Chưa upload Hub, chưa có link
B5; B3 chưa thực hiện. Các công việc chuẩn bị này không chuyển thành điểm thưởng measured.
Trợ lý chưa chạy được Colab vì công cụ browser không khởi tạo; cần học viên mở notebook T4
và cung cấp các ZIP mới. Không tuyên bố đã nộp LMS hoặc hoàn thành toàn bộ rubric.
