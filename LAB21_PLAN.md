# Lab Day 21 — Kế hoạch học và hoàn thiện bài fine-tuning

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Kế hoạch này chưa triển khai training. Khi thực hiện trong chat hiện tại, đi lần lượt và báo ngắn sau mỗi checkpoint. Không tự publish/push adapter hoặc mã nguồn khi chưa có yêu cầu tương ứng.

**Goal:** Hoàn thành toàn bộ core NB1–NB5, thí nghiệm công bằng, bằng chứng đủ truy vết, report tự viết và gói nộp được kiểm tra theo rubric; có lộ trình riêng cho tất cả năm mục thưởng.

**Architecture:** Giữ pipeline có sẵn của lab, cùng base model và tập eval cho mọi run trong cùng thí nghiệm. NB1 chứng minh mask; NB2 đo và đóng băng baseline trước training; NB3/NB4 huấn luyện bốn run có ngân sách công bằng; NB5 đánh giá và phán quyết. Chỉ bổ sung lưu output và thông tin tái lập nếu cần, không thay đổi scorer hoặc gate để tạo kết quả tốt hơn.

**Tech Stack:** Python, Hugging Face Transformers, PEFT/LoRA, TRL/SFTTrainer, datasets, PyTorch/CUDA, bitsandbytes cho đối chứng QLoRA, pytest, Colab Free T4.

**Spec:** [README](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab/blob/d27c1c02ebe99f32f52f706be88b4c30fb1d7fca/README.md), [rubric](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab/blob/d27c1c02ebe99f32f52f706be88b4c30fb1d7fca/rubric.md), [hardware](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab/blob/d27c1c02ebe99f32f52f706be88b4c30fb1d7fca/HARDWARE-GUIDE.md), [bonus](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab/blob/d27c1c02ebe99f32f52f706be88b4c30fb1d7fca/BONUS-CHALLENGE.md).

Nguồn được đọc trực tiếp ngày 07/10/2026, revision `d27c1c02ebe99f32f52f706be88b4c30fb1d7fca`.
Bản tham khảo đã tải tại `C:/Users/SAM/IdeaProjects/KTThuatToan/untitled/output/day21-reference/Day21-Track3-Finetuning-Lab-d27c1c02ebe99f32f52f706be88b4c30fb1d7fca/`.
Bản này phục vụ đọc đề; tạo checkout bài lab riêng khi triển khai, không trộn vào bài KTThuatToan hiện tại.

## Global Constraints

- Core là NB1–NB5; NB6 tùy chọn. Rubric: 30 + 25 + 25 + 20 = 100, thưởng tối đa +15.
- Cùng base model cho baseline và fine-tune. Khai báo model, dataset và lý do lựa chọn.
- Đo baseline (a)/(b) và đóng băng trước khi train; (c) là fine-tune được đo sau, tại NB5. Cụm “ba baseline trước train” trong tài liệu không có nghĩa train trước để đo (c).
- Không đổi tập eval, prompt baseline đã đóng băng, scorer hoặc ngưỡng sau khi thấy kết quả để làm FT trông thắng.
- `attn_only` phải khớp số tham số trainable với `correct`: sai lệch <5%; rank do `matched_rank()` tính.
- Bốn run `correct`, `attn_only`, `wrong_lr`, `qlora` phải có cùng `max_steps`.
- Xếp hạng bốn run bằng target tại NB5, không bằng final training loss.
- Mẫu nộp đầy đủ phải bỏ `EVAL_LIMIT`; dùng `EPOCHS=2` mặc định. Chế độ rút gọn chỉ để thử pipeline, phải lưu tách biệt.
- Kết luận >=150 từ; >=5 ví dụ định tính, trong đó >=2 ca FT thua; phản tư cụ thể. Số liệu phải khớp `results/`.
- Không đưa `.env`, token, cache model, môi trường ảo hoặc full merged weights vào gói core.
- PASS/FAIL của model khác PASS/FAIL của kiểm tra bài nộp. Model verdict FAILED vẫn được chấm đầy đủ khi phân tích đúng.

## Kiến thức cần nắm và tiêu chuẩn tự kiểm tra

| Kiến thức | Giải thích ngắn | Tự kiểm tra trước khi thực hành |
|---|---|---|
| SFT | Học từ cặp input → output chuẩn | Biết vì sao đây là supervised fine-tuning, không phải DPO/RL |
| LoRA | Đóng băng trọng số gốc, học phần cập nhật nhỏ `ΔW=(alpha/r)BA` | Phân biệt base model, adapter, rank, alpha, vị trí gắn |
| QLoRA | Base được lượng tử hóa 4-bit; adapter vẫn được học | Biết giảm VRAM không đảm bảo giữ nguyên chất lượng |
| Chat template | Tokenizer biến system/user/assistant thành chuỗi token đúng chuẩn model | Đọc chuỗi render thật, kiểm tra EOS và `<think>` |
| Loss mask | Label `-100` không đóng góp loss; chỉ tính loss phần muốn model học | Decode phần supervised và chứng minh có answer, không có question |
| Độ dài và batching | p95 giúp chọn độ dài; batch hiệu dụng = batch × gradient accumulation, với một GPU | T4 mặc định 1 × 16 = 16; phân biệt optimizer step với microbatch |
| Thiết kế thí nghiệm | Giữ ngân sách và các yếu tố còn lại, thay một yếu tố nghiên cứu | Giải thích vì sao cùng rank chưa chắc cùng số tham số |
| Đánh giá và forgetting | Target tốt hơn chưa đủ nếu năng lực phổ thông tụt | Đọc được bốn nhóm metric và gate hiện tại |
| Reproducibility | Seed, model/tokenizer revision, phiên bản gói, prompt, dữ liệu, cấu hình | Truy được một con số trong report về artifact và lần chạy |
| Merge và serving | Merge adapter vào base hoặc giữ nhiều adapter trên một base | Chỉ cần cho NB6/bonus; không cần xây server riêng để hoàn thành core |

Tài liệu ưu tiên: README → rubric → NB1/NB2 → config/modeling/train/evaluate → các notebook còn lại. Deck được README tham chiếu nhưng không có trong bản repo tải về; kế hoạch không giả định đã đọc deck.

## Bản đồ file và đầu ra

Đường dẫn dưới đây tính từ checkout bài lab sẽ triển khai.

| File | Vai trò | Hành động dự kiến |
|---|---|---|
| `.env.example`, `.env` | Tier, model, mask, epochs | Copy và đặt cấu hình; không commit `.env` |
| `notebooks/01_data_and_mask.py` | Template, mask, p95, split | Chạy, đọc và giải thích output |
| `notebooks/02_baselines.py` | Base + hai prompt, frozen baseline | Bổ sung lưu prediction ngay tại lần đo trước train |
| `notebooks/03_train_correct.py` | Run LoRA chính | Chạy cấu hình có sẵn; giữ output train |
| `notebooks/04_misconfig_autopsy.py` | Ba run đối chứng | Chạy, kiểm tra rank/params/steps, resume đúng cấu hình |
| `notebooks/05_evaluate_and_verdict.py` | Score, autopsy, qualitative | Bổ sung bằng chứng paired baseline (b) vs FT |
| `notebooks/06_merge_and_serve.py` | Merge, hot-swap | Chỉ thực hiện sau core để lấy B1 |
| `src/labkit/config.py` | Tier, LoRA specs và hai prompt | Chỉ đọc; nếu đổi prompt/model/length thì quyết định trước NB2 và khai báo |
| `src/labkit/data.py`, `modeling.py`, `train.py` | Mask, placement, training | Giữ logic chính; sửa lỗi chỉ khi có bằng chứng tái hiện |
| `src/labkit/evaluate.py` | Metric và gate | Giữ scorer/ngưỡng của lab |
| `tests/`, `scripts/verify.py`, `data/checksums.json` | Kiểm tra tham chiếu | Không sửa để ép bài pass |
| `submission/REPORT.md` | Report tự viết | Viết theo kết quả thực, không chỉ giữ mẫu |
| `submission/REFLECTION.md` | Câu hỏi phản tư bổ trợ | Trả lời nếu dùng; rubric chấm phản tư trong report |
| `results/experiment_manifest.json` | Artifact bổ sung | Cấu hình, revision, hash, UTC time trước train |
| `results/baseline_predictions.json` | Artifact bổ sung | Raw outputs (a)/(b), nhãn và điểm từng mẫu |
| `results/paired_qualitative.json` | Artifact bổ sung | Raw output/score của (b) và FT trên cùng ticket |
| `results/environment.txt`, `results/verify.txt` | Artifact bổ sung | Phiên bản gói và kiểm tra cuối |

## Task 0: Chốt môi trường và lịch làm — 20–40 phút

**Files:** đọc README/rubric/hardware/requirements; tạo `.env` trong checkout bài lab mới.
**Consumes:** URL bài lab, khả năng có GPU, thời hạn nộp.
**Produces:** môi trường CPU hoặc T4 dùng được, model/config được xác nhận, lịch chạy đủ thời gian.

- [ ] Chọn phương án mặc định: corpus sẵn có, T4, `unsloth/Qwen3.5-4B`. Lựa chọn này giảm biến số để tập trung hoàn thành core.
- [ ] Tạo checkout bài lab riêng. Nếu dùng Colab, mở RUN_ALL từ GitHub bằng tab mới; chọn T4 GPU, chạy Setup.
- [ ] Đổi `EVAL_LIMIT = "8"` trong ô Core thành `EVAL_LIMIT = ""`. Không giữ mặc định smoke khi chạy bản chính thức.
- [ ] Đặt `COMPUTE_TIER=T4`, `MASK_MODE=assistant-only`, `EPOCHS=2`; không đặt `BASE_MODEL` nếu giữ model mặc định.
- [ ] Kiểm tra GPU, precision thực tế. T4 dùng fp16 vì không hỗ trợ bf16 native; không ép `bf16=True`.
- [ ] Chạy smoke; nếu lỗi, sửa môi trường trước khi tải/chạy model train. Không cần API OpenAI/Gemini cho pipeline mặc định; HF token chỉ tùy chọn.
- [ ] Lưu revision của checkout và phiên bản gói thực tế. Repo dùng khoảng phiên bản, không phải lockfile tuyệt đối.

Colab sau Setup, chạy trong một ô:

```python
import os, subprocess, sys
os.environ['COMPUTE_TIER'] = 'T4'
os.environ['MASK_MODE'] = 'assistant-only'
os.environ['EPOCHS'] = '2'
os.environ.pop('EVAL_LIMIT', None)
subprocess.run([sys.executable, 'scripts/verify.py', '--smoke'], check=True)
```

Windows PowerShell cho phần CPU, chạy từ thư mục cha bạn chọn cho lab:

```powershell
git clone https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab.git
Set-Location Day21-Track3-Finetuning-Lab
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
Copy-Item .env.example .env
$env:COMPUTE_TIER = 'CPU'
.\.venv\Scripts\python.exe scripts/verify.py --smoke
.\.venv\Scripts\python.exe notebooks/01_data_and_mask.py
```

NB1 CPU dùng tokenizer model tier CPU. Trước bản training T4 phải chạy lại NB1 ở T4 để proof là của đúng tokenizer/model dùng cho thí nghiệm. Makefile sử dụng `.venv/bin`, grep/awk và shell Linux; không bê nguyên lệnh `make setup` sang PowerShell.

**Checkpoint:** smoke exit 0; model/tier/precision in đúng; chưa train. Thời gian core do repo báo khoảng 100–130 phút T4, là số đo lịch sử của tác giả, không phải đảm bảo tốc độ máy của bạn. Dành 4–6 giờ cho cả chuẩn bị, kiểm tra, report và dự phòng.

## Task 1: NB1 — Chứng minh dữ liệu và mask — 15–30 phút đọc/kiểm tra

**Files:** chạy `notebooks/01_data_and_mask.py`; sinh `results/{mask_proof,template_check,token_stats}.json`, `data/split/{train,val}.jsonl`.
**Consumes:** tokenizer của base model chính thức, corpus.
**Produces:** bằng chứng mask, template, độ dài và split dùng cho NB3/NB4.

- [ ] Đọc 3–5 ticket và nhãn để hiểu bốn trường: intent, urgency, product, sentiment.
- [ ] Kiểm đếm corpus mặc định: 250 train_seed; 50 eval_target; 15 eval_regression. NB1 chia train_seed thành 225 train, 25 val bằng seed 42; nếu tokenizer loại mẫu không supervised thì ghi số train thực dùng.
- [ ] Chạy NB1. Decode phần được tính loss, so `assistant-only` với `everything` để nhìn thấy lỗi train cả prompt.
- [ ] Kiểm tra `answer_is_supervised=true`, `question_is_masked=true`, `supervised_fraction<0.95`.
- [ ] Đọc chuỗi template thật và verdict có giữ `<think>` không; ghi observation cụ thể.
- [ ] Ghi p95 và suggested_max_length. Code chỉ gợi ý, không tự thay `TIER.max_length`. Nếu giữ T4=1024 khác p95 thì giải thích lựa chọn và kiểm tra không cắt answer; nếu đổi, thực hiện trước NB2 và dùng thống nhất mọi run.
- [ ] Nếu đổi base, chạy thêm `python scripts/check_mask_agreement.py` trước train.

```python
subprocess.run([sys.executable, 'scripts/colab_run.py', 'nb1'], check=True)
```

**Checkpoint:** ba JSON tồn tại và proof đúng; split đúng seed; đủ giải thích template/p95. Chưa đạt thì chưa train.

## Task 2: NB2 — Đo baseline và giữ bằng chứng trước train — 17–23 phút GPU + kiểm tra

**Files:** bổ sung `notebooks/02_baselines.py`, sinh `baselines_frozen.json`, `baseline_predictions.json`, `experiment_manifest.json`, `environment.txt`.
**Consumes:** base model chưa fine-tune, eval đầy đủ, prompt (a)/(b), cấu hình đã chốt.
**Produces:** baseline đóng băng và raw prediction để NB5 phân tích thắng/thua.

- [ ] Đảm bảo không bật EVAL_LIMIT và chưa có run train trong thí nghiệm này.
- [ ] Trước lần chạy, bổ sung lưu các prediction NB2 đã sinh; không cần chạy thêm inference để lấy raw output.
- [ ] Chạy NB2; kiểm tra frozen n_target=50, n_regression=15, smoke_mode=false với corpus mặc định.
- [ ] Xác nhận (b).target > (a).target. Nếu chưa, cải thiện prompt trên train/val, khai báo thay đổi và đo lại trước training; không tinh chỉnh theo lỗi eval/holdout.
- [ ] Lưu manifest: model ID và resolved revision nếu lấy được, tokenizer, commit, seed, tier, precision, epochs, length, eval count, hash dữ liệu/prompt, thời điểm UTC. Lưu cùng config trước train và không ghi đè ở bước sau.
- [ ] Lưu artifact ra nơi bền vững để không mất khi Colab recycle. Giữ baseline chính thức, không trộn với smoke.

Ngay sau hai lệnh `score_run` trong NB2, bổ sung:

```python
report.write_json([
    {
        'i': i, 'ticket': row['input'], 'label': row['label'],
        'baseline_a_pred': pa, 'baseline_b_pred': pb,
        'baseline_a_score': ev.triage_field_accuracy(pa, row['label']),
        'baseline_b_score': ev.triage_field_accuracy(pb, row['label']),
    }
    for i, (row, pa, pb) in enumerate(zip(target, preds_a, preds_b))
], 'baseline_predictions.json', results_dir=ROOT / 'results')
```

```python
subprocess.run([sys.executable, 'scripts/colab_run.py', 'nb2'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
               stdout=open('results/environment.txt', 'w', encoding='utf-8'), check=True)
```

**Checkpoint:** baseline và raw output tồn tại, (b) mạnh hơn (a), hash/config được lưu trước training. Không giả định verify chứng minh được thứ tự thời gian vì code hiện tại không lưu timestamp cho việc đó.

## Task 3: NB3 — Train LoRA chính — 15–25 phút GPU + kiểm tra

**Files:** chạy `notebooks/03_train_correct.py`; sinh `adapters/correct/`, thêm dòng `correct` trong `results/runs.csv`.
**Consumes:** split và mask đã proof, base model đúng, baseline đã frozen.
**Produces:** adapter chính cùng loss, VRAM, cấu hình, thời gian và số step.

- [ ] Đọc layer_types và module placement thật từ model. Dùng text-linear, không gắn vào vision tower.
- [ ] Kiểm tra r=16, alpha=32, LR=1e-4, batch hiệu dụng=16, packing=False với mặc định T4.
- [ ] Giữ cùng mask đã proof. Không thay bằng cờ thư viện mà không kiểm tra template/mask.
- [ ] Chạy NB3; giữ log train, ghi nhận loss và precision. Nếu nan kéo dài suốt run thì xem là lỗi cần xử lý, không dùng loss đó để kết luận.
- [ ] Kiểm tra adapter_model.safetensors và adapter_config.json được lưu; runs.csv có correct, trainable_params, final_loss, max_steps, peak_vram_gb.
- [ ] Với 225 mẫu không bị loại, batch=16, epochs=2: dự kiến 30 optimizer steps; lấy giá trị thực trong runs.csv làm bằng chứng.

```python
subprocess.run([sys.executable, 'scripts/colab_run.py', 'nb3'], check=True)
```

**Checkpoint:** adapter nạp lại được, log không cho thấy run chết, thông tin train đầy đủ. Không kết luận quality chỉ từ loss.

## Task 4: NB4 — Ba đối chứng công bằng — 45–60 phút GPU + kiểm tra

**Files:** chạy `notebooks/04_misconfig_autopsy.py`; sinh ba adapter và các dòng tương ứng trong runs.csv.
**Consumes:** cùng base, split, mask, epochs, length như correct.
**Produces:** bốn run so sánh được về chất lượng, thời gian và VRAM.

| Run | Khác correct | Điều cần xác nhận |
|---|---|---|
| correct | Run tham chiếu | text-linear, r16, LR1e-4, base 16-bit |
| attn_only | Vị trí gắn; rank điều chỉnh để giữ budget | rank runtime, params sai lệch <5% |
| wrong_lr | LR giảm từ 1e-4 xuống 1e-5 | Các yếu tố còn lại giữ nguyên |
| qlora | Base 4-bit thay base 16-bit | Cùng LoRA spec; đánh giá lại với base 4-bit |

- [ ] Chạy NB4, không tự cố định rank của attn_only từ comment: rank phụ thuộc kiến trúc thực.
- [ ] Kiểm tra cả bốn max_steps bằng nhau và ngân sách attn_only lệch <5%.
- [ ] Ghi VRAM/time thực, không dùng số trong docs/MEASURED làm kết quả cá nhân.
- [ ] Nếu runtime bị đứt: phục hồi artifact trước, chạy lại chỉ run thiếu; kiểm tra adapter tồn tại có đúng model/config/steps của lần này. Cơ chế skip dựa vào file tồn tại, không tự chứng minh cấu hình còn khớp.
- [ ] Với run chưa hoàn tất, ONLY chọn run cần làm. Muốn train lại run đã lưu phải thêm FORCE_RETRAIN=1; ONLY một mình không ép train lại.
- [ ] Không đổi epochs giữa NB3 và NB4. Nếu phải đổi cấu hình, lưu một thí nghiệm riêng và train lại mọi run liên quan.

```python
subprocess.run([sys.executable, 'scripts/colab_run.py', 'nb4'], check=True)
# Chỉ dùng khi cần làm lại qlora:
subprocess.run([sys.executable, 'scripts/colab_run.py', 'nb4'],
    env={**os.environ, 'ONLY': 'qlora', 'FORCE_RETRAIN': '1'}, check=True)
```

**Checkpoint:** đủ bốn run và adapter; cùng max_steps, params match, log/config đầy đủ. Chưa xếp hạng bằng final_loss.

## Task 5: NB5 — Đánh giá, verdict và ví dụ thật — khoảng 21 phút GPU + phân tích

**Files:** bổ sung `notebooks/05_evaluate_and_verdict.py`; sinh verdict.json, autopsy.json, qualitative.json, paired_qualitative.json.
**Consumes:** frozen baseline, raw prediction (b), correct và ba adapter đối chứng.
**Produces:** phán quyết có bằng chứng và >=5 ví dụ định tính theo rubric.

- [ ] Đánh giá correct cùng full eval như NB2. FT dùng prompt ngắn (a), baseline (b) dùng prompt tối ưu: đây là thiết kế chủ định của lab.
- [ ] Đọc bốn nhóm: target=trung bình tỷ lệ 4 field đúng; regression=keyword recall trên 15 instruction; format=trung bình tỷ lệ key bắt buộc có mặt sau loose JSON parse; latency=ms/mẫu trong chế độ generate batch của lab.
- [ ] Không diễn giải format=1 như bảo đảm JSON thuần tuyệt đối: scorer có thể lấy object nằm trong văn bản/fence. Giới hạn này nên ghi nếu ảnh hưởng kết luận.
- [ ] Đọc gate hiện tại: target_delta>0 và regression_delta>=-0.02. Format/latency được báo cáo nhưng không trực tiếp quyết định boolean PASS/FAIL.
- [ ] Autopsy có đủ correct + attn_only + wrong_lr + qlora; xếp hạng target, đối chiếu training loss.
- [ ] Bổ sung bảng ghép prediction (b) trước train với FT trên cùng i/ticket, có nhãn và điểm. qualitative.json có sẵn chỉ sắp theo ft_score, chưa chứng minh thua (b).
- [ ] Chọn ít nhất 5 ví dụ, ít nhất 2 ca delta<0; khuyến nghị thêm 2 ca delta>0 nếu có và một ca hòa/khó. “FT sai” chưa tự động là “FT thua baseline”.
- [ ] Nếu không có đủ 2 ca FT thua, báo rõ số thực tế và điểm rubric chưa được chứng minh; không bịa hoặc thay eval để tạo ví dụ. Có thể bổ sung một tập challenge khai báo riêng, giữ nguyên bảng core, nhưng cần xác nhận với giảng viên việc dùng tập bổ sung cho mục này.

Sau khi sinh scores_ft/preds_ft trong NB5, thêm:

```python
baseline_rows = json.loads(
    (ROOT / 'results' / 'baseline_predictions.json').read_text(encoding='utf-8'))
assert len(baseline_rows) == len(target) == len(preds_ft)
paired = []
for i, (row, bp, fp) in enumerate(zip(target, baseline_rows, preds_ft)):
    assert bp['i'] == i and bp['ticket'] == row['input']
    assert bp['label'] == row['label']
    bscore = ev.triage_field_accuracy(bp['baseline_b_pred'], row['label'])
    fscore = ev.triage_field_accuracy(fp, row['label'])
    paired.append({
        'i': i, 'ticket': row['input'], 'label': row['label'],
        'baseline_b_pred': bp['baseline_b_pred'], 'ft_pred': fp,
        'baseline_b_score': bscore, 'ft_score': fscore, 'delta': fscore - bscore,
    })
report.write_json(paired, 'paired_qualitative.json', results_dir=ROOT / 'results')
print('FT losses:', sum(row['delta'] < 0 for row in paired))
print('FT wins:', sum(row['delta'] > 0 for row in paired))
```

```python
subprocess.run([sys.executable, 'scripts/colab_run.py', 'nb5'], check=True)
```

**Checkpoint:** full counts khớp baseline, bốn nhóm có số, autopsy đủ run, verdict có lý do; ví dụ được chọn từ paired evidence. FAILED là kết quả hợp lệ, không sửa threshold/prompt/eval để đổi thành PASSED.

## Task 6: Report và phản tư — 45–75 phút

**Files:** viết lại submission/REPORT.md; tùy chọn bổ sung REFLECTION.md.
**Consumes:** tất cả artifact của NB1–NB5.
**Produces:** report tự cấu trúc, kết luận >=150 từ, con số có nguồn và lập luận rõ.

- [ ] Tự viết cấu trúc: bài toán/lựa chọn → mask/template/length → baseline frozen → thiết kế đối chứng → kết quả → ví dụ → phán quyết → phản tư.
- [ ] Ghi model ID đúng namespace, dataset/số mẫu/seed, GPU và precision thực, max_steps và môi trường chạy.
- [ ] Bảng ba phương án (a)/(b)/(c) đủ target/regression/format/latency; giải thích sao (b) là mốc thật phải vượt.
- [ ] Bảng bốn run đủ placement/rank/params/LR/loss/target/time/VRAM; trả lời 3 câu NB4: placement vs rank, LR, VRAM vs quality QLoRA.
- [ ] Diễn giải verdict tối thiểu khoảng 100 từ theo mẫu report, giữ kết luận >=150 từ theo rubric. Nêu nhân quả có kiểm soát và giới hạn của bằng chứng.
- [ ] Có >=5 ví dụ với ticket, nhãn, output (b), output FT, điểm từng bên và nhận xét; >=2 ca thua được chứng minh.
- [ ] Viết phản tư cá nhân bằng việc thực sự quan sát: dự đoán nào sai, lỗi nào gặp, vai trò AI assistant, điều sẽ làm khác. Không điền trải nghiệm chưa xảy ra.
- [ ] Mỗi con số truy được về results; không copy kết quả demo/tác giả thành kết quả của mình.

**Checkpoint:** report không còn placeholder; đọc được mà không cần chat; đủ bốn nhóm điểm rubric. Không đồng nhất model gate PASSED với “sẵn sàng production” vì eval nhỏ và regression dùng keyword recall.

## Task 7: Nghiệm thu và đóng gói — 20–40 phút

**Files:** đọc scripts/verify.py và rubric; tạo results/verify.txt; tạo ZIP option A hoặc cấu trúc B/C theo lựa chọn.
**Consumes:** report hoàn chỉnh và toàn bộ results, adapter, notebook.
**Produces:** gói nộp có thể kiểm tra chéo, không lẫn smoke/secret/cache.

- [ ] Chạy unit tests và verify đầy đủ sau khi report hoàn chỉnh; lưu output, kiểm tra exit code.
- [ ] Đọc tất cả warning. Thiếu đối chứng, (b)<= (a), report ngắn hoặc FAILED model có thể chỉ WARN; verify exit 0 không bảo đảm đạt hết rubric.
- [ ] Kiểm tra thủ công có qualitative/paired evidence, >=5 ví dụ, >=2 ca thua, conclusion >=150 từ, đủ bốn run/autopsy. verify hiện chưa bao phủ hết các điều này.
- [ ] Kiểm tra số liệu report khớp results và không trộn artifacts từ model/config/epoch/slice khác.
- [ ] Mặc định chọn Option A: REPORT.md + tất cả results + adapters/correct (hai file adapter chính) + notebook .py hoặc .ipynb clear output.
- [ ] Dung lượng adapter lấy từ file thực, không cam kết ZIP 5–15 MB: số tham số trainable và dtype có thể làm adapter lớn hơn ví dụ trong rubric.
- [ ] Không đưa merged full weights, .env, token, model cache, .venv vào archive. Giữ logs riêng nếu có thông tin nhạy cảm; rà soát trước khi nộp.
- [ ] Giải nén ZIP vào thư mục kiểm tra và xác nhận đủ JSON parse được, runs.csv, report và adapter. Verify đầy đủ cần cả checkout/môi trường, nên chạy trước đóng gói; không giả định ZIP tối giản tự chạy verify được.
- [ ] Nếu chọn GitHub: .gitignore mặc định bỏ qua results/*.json, results/*.csv, adapters và CUSTOM_DATASET.md; chuẩn bị quy tắc track artifact cần nộp, kiểm tra git diff trước push. Không giả định git add đã đưa results lên remote.
- [ ] Xác nhận độc lập link nộp/remote khi thực sự nộp; local ZIP không chứng minh đã nộp LMS.

```python
subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q'], check=True)
checked = subprocess.run([sys.executable, 'scripts/verify.py'],
                         capture_output=True, text=True)
print(checked.stdout)
print(checked.stderr)
from pathlib import Path
Path('results/verify.txt').write_text(checked.stdout + checked.stderr, encoding='utf-8')
checked.check_returncode()
```

**Checkpoint cuối:** không còn FAIL; WARN được giải thích hoặc xử lý; checklist rubric đủ bằng chứng; ZIP và phương thức nộp đã kiểm tra.

## Lộ trình thưởng sau core

Không ghi thưởng “đã hoàn thành” trước khi có bằng chứng. Giữ core trong một bộ artifact ổn định; các thí nghiệm bổ sung ở checkout/thư mục riêng để không ghi đè adapters/correct, baselines_frozen và verdict của core.

| Thưởng | Điểm | Việc cần làm | Bằng chứng/điều kiện |
|---|---:|---|---|
| B1 merge/hot-swap | +3 | NB6; kiểm tra before/after; đổi >=2 adapter trên cùng base | merge_check.json, delta>=-0.01, log tên adapter/output; ưu tiên correct + attn_only để tránh dùng adapter QLoRA trên base không lượng tử |
| B2 dữ liệu miền riêng | +3 | >=200 mẫu chất lượng, nguồn rõ, split độc lập, khử trùng lặp/nhiễm | CUSTOM_DATASET.md và một thí nghiệm đầy đủ NB1–NB5 riêng; khai báo eval/checksum mới trước train |
| B3 reasoning traces | +4 | Hai run assistant-only/response-only trên model thinking và dữ liệu có trace thật | Chứng minh mask hai mode khác nhau; template giữ trace; inference bật thinking; giữ raw `<think>` để valid_trace_rate có nghĩa; bảng target/trace/regression |
| B4 rank sweep | +3 | text-linear cố định, r={8,16,64}, cùng LR/steps/data/mask; alpha theo 2r | Cấu hình, train logs, prediction và score riêng từng rank; so biên độ với NB4 |
| B5 HF public | +2 | Chuẩn bị adapter/model card và publish khi người dùng yêu cầu | Repo HF công khai truy cập được, có file adapter và link trong report |

B1: NB6 hiện lưu full merged model vào adapters/merged; không bỏ full model này vào ZIP core. Hot-swap QLoRA adapter trên fp16 base là minh họa phục vụ, không dùng để kết luận chất lượng của run QLoRA đã train trên 4-bit.

B3: corpus mặc định có answer JSON trần nên assistant-only, masked-think và response-only có thể cho mask giống nhau. Chỉ chạy hai lệnh đổi MASK_MODE chưa đủ chứng minh reasoning collapse. Hàm generate_batch hiện mặc định enable_thinking=False; phải thiết kế eval riêng trước thí nghiệm thinking và xác nhận decode không làm mất tag. Không so với baseline core dùng chế độ khác rồi gọi đó là cùng thí nghiệm.

Thứ tự gợi ý theo effort: B1 → B5 nếu muốn công khai → B4 → B2/B3. Để đủ +15 phải làm đủ B1–B5, nên dự trù thêm vài buổi; B2/B3 cần xây dữ liệu và kiểm tra giao thức trước khi tính giờ GPU. B6 optimizer mismatch và B7 MoE không tính điểm, không cần để hoàn thành bài nộp.

## Checklist đối chiếu rubric

| Tiêu chí | Điểm | Task/bằng chứng |
|---|---:|---|
| 1.1 mask proof | 10 | Task1, hai bool true và supervised_fraction<0.95 |
| 1.2 template | 5 | Task1 + giải thích trong report |
| 1.3 p95/length | 5 | Task1 + lựa chọn có căn cứ |
| 1.4 correct train | 10 | Task3, adapter + runs.csv loss/VRAM |
| 2.1 ngân sách params | 10 | Task4, attn_only lệch <5% |
| 2.2 cùng step | 5 | Task4, bốn max_steps bằng nhau |
| 2.3 một yếu tố | 5 | Task4/6, bảng cấu hình |
| 2.4 placement vs rank | 5 | Task5/6, target + budget |
| 3.1 baseline (b) | 5 | Task2, frozen trước train và b>a |
| 3.2 bốn nhóm | 10 | Task5, verdict/comparison |
| 3.3 giải thích verdict | 5 | Task6 |
| 3.4 định tính | 5 | Task5/6, >=5 ví dụ và >=2 thua |
| 4.1 report tự cấu trúc | 5 | Task6 |
| 4.2 kết luận >=150 từ | 5 | Task6/7 |
| 4.3 số liệu khớp | 5 | Task6/7 |
| 4.4 phản tư cụ thể | 5 | Task6 |

## Trạng thái khi bàn giao kế hoạch

Đã đọc nguồn, cấu hình, notebooks, gatekeeper, mẫu report và bonus; đã lưu bản nguồn tham khảo và kế hoạch. Chưa cài môi trường lab, chưa chạy tests/NB1, chưa train, chưa có kết quả cá nhân, chưa commit/push/publish/nộp bài. Người dùng đã xác nhận dùng Colab Free T4. Hạn nộp và lựa chọn điểm thưởng chưa được cung cấp; lịch mặc định một buổi 4–6 giờ cho core + report, thêm các buổi riêng nếu làm đủ thưởng.
