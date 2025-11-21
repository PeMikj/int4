# Quantization helper for Kaggle notebooks

Интенсивные куски из `sym.ipynb` и `assym.ipynb` вынесены в модуль `quant_lib`.
Он содержит симметричную и асимметричную int4-квантизации с выбором слоёв через `TARGETS`.

## Использование в Kaggle/Colab

```bash
pip install git+https://github.com/<your-org>/<your-repo>.git  # после того как запушите этот код
```

Пример ноутбука:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from quant_lib import replace_linears_with_quant, DEFAULT_TARGETS

# какие линейные слои квантизировать
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"}  # или DEFAULT_TARGETS

model_name = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")

# mode="sym" или mode="asym"
replace_linears_with_quant(model, targets=TARGETS, mode="sym")

prompt = "Who is Albert Einstein?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Если нужен асимметричный вариант, просто передайте `mode="asym"`. Код ядра соответствует содержимому оригинальных ноутов; `TARGETS`
подхватывается как множество имён дочерних модулей `torch.nn.Linear`, которые будут заменены на int4-слои.

## Сохранение и логирование (пример для Kaggle)

```python
# после замены слоёв и расчёта ppl = fast_ppl(...)
from pathlib import Path
import csv, datetime

save_dir = Path(f"quantized_model_sym")  # или asym
save_dir.mkdir(exist_ok=True)
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)

log_path = Path("metrics_log.csv")
ppl = fast_ppl(model, tokenizer, batch_size=8, max_tokens=128, limit=5000)  # или подставьте своё значение
row = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "model": model_name,
    "quant": "sym",
    "targets": ";".join(sorted(TARGETS)),
    "ppl": float(ppl),
    "batch_size": 8,
    "max_tokens": 128,
    "limit": 5000,
}
exists = log_path.exists()
with log_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if not exists:
        writer.writeheader()
    writer.writerow(row)
print("Saved model to", save_dir, "and metrics to", log_path)
```

Если на среде Triton не собирается, можно принудительно включить торч-фоллбек: `INT4_USE_TRITON=0` в переменных окружения.

## Быстрый сценарий: квантизация → перплексия → сохранение → лог

```python
# 1) Установка зависимостей (Kaggle)
!pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q triton==2.3.1 bitsandbytes transformers accelerate datasets git+https://github.com/PeMikj/int4.git

import os, csv, datetime, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from quant_lib import replace_linears_with_quant, DEFAULT_TARGETS
from datasets import load_dataset

# 2) Подготовка модели и квантизация
TARGETS = DEFAULT_TARGETS
mode = "sym"  # или "asym"
model_name = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
replace_linears_with_quant(model, targets=TARGETS, mode=mode)

# 3) Перплексия (компактный расчёт)
def fast_ppl(model, tokenizer, batch_size=8, max_tokens=128, limit=5000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = ds["text"][:limit]
    def batchify(texts):
        for i in range(0, len(texts), batch_size):
            yield texts[i:i+batch_size]
    total_nll, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in batchify(texts):
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to("cuda")
            labels = toks["input_ids"].clone()
            out = model(**toks, labels=labels)
            nll = out.loss.item() * labels.numel()
            total_nll += nll
            total_tokens += labels.numel()
    return math.exp(total_nll / total_tokens)

ppl = fast_ppl(model, tokenizer, batch_size=8, max_tokens=128, limit=5000)

# 4) Сохранение модели
save_dir = Path(f"quantized_model_{mode}")
save_dir.mkdir(exist_ok=True)
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)

# 5) Лог в CSV
log_path = Path("metrics_log.csv")
row = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "model": model_name,
    "quant": mode,
    "targets": ";".join(sorted(TARGETS)),
    "ppl": float(ppl),
    "batch_size": 8,
    "max_tokens": 128,
    "limit": 5000,
}
exists = log_path.exists()
with log_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if not exists:
        writer.writeheader()
    writer.writerow(row)
print("PPL:", ppl, "| saved:", save_dir, "| log:", log_path)
```
