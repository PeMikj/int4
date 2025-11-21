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
