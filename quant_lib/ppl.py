import math
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def fast_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    text_field: str = "text",
    limit: int = 5000,
    batch_size: int = 8,
    max_tokens: int = 128,
    device: Optional[str] = None,
) -> float:
    """
    Быстрая оценка перплексии: грузим датасет, токенизируем, суммируем NLL.
    Паддинги маскируются (attention_mask -> labels == -100).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset(dataset_name, dataset_config, split=split)
    texts: Iterable[str] = ds[text_field][:limit]

    def batchify(items):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch_texts in batchify(texts):
            toks = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(device)

            labels = toks["input_ids"].clone()
            attention = toks["attention_mask"]
            labels[attention == 0] = -100  # не учитываем паддинг
            token_count = int(attention.sum().item())

            out = model(**toks, labels=labels)
            total_nll += out.loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("No tokens processed; check dataset or tokenizer settings")

    return float(math.exp(total_nll / total_tokens))


__all__ = ["fast_ppl"]
