"""Dataset y utilidades para fine-tuning de transformers."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def prepare_transformer_input(row, max_chars: int = 3000) -> str:
    title = str(row.get("title", ""))
    body = str(row.get("text", ""))
    full = f"{title} {body}".strip()
    if len(full) > max_chars:
        first_paragraph = body.split("\n\n", 1)[0]
        short_body = first_paragraph[:2000]
        full = f"{title} {short_body}".strip()
    return full[:max_chars]


def prepare_transformer_inputs(df, max_chars: int = 3000) -> list[str]:
    return [
        prepare_transformer_input(row, max_chars=max_chars)
        for row in df.to_dict("records")
    ]


class NewsDataset(Dataset):
    """Tokeniza por ítem; usar con DataCollatorWithPadding en el Trainer."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
