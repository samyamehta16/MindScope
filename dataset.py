from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_LEN = 48


def load_bert_tokenizer(max_len: int = MAX_LEN) -> Tokenizer:
    path = hf_hub_download("bert-base-uncased", "tokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0
    tokenizer.enable_truncation(max_length=max_len)
    tokenizer.enable_padding(length=max_len, pad_id=pad_id, pad_token="[PAD]")
    return tokenizer


def encode_texts(tokenizer: Tokenizer, texts: list[str]):
    encodings = tokenizer.encode_batch(texts)
    ids = np.stack([np.array(enc.ids, dtype=np.int64) for enc in encodings])
    mask = np.stack([np.array(enc.attention_mask, dtype=np.int64) for enc in encodings])
    tokens = [enc.tokens for enc in encodings]
    return ids, mask, tokens


def load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


class DepressionMMDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Tokenizer, root: Path = ROOT):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        ids, mask, tokens = encode_texts(tokenizer, self.frame["text"].tolist())
        self.input_ids = torch.from_numpy(ids)
        self.attention_mask = torch.from_numpy(mask)
        self.tokens = tokens
        self.labels = torch.tensor(self.frame["label"].to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = load_image_tensor(self.root / row["image_path"])
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "image": image,
            "label": self.labels[idx],
            "post_id": row["post_id"],
            "text": row["text"],
            "tokens": self.tokens[idx],
        }


def load_splits(root: Path = ROOT):
    frame = pd.read_csv(root / "data" / "posts.csv")
    splits = json.loads((root / "data" / "splits.json").read_text(encoding="utf-8"))
    return frame, splits


def collate_mm(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "post_id": [b["post_id"] for b in batch],
        "text": [b["text"] for b in batch],
        "tokens": [b["tokens"] for b in batch],
    }


def make_loaders(batch_size: int = 16, root: Path = ROOT, num_workers: int = 0):
    frame, splits = load_splits(root)
    tokenizer = load_bert_tokenizer()
    loaders = {}
    for name, idx in splits.items():
        subset = frame.iloc[idx]
        shuffle = name == "train"
        data = DepressionMMDataset(subset, tokenizer, root=root)
        loaders[name] = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_mm,
        )
    return loaders, tokenizer, frame, splits
