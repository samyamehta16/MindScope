"""Lightweight multimodal encoders with bidirectional cross-attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Small conv stack (MobileNet-style depth) producing 14x14 patches."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, out_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim
        self.last_feat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        self.last_feat = feat
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2)


class TextLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 96, hidden: int = 128, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )
        return self.norm(out)


class TextCNNEncoder(nn.Module):
    """Token-level TextCNN (local n-gram filters, padded to keep sequence length)."""

    def __init__(self, vocab_size: int, embed_dim: int = 96, hidden: int = 128, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.conv3 = nn.Conv1d(embed_dim, hidden // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, hidden // 2, kernel_size=5, padding=2)
        self.proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids).transpose(1, 2)
        feat = torch.cat([F.relu(self.conv3(emb)), F.relu(self.conv5(emb))], dim=1)
        feat = feat.transpose(1, 2)
        feat = self.norm(self.proj(feat))
        return feat * attention_mask.unsqueeze(-1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, hidden: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.text_to_image = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden)
        self.norm_i = nn.LayerNorm(hidden)
        self.hidden = hidden

    def forward(self, text_tokens, image_patches, text_mask):
        t = self.text_proj(text_tokens)
        i = self.image_proj(image_patches)
        key_pad = ~text_mask.bool()
        t_att, attn_t2i = self.text_to_image(t, i, i, need_weights=True, average_attn_weights=True)
        i_att, attn_i2t = self.image_to_text(
            i, t, t, key_padding_mask=key_pad, need_weights=True, average_attn_weights=True
        )
        t = self.norm_t(t + t_att)
        i = self.norm_i(i + i_att)
        mask_f = text_mask.unsqueeze(-1).float()
        t_pool = (t * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        i_pool = i.mean(dim=1)
        fused = torch.cat([t_pool, i_pool], dim=-1)
        return fused, attn_t2i, attn_i2t, t_pool, i_pool


class Classifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MindScopeModel(nn.Module):
    def __init__(self, text_encoder: nn.Module, image_encoder: ImageEncoder, fusion: CrossAttentionFusion):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = fusion
        self.mm_head = Classifier(fusion.hidden * 2)
        self.text_head = Classifier(fusion.hidden)
        self.image_head = Classifier(fusion.hidden)

    def encode(self, batch):
        text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        image = self.image_encoder(batch["image"])
        fused, attn_t2i, attn_i2t, t_pool, i_pool = self.fusion(
            text, image, batch["attention_mask"]
        )
        return {
            "fused": fused,
            "text_pool": t_pool,
            "image_pool": i_pool,
            "attn_t2i": attn_t2i,
            "attn_i2t": attn_i2t,
        }

    def _pool_text(self, text_tokens, text_mask):
        t = self.fusion.text_proj(text_tokens)
        mask_f = text_mask.unsqueeze(-1).float()
        return (t * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def forward(self, batch, mode: str = "multimodal"):
        if mode == "text":
            text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
            t_pool = self._pool_text(text, batch["attention_mask"])
            return {"logits": self.text_head(t_pool), "text_pool": t_pool}
        if mode == "image":
            image = self.image_encoder(batch["image"])
            i_pool = self.fusion.image_proj(image).mean(dim=1)
            return {"logits": self.image_head(i_pool), "image_pool": i_pool}
        feats = self.encode(batch)
        feats["logits"] = self.mm_head(feats["fused"])
        return feats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, vocab_size: int, pad_id: int = 0) -> MindScopeModel:
    image = ImageEncoder(out_dim=128)
    if name == "lstm":
        text = TextLSTMEncoder(vocab_size, pad_id=pad_id)
    elif name == "cnn":
        text = TextCNNEncoder(vocab_size, pad_id=pad_id)
    else:
        raise ValueError(f"Unknown model {name}")
    fusion = CrossAttentionFusion(text.out_dim, image.out_dim, hidden=128, n_heads=4)
    return MindScopeModel(text, image, fusion)
