"""Unit tests for MindScope data, models, and metrics helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import CrossAttentionFusion, ImageEncoder, build_model, count_parameters
from prepare_data import generate_post_image


class ImageGenTests(unittest.TestCase):
    def test_image_shape_and_mode(self):
        import random

        rng = random.Random(0)
        img = generate_post_image(label=1, rng=rng, mismatch=False)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, "RGB")

    def test_mismatch_changes_pixels(self):
        import random

        a = generate_post_image(1, random.Random(1), mismatch=False)
        b = generate_post_image(1, random.Random(1), mismatch=True)
        self.assertFalse(np.array_equal(np.array(a), np.array(b)))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "image": torch.randn(4, 3, 224, 224),
            "label": torch.tensor([0, 1, 1, 0]),
        }
        self.batch["input_ids"][:, -2:] = 0
        self.batch["attention_mask"][:, -2:] = 0

    def test_lstm_forward_shapes(self):
        model = build_model("lstm", vocab_size=200, pad_id=0)
        out = model(self.batch, mode="multimodal")
        self.assertEqual(tuple(out["logits"].shape), (4, 2))
        self.assertEqual(tuple(out["fused"].shape), (4, 256))
        self.assertEqual(out["attn_t2i"].shape[0], 4)

    def test_cnn_forward_shapes(self):
        model = build_model("cnn", vocab_size=200, pad_id=0)
        for mode in ("multimodal", "text", "image"):
            out = model(self.batch, mode=mode)
            self.assertEqual(tuple(out["logits"].shape), (4, 2), msg=mode)

    def test_parameter_counts_are_lightweight(self):
        lstm = build_model("lstm", vocab_size=30522, pad_id=0)
        cnn = build_model("cnn", vocab_size=30522, pad_id=0)
        n_lstm = count_parameters(lstm)
        n_cnn = count_parameters(cnn)
        self.assertLess(n_lstm, 5_000_000)
        self.assertLess(n_cnn, 5_000_000)
        self.assertGreater(n_lstm, 100_000)

    def test_cross_attention_mask(self):
        fusion = CrossAttentionFusion(32, 32, hidden=32, n_heads=4)
        text = torch.randn(2, 8, 32)
        image = torch.randn(2, 16, 32)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, 5:] = 0
        fused, a_t2i, a_i2t, *_ = fusion(text, image, mask)
        self.assertEqual(tuple(fused.shape), (2, 64))
        self.assertFalse(torch.isnan(fused).any())

    def test_image_encoder_spatial_map(self):
        enc = ImageEncoder(out_dim=128)
        x = torch.randn(2, 3, 224, 224)
        patches = enc(x)
        self.assertEqual(tuple(enc.last_feat.shape), (2, 128, 14, 14))
        self.assertEqual(tuple(patches.shape), (2, 196, 128))

    def test_one_train_step(self):
        model = build_model("cnn", vocab_size=200, pad_id=0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        out = model(self.batch, mode="multimodal")
        loss = torch.nn.functional.cross_entropy(out["logits"], self.batch["label"])
        loss.backward()
        opt.step()
        self.assertTrue(torch.isfinite(loss))


class TokenizerTests(unittest.TestCase):
    def test_bert_tokenizer_padding(self):
        from dataset import encode_texts, load_bert_tokenizer

        tok = load_bert_tokenizer(max_len=16)
        ids, mask, tokens = encode_texts(tok, ["feeling empty today", "had a great morning walk"])
        self.assertEqual(ids.shape, (2, 16))
        self.assertEqual(mask.shape, (2, 16))
        self.assertEqual(len(tokens), 2)
        self.assertTrue((mask[0] == 1).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
