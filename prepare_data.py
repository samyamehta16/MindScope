"""Build the MindScope multimodal corpus.

The ECIR 2023 Twitter multimodal depression benchmark is not publicly
hosted (author request required). This script follows the project's
fallback plan: public social-media depression text paired with visual
posts whose tone, color, and content carry affective cues.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
SPLIT_PATH = DATA_DIR / "splits.json"
CSV_PATH = DATA_DIR / "posts.csv"

N_TOTAL = 720
SEED = 42
IMAGE_SIZE = 224
VISUAL_NOISE = 0.18


def _usable(row) -> bool:
    text = str(row["text"]).strip()
    n_words = len(text.split())
    if not (8 <= n_words <= 70):
        return False
    if float(row.get("text_length", n_words)) < 8:
        return False
    ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
    return ascii_ratio > 0.95


def load_public_texts(n: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset

    raw = load_dataset("mdsajjadullah/explainDepression-social-media-xai", split="train")
    df = raw.to_pandas()
    df = df[df["source"] == "reddit"].copy()
    df = df[df.apply(_usable, axis=1)]
    df["label"] = df["binary_label"].astype(int)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["text"])

    parts = []
    rng = np.random.RandomState(seed)
    for label in (0, 1):
        subset = df[df["label"] == label]
        take = min(n // 2, len(subset))
        idx = rng.choice(subset.index.to_numpy(), size=take, replace=False)
        parts.append(subset.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["post_id"] = [f"p{i:04d}" for i in range(len(out))]
    return out[["post_id", "text", "label"]]


def _gradient(size: int, c1, c2, vertical: bool = True) -> np.ndarray:
    t = np.linspace(0, 1, size, dtype=np.float32)
    if vertical:
        t = t[:, None]
    else:
        t = t[None, :]
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    return (c1 * (1 - t) + c2 * t).astype(np.uint8)


def generate_post_image(label: int, rng: random.Random, mismatch: bool) -> Image.Image:
    """Render a 224x224 social-style image with affective visual cues."""
    visual = label if not mismatch else 1 - label
    size = IMAGE_SIZE

    if visual == 1:
        top, bot = (16, 22, 48), (38, 42, 68)
        accent = (92, 108, 168)
        figure = (28, 30, 46)
        cool = True
    else:
        top, bot = (255, 186, 72), (98, 186, 214)
        accent = (255, 146, 42)
        figure = (42, 48, 70)
        cool = False

    jitter = rng.randint(-12, 13)
    top = tuple(int(np.clip(c + jitter, 0, 255)) for c in top)
    bot = tuple(int(np.clip(c + jitter, 0, 255)) for c in bot)

    arr = np.zeros((size, size, 3), dtype=np.uint8)
    grad = _gradient(size, top, bot, vertical=True)
    arr[:] = grad[:, None, :]
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img, "RGBA")

    if cool:
        for _ in range(36):
            x = rng.randint(0, size)
            y = rng.randint(0, size)
            draw.line((x, y, x - 5, y + 22), fill=(170, 180, 210, 140), width=1)
        wx, wy = rng.randint(18, 40), rng.randint(28, 50)
        draw.rectangle((wx, wy, wx + 54, wy + 70), outline=accent + (220,), width=3)
        draw.line((wx + 27, wy, wx + 27, wy + 70), fill=accent + (180,), width=2)
        draw.line((wx, wy + 35, wx + 54, wy + 35), fill=accent + (180,), width=2)
        fx = rng.randint(90, 150)
        fy = rng.randint(108, 130)
        draw.ellipse((fx, fy, fx + 16, fy + 16), fill=figure + (255,))
        draw.line((fx + 8, fy + 16, fx + 8, fy + 54), fill=figure + (255,), width=3)
        draw.line((fx + 8, fy + 28, fx - 6, fy + 44), fill=figure + (255,), width=3)
        draw.line((fx + 8, fy + 28, fx + 22, fy + 44), fill=figure + (255,), width=3)
        draw.ellipse((size - 70, 18, size - 28, 60), fill=(210, 214, 230, 80))
    else:
        sx, sy = rng.randint(140, 168), rng.randint(12, 28)
        draw.ellipse((sx, sy, sx + 52, sy + 52), fill=(255, 224, 90, 255))
        for k in range(8):
            ang = k * 45
            rad = np.deg2rad(ang)
            x1 = sx + 26 + 28 * np.cos(rad)
            y1 = sy + 26 + 28 * np.sin(rad)
            x2 = sx + 26 + 40 * np.cos(rad)
            y2 = sy + 26 + 40 * np.sin(rad)
            draw.line((x1, y1, x2, y2), fill=(255, 210, 70, 255), width=3)
        base_x = rng.randint(40, 80)
        for i, dx in enumerate((-28, 0, 28)):
            fx = base_x + 70 + dx
            fy = 118 + (i % 2) * 4
            draw.ellipse((fx, fy, fx + 14, fy + 14), fill=figure + (255,))
            draw.line((fx + 7, fy + 14, fx + 7, fy + 48), fill=figure + (255,), width=3)
            draw.line((fx + 7, fy + 24, fx - 8, fy + 38), fill=figure + (255,), width=3)
            draw.line((fx + 7, fy + 24, fx + 22, fy + 38), fill=figure + (255,), width=3)
        for _ in range(5):
            cx, cy = rng.randint(16, 200), rng.randint(150, 200)
            col = (rng.randint(40, 90), rng.randint(160, 210), rng.randint(70, 120), 255)
            draw.ellipse((cx, cy, cx + 16, cy + 16), fill=col)
            draw.rectangle((cx + 6, cy + 16, cx + 10, cy + 28), fill=(46, 110, 58, 255))

    img = img.convert("RGB")
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    pixels = np.array(img, dtype=np.int16)
    noise = np.random.RandomState(rng.randint(0, 10_000_000)).randint(-10, 11, size=pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def stratified_splits(labels: np.ndarray, seed: int):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    train_idx, rest_idx, y_train, y_rest = train_test_split(
        idx, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, random_state=seed, stratify=y_rest
    )
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_TOTAL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading public depression text (Reddit / HuggingFace)...")
    df = load_public_texts(args.n, args.seed)
    rng = random.Random(args.seed)

    mismatches = 0
    for i, row in df.iterrows():
        mismatch = rng.random() < VISUAL_NOISE
        mismatches += int(mismatch)
        image = generate_post_image(int(row["label"]), rng, mismatch)
        path = IMAGE_DIR / f"{row['post_id']}.jpg"
        image.save(path, quality=88)
        df.at[i, "image_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        df.at[i, "visual_mismatch"] = int(mismatch)

    df.to_csv(CSV_PATH, index=False)
    splits = stratified_splits(df["label"].to_numpy(), args.seed)
    SPLIT_PATH.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(f"Wrote {len(df)} posts -> {CSV_PATH}")
    print(f"Images -> {IMAGE_DIR}  (visual mismatch {mismatches}/{len(df)})")
    print(
        "Split sizes:",
        {k: len(v) for k, v in splits.items()},
        "positives:",
        int(df["label"].sum()),
        "negatives:",
        int((1 - df["label"]).sum()),
    )


if __name__ == "__main__":
    main()
