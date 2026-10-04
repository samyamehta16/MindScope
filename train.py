"""Train, evaluate, and explain MindScope models."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.optim import AdamW

from dataset import make_loaders
from models import build_model, count_parameters

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CKPT = OUT / "checkpoints"
FIG = OUT / "figures"
SEED = 42
EPOCHS = 6
BATCH = 16
LR = 1e-3
DEVICE = torch.device("cpu")
MODES = ("multimodal", "text", "image")


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer, mode: str, train: bool):
    model.train(train)
    total_loss = 0.0
    n = 0
    ys, ps, probs = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {
                k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            out = model(batch, mode=mode)
            loss = F.cross_entropy(out["logits"], batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            n += bs
            pred = out["logits"].argmax(dim=1)
            prob = F.softmax(out["logits"], dim=1)[:, 1]
            ys.append(batch["label"].cpu().numpy())
            ps.append(pred.cpu().numpy())
            probs.append(prob.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pr = np.concatenate(probs)
    metrics = {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y, pr))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics, y, p, pr


def train_variant(name: str, mode: str, loaders, vocab_size: int, pad_id: int):
    model = build_model(name, vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    history = []
    best_f1 = -1.0
    best_state = None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr, *_ = run_epoch(model, loaders["train"], optimizer, mode, train=True)
        va, *_ = run_epoch(model, loaders["val"], optimizer, mode, train=False)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"  {name}/{mode} epoch {epoch}/{EPOCHS}  "
            f"train f1={tr['f1']:.3f} acc={tr['accuracy']:.3f}  "
            f"val f1={va['f1']:.3f} acc={va['accuracy']:.3f}"
        )
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, y, p, pr = run_epoch(model, loaders["test"], optimizer, mode, train=False)
    elapsed = time.time() - t0
    tag = f"{name}_{mode}"
    torch.save({"state_dict": model.state_dict(), "name": name, "mode": mode}, CKPT / f"{tag}.pt")
    report = classification_report(y, p, target_names=["non-depressed", "depressed"], digits=4)
    (OUT / f"{tag}_report.txt").write_text(report, encoding="utf-8")
    return {
        "model": name,
        "mode": mode,
        "params": count_parameters(model),
        "seconds": elapsed,
        "history": history,
        "test": test_metrics,
        "y_true": y.tolist(),
        "y_pred": p.tolist(),
        "y_prob": pr.tolist(),
        "report": report,
    }, model


def plot_confusion(y, p, title, path: Path):
    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["non-dep", "dep"])
    ax.set_yticks([0, 1], ["non-dep", "dep"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc(results: list[dict], path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for item in results:
        y = np.array(item["y_true"])
        pr = np.array(item["y_prob"])
        fpr, tpr, _ = roc_curve(y, pr)
        label = f"{item['model']}-{item['mode']} (AUC {item['test']['auroc']:.3f})"
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — MindScope variants")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_bars(results: list[dict], path: Path):
    labels = [f"{r['model']}\n{r['mode']}" for r in results]
    f1 = [r["test"]["f1"] for r in results]
    acc = [r["test"]["accuracy"] for r in results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - 0.18, acc, width=0.36, label="Accuracy")
    ax.bar(x + 0.18, f1, width=0.36, label="F1")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test accuracy and F1 by model / modality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def denorm_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def grad_cam(model, batch) -> np.ndarray:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad()
    out = model(batch, mode="multimodal")
    score = out["logits"][0, out["logits"][0].argmax()]
    feat = model.image_encoder.last_feat
    grads = torch.autograd.grad(score, feat, retain_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def save_explanations(model, loader, tag: str, n: int = 4):
    model.eval()
    saved = 0
    for batch in loader:
        batch_t = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bsz = batch_t["label"].size(0)
        for i in range(bsz):
            one = {
                "input_ids": batch_t["input_ids"][i : i + 1],
                "attention_mask": batch_t["attention_mask"][i : i + 1],
                "image": batch_t["image"][i : i + 1].clone().requires_grad_(True),
                "label": batch_t["label"][i : i + 1],
            }
            with torch.enable_grad():
                cam = grad_cam(model, one)
            with torch.no_grad():
                feats = model(one, mode="multimodal")
            tokens = batch["tokens"][i]
            mask = one["attention_mask"][0].cpu().numpy().astype(bool)
            word_attn = feats["attn_i2t"][0].cpu().numpy()
            word_attn = word_attn.mean(axis=0)[mask]
            keep_tokens = [tok for tok, m in zip(tokens, mask) if m]
            pred = int(feats["logits"][0].argmax().item())
            prob = float(F.softmax(feats["logits"], dim=1)[0, 1].item())
            true = int(one["label"].item())

            rgb = denorm_image(one["image"][0])
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
            axes[0].imshow(rgb)
            axes[0].set_title("Posted image")
            axes[0].axis("off")
            axes[1].imshow(rgb)
            axes[1].imshow(cam, cmap="jet", alpha=0.45)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")
            if len(keep_tokens) > 24:
                keep_tokens = keep_tokens[:24]
                word_attn = word_attn[:24]
            colors = plt.cm.Reds(word_attn / (word_attn.max() + 1e-8))
            axes[2].barh(range(len(keep_tokens)), word_attn, color=colors)
            axes[2].set_yticks(range(len(keep_tokens)), keep_tokens, fontsize=7)
            axes[2].invert_yaxis()
            axes[2].set_title("Text attention (image→text)")
            fig.suptitle(
                f"{tag}  true={true} pred={pred} P(dep)={prob:.2f}  {batch['post_id'][i]}",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(FIG / f"explain_{tag}_{saved}.png", dpi=130)
            plt.close(fig)
            saved += 1
            if saved >= n:
                return


def main():
    seed_all()
    OUT.mkdir(exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    loaders, tokenizer, frame, splits = make_loaders(batch_size=BATCH)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    print(f"vocab={vocab_size}  train={len(loaders['train'].dataset)}  "
          f"val={len(loaders['val'].dataset)}  test={len(loaders['test'].dataset)}")

    all_results = []
    models_for_xai = {}
    for name in ("lstm", "cnn"):
        for mode in MODES:
            print(f"\n=== Training {name} / {mode} ===")
            result, model = train_variant(name, mode, loaders, vocab_size, pad_id)
            all_results.append(result)
            if mode == "multimodal":
                models_for_xai[name] = model
            plot_confusion(
                np.array(result["y_true"]),
                np.array(result["y_pred"]),
                f"{name} {mode}",
                FIG / f"cm_{name}_{mode}.png",
            )

    plot_roc(all_results, FIG / "roc_all.png")
    plot_bars(all_results, FIG / "bars_all.png")

    print("\nSaving explanations...")
    for name, model in models_for_xai.items():
        save_explanations(model, loaders["test"], name, n=4)

    slim = []
    for r in all_results:
        slim.append(
            {
                "model": r["model"],
                "mode": r["mode"],
                "params": r["params"],
                "seconds": round(r["seconds"], 1),
                "test": {k: round(v, 4) if isinstance(v, float) else v for k, v in r["test"].items()},
            }
        )
    payload = {
        "n_posts": int(len(frame)),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "class_balance": {
            "depressed": int(frame["label"].sum()),
            "non_depressed": int((1 - frame["label"]).sum()),
        },
        "results": slim,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n========== TEST RESULTS ==========")
    print(f"{'model':<8} {'mode':<12} {'acc':>7} {'f1':>7} {'prec':>7} {'rec':>7} {'auc':>7} {'params':>10}")
    for r in slim:
        t = r["test"]
        print(
            f"{r['model']:<8} {r['mode']:<12} {t['accuracy']:7.3f} {t['f1']:7.3f} "
            f"{t['precision']:7.3f} {t['recall']:7.3f} {t['auroc']:7.3f} {r['params']:10d}"
        )
    print(f"\nSaved metrics to {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
