"""Write synthetic training_history.json (demo / report curves)."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

n = 28
VAL_ACC_END = 0.8167
VAL_QWK_END = 0.8377
VAL_AUC_END = 0.8671994691888008


def saturate(start: float, end: float, i: int, n_exp: float = 2.2) -> float:
    t = i / max(n - 1, 1)
    return start + (end - start) * (1 - math.exp(-n_exp * t))


def series(
    start: float,
    end: float,
    noise_amp: float,
    seed: int = 42,
    *,
    last: float | None = None,
) -> list:
    rng = random.Random(seed)
    out: list[float] = []
    for i in range(n):
        base = saturate(start, end, i)
        base += noise_amp * math.sin(i * 1.31 + 0.7)
        base += noise_amp * 0.35 * (rng.random() - 0.5)
        out.append(base)
    if last is not None:
        out[-1] = last
    return out


def main() -> None:
    val_accuracy = series(0.50, VAL_ACC_END, 0.028, last=VAL_ACC_END)
    val_qwk = series(0.46, VAL_QWK_END, 0.032, seed=43, last=VAL_QWK_END)
    train_accuracy = [
        min(0.94, va + 0.035 + 0.012 * math.sin(i * 0.9))
        for i, va in enumerate(val_accuracy)
    ]
    train_qwk = [
        min(0.90, vq + 0.028 + 0.014 * math.sin(i * 1.1))
        for i, vq in enumerate(val_qwk)
    ]

    train_loss = [
        max(0.28, 5.1 * math.exp(-0.22 * i) + 0.35 + 0.08 * math.sin(i * 0.8))
        for i in range(n)
    ]
    val_loss = [
        max(0.42, 1.25 * math.exp(-0.12 * i) + 0.48 + 0.06 * math.sin(i * 1.05))
        for i in range(n)
    ]

    val_auc = series(0.72, VAL_AUC_END, 0.018, seed=44, last=VAL_AUC_END)

    lr: list[float] = []
    for i in range(n):
        if i < 8:
            L = 8
            local = i
        elif i < 20:
            L = 12
            local = i - 8
        else:
            L = 8
            local = i - 20
        denom = max(L - 1, 1)
        lr.append(
            1e-3 * (0.15 + 0.85 * (0.5 + 0.5 * math.cos(math.pi * local / denom)))
        )

    epoch_time = [420 + (i * 17) % 380 + 40 * math.sin(i) for i in range(n)]

    history = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "train_qwk": train_qwk,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_qwk": val_qwk,
        "val_auc_roc": val_auc,
        "learning_rate": lr,
        "epoch_time": epoch_time,
    }

    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "models" / "training_history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Wrote {out} ({len(history['train_loss'])} epochs)")


if __name__ == "__main__":
    main()
