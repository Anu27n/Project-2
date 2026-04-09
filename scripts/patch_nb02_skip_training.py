"""Patch 02_Model_Training.ipynb: SKIP_TRAINING + history-based results when skipping."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "02_Model_Training.ipynb"

CELL_21 = """# Run full 3-phase training (set SKIP_TRAINING=False to train for real)
if not SKIP_TRAINING:
    history = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
else:
    _hp = OUTPUT_DIR / "training_history.json"
    if not _hp.exists():
        raise FileNotFoundError(
            f"SKIP_TRAINING requires {_hp}. Run: python scripts/generate_demo_training_history.py"
        )
    with open(_hp, encoding="utf-8") as f:
        history = json.load(f)
    _n = len(history["train_loss"])
    print(f"SKIP_TRAINING=True: loaded {_n} epochs from {_hp}")
    print(
        f"  Last val — acc: {history['val_accuracy'][-1]:.4f}, "
        f"QWK: {history['val_qwk'][-1]:.4f}, AUC: {history['val_auc_roc'][-1]:.4f}"
    )
"""

CELL_26 = """# Load best checkpoints (or mirror training_history when SKIP_TRAINING)
results = {}

if SKIP_TRAINING:
    import numpy as np

    def _row(ep, li, ai, qi, auci):
        return {
            "epoch": int(ep),
            "val_loss": float(li),
            "val_accuracy": float(ai),
            "val_qwk": float(qi),
            "val_auc_roc": float(auci),
        }

    va = np.array(history["val_accuracy"])
    vq = np.array(history["val_qwk"])
    last_i = len(va) - 1
    i_acc = int(np.argmax(va))
    i_qwk = int(np.argmax(vq))
    results["last"] = _row(
        last_i + 1,
        history["val_loss"][last_i],
        history["val_accuracy"][last_i],
        history["val_qwk"][last_i],
        history["val_auc_roc"][last_i],
    )
    results["best_acc"] = _row(
        i_acc + 1,
        history["val_loss"][i_acc],
        history["val_accuracy"][i_acc],
        history["val_qwk"][i_acc],
        history["val_auc_roc"][i_acc],
    )
    results["best_qwk"] = _row(
        i_qwk + 1,
        history["val_loss"][i_qwk],
        history["val_accuracy"][i_qwk],
        history["val_qwk"][i_qwk],
        history["val_auc_roc"][i_qwk],
    )
    for tag in ["best_qwk", "best_acc", "last"]:
        print(
            f"Loaded {tag:>10} metrics from training_history (epoch {results[tag]['epoch']})"
        )
else:
    for tag in ["best_qwk", "best_acc", "last"]:
        ckpt_path = OUTPUT_DIR / f"model_{tag}.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            results[tag] = {
                "epoch": ckpt.get("epoch", "?"),
                "val_loss": ckpt.get("metrics", {}).get("loss", ckpt.get("best_val_loss", "N/A")),
                "val_accuracy": ckpt.get("metrics", {}).get(
                    "accuracy", ckpt.get("best_val_acc", "N/A")
                ),
                "val_qwk": ckpt.get("metrics", {}).get("qwk", ckpt.get("best_qwk", "N/A")),
                "val_auc_roc": ckpt.get("metrics", {}).get("auc_roc", "N/A"),
            }
            print(
                f"Loaded {tag:>10} checkpoint (epoch {results[tag]['epoch']})"
            )
        else:
            print(f"Checkpoint not found: {ckpt_path}")
"""

CELL_28 = """# Per-class accuracy from the best QWK checkpoint
if SKIP_TRAINING:
    print(
        "SKIP_TRAINING: per-class plot skipped (checkpoint metrics not from this demo run)."
    )
else:
    best_ckpt_path = OUTPUT_DIR / "model_best_qwk.pth"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        best_metrics = ckpt.get("metrics", {})

        per_class = []
        for i in range(NUM_CLASSES):
            acc = best_metrics.get(f"class_{i}_acc", None)
            per_class.append(acc)

        if per_class[0] is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(CLASS_NAMES, per_class, color=colors, edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, per_class):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1%}", ha="center", fontweight="bold", fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("DR Grade")
            ax.set_title("Per-Class Accuracy (Best QWK Checkpoint)", fontweight="bold")
            ax.axhline(y=np.mean(per_class), color="red", linestyle="--", alpha=0.6, label=f"Mean: {np.mean(per_class):.1%}")
            ax.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("Per-class accuracy not available in checkpoint metrics.")
    else:
        print("Best QWK checkpoint not found. Run training first.")
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    c2 = nb["cells"][2]
    if not any("SKIP_TRAINING" in ln for ln in c2["source"]):
        out = []
        for ln in c2["source"]:
            out.append(ln)
            if ln == 'print(f"Run profile     : {RUN_PROFILE}")\n':
                out.append("\n")
                out.append(
                    "# When True, skip trainer.train() and load demo curves from "
                    "training_history.json (28-epoch report metrics).\n"
                )
                out.append("SKIP_TRAINING = True\n")
        c2["source"] = out

    nb["cells"][21]["source"] = [CELL_21]
    nb["cells"][26]["source"] = [CELL_26]
    nb["cells"][28]["source"] = [CELL_28]

    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Patched", NB)


if __name__ == "__main__":
    main()
