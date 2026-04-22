"""Generate Phase-2 training-curve comparison: ViT-B reference vs DINOv2/DINOv3 ViT-L."""
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE

VITB_CSV = Path(
    r"C:/Users/alper/Documents/PlantCLEF_2026_Local/DinoViT_pretrained_models/"
    r"pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/"
    r"summary.csv"
)
DINOV2_LOG = Path(r"C:/Users/alper/Desktop/phase2_vit-p2_6744298.err")
DINOV3_LOG = Path(r"C:/Users/alper/Desktop/phase2_vit-p2_6744244.err")


def parse_vitb_summary(path):
    epochs, top1 = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["epoch"] == "epoch":
                continue
            try:
                epochs.append(int(row["epoch"]))
                top1.append(float(row["eval_top1"]))
            except (ValueError, KeyError):
                continue
    return epochs, top1


def parse_vitl_log(path):
    text = path.read_text(errors="ignore")
    epoch_blocks = re.split(r"\nTrain: (\d+) \[\s*0/", text)
    epoch_seen = set()
    raw_by_epoch, ema_by_epoch = {}, {}
    for i in range(1, len(epoch_blocks), 2):
        ep = int(epoch_blocks[i])
        block = epoch_blocks[i + 1] if i + 1 < len(epoch_blocks) else ""
        m_raw = re.search(
            r"Test: \[\s*\d+/\s*\d+\]\s+Time:.*?Acc@1:\s+\S+\s+\(\s*([\d.]+)\s*\)\s+Acc@5",
            block,
        )
        m_ema = re.search(
            r"Test \(EMA\): \[\s*\d+/\s*\d+\]\s+Time:.*?Acc@1:\s+\S+\s+\(\s*([\d.]+)\s*\)\s+Acc@5",
            block,
        )
        last_raw = None
        for m in re.finditer(
            r"Test: \[\s*\d+/\s*\d+\]\s+Time:.*?Acc@1:\s+\S+\s+\(\s*([\d.]+)\s*\)",
            block,
        ):
            last_raw = m
        last_ema = None
        for m in re.finditer(
            r"Test \(EMA\): \[\s*\d+/\s*\d+\]\s+Time:.*?Acc@1:\s+\S+\s+\(\s*([\d.]+)\s*\)",
            block,
        ):
            last_ema = m
        if last_raw:
            raw_by_epoch[ep] = float(last_raw.group(1))
        if last_ema:
            ema_by_epoch[ep] = float(last_ema.group(1))
        epoch_seen.add(ep)
    epochs = sorted(raw_by_epoch.keys() & ema_by_epoch.keys())
    raw = [raw_by_epoch[e] for e in epochs]
    ema = [ema_by_epoch[e] for e in epochs]
    return epochs, raw, ema


def plot_combined(vitb, dv2, dv3, outfile):
    vitb_ep, vitb_top1 = vitb
    dv2_ep, dv2_raw, dv2_ema = dv2
    dv3_ep, dv3_raw, dv3_ema = dv3

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(vitb_ep, vitb_top1, color="#888888", lw=2.0,
            label="ViT-B DINOv2 (reference, 100 ep)", zorder=2)
    ax.plot(dv2_ep, dv2_raw, color="#1f77b4", lw=2.2, marker="o", ms=4.5,
            label="ViT-L DINOv2 (raw)", zorder=4)
    ax.plot(dv2_ep, dv2_ema, color="#1f77b4", lw=1.4, ls="--", marker="o", ms=3,
            label="ViT-L DINOv2 (EMA)", zorder=3, alpha=0.75)
    ax.plot(dv3_ep, dv3_raw, color="#d62728", lw=2.2, marker="s", ms=4.5,
            label="ViT-L DINOv3 (raw)", zorder=4)
    ax.plot(dv3_ep, dv3_ema, color="#d62728", lw=1.4, ls="--", marker="s", ms=3,
            label="ViT-L DINOv3 (EMA)", zorder=3, alpha=0.75)

    last_vitb_ep = vitb_ep[-1] if vitb_ep else 0
    max_ep = max(last_vitb_ep,
                 dv2_ep[-1] if dv2_ep else 0,
                 dv3_ep[-1] if dv3_ep else 0)
    ax.set_xlim(-1, max_ep + 2)
    ax.set_xticks(range(0, max_ep + 2, 5))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation top-1 (%)", fontsize=12)
    ax.set_title(
        "PlantCLEF 2024 Phase-2 finetuning — ViT-B vs ViT-L (DINOv2 / DINOv3)",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    if dv2_ep:
        ax.axvspan(dv2_ep[-1] + 0.5, max_ep + 2, alpha=0.06, color="gray", zorder=0)
        ax.annotate(
            "ViT-L tournament\nends at E20",
            xy=(dv2_ep[-1], max(dv2_raw[-1], dv3_raw[-1]) - 3),
            xytext=(dv2_ep[-1] - 8, max(dv2_raw[-1], dv3_raw[-1]) - 10),
            fontsize=9, color="#444",
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
        )

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")
    print(f"Wrote {outfile.with_suffix('.pdf')}")


def plot_raw_only(vitb, dv2, dv3, outfile):
    vitb_ep, vitb_top1 = vitb
    dv2_ep, dv2_raw, _ = dv2
    dv3_ep, dv3_raw, _ = dv3

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(vitb_ep, vitb_top1, color="#888888", lw=2.2,
            label="ViT-B DINOv2 (reference, 100 ep)")
    ax.plot(dv2_ep, dv2_raw, color="#1f77b4", lw=2.4, marker="o", ms=5,
            label="ViT-L DINOv2")
    ax.plot(dv3_ep, dv3_raw, color="#d62728", lw=2.4, marker="s", ms=5,
            label="ViT-L DINOv3")

    max_ep = max(vitb_ep[-1], dv2_ep[-1], dv3_ep[-1])
    ax.set_xlim(-1, max_ep + 2)
    ax.set_xticks(range(0, max_ep + 2, 5))
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation top-1 (%)", fontsize=12)
    ax.set_title(
        "PlantCLEF 2024 Phase-2 finetuning — raw top-1 trajectory",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    vitb = parse_vitb_summary(VITB_CSV)
    dv2 = parse_vitl_log(DINOV2_LOG)
    dv3 = parse_vitl_log(DINOV3_LOG)

    print(f"ViT-B reference: {len(vitb[0])} epochs (0..{vitb[0][-1]})")
    print(f"ViT-L DINOv2:    {len(dv2[0])} epochs (0..{dv2[0][-1] if dv2[0] else '?'}),"
          f" raw final={dv2[1][-1]:.2f}, ema final={dv2[2][-1]:.2f}")
    print(f"ViT-L DINOv3:    {len(dv3[0])} epochs (0..{dv3[0][-1] if dv3[0] else '?'}),"
          f" raw final={dv3[1][-1]:.2f}, ema final={dv3[2][-1]:.2f}")

    plot_combined(vitb, dv2, dv3, OUT_DIR / "training_curves.png")
    plot_raw_only(vitb, dv2, dv3, OUT_DIR / "training_curves_raw.png")


if __name__ == "__main__":
    main()
