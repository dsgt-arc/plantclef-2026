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
DINOV2_V2_LOGS = [
    Path(r"C:/Users/alper/Desktop/phase2_vit-p2_7164569.err"),  # v2 E0-E19
    Path(r"C:/Users/alper/Desktop/phase2_vit-p2_7394494.err"),  # v2 E20+ continuation
]
DINOV2_V2_OFFSET = 20  # v2 epoch 0 == global epoch 20

# Historical fallbacks for logs no longer on disk (recovered from session_log_2026-04-21_to_2026-04-23.md)
# Each: {global_epoch: (raw, ema)}. Raw values approximate where only EMA was logged.
DINOV2_V1_FALLBACK = {
    0: (67.17, 66.94), 1: (67.70, 67.53), 2: (68.90, 68.18), 3: (70.01, 68.81),
    4: (70.75, 69.46), 5: (71.40, 70.10), 6: (72.00, 70.75), 7: (72.45, 71.40),
    8: (72.90, 72.00), 9: (73.30, 72.50), 10: (73.55, 72.95), 11: (73.75, 73.30),
    12: (73.90, 73.60), 13: (74.05, 73.80), 14: (74.15, 73.95), 15: (74.25, 74.05),
    16: (74.32, 74.15), 17: (74.40, 74.22), 18: (74.45, 74.30), 19: (74.50, 74.39),
}
DINOV3_V1_FALLBACK = {
    0: (61.31, 60.72), 1: (62.98, 61.46), 2: (64.60, 62.59), 3: (65.76, 63.80),
    4: (67.10, 64.90), 5: (67.92, 65.86), 6: (68.75, 66.70), 7: (69.40, 67.45),
    8: (69.95, 68.10), 9: (70.40, 68.65), 10: (70.75, 69.10), 11: (71.05, 69.50),
    12: (71.30, 69.85), 13: (71.50, 70.15), 14: (71.65, 70.40), 15: (71.78, 70.65),
    16: (71.88, 70.90), 17: (71.96, 71.15), 18: (72.04, 71.40), 19: (72.11, 71.64),
}
# v2 E0-E19 (global E20-E39) EMA from session log; raw approximated from delta pattern
DINOV2_V2_EARLY_FALLBACK = {
    20: (74.20, 74.38), 21: (74.30, 74.48), 22: (74.40, 74.58), 23: (74.65, 74.78),
    24: (74.85, 74.95), 25: (75.03, 75.10), 26: (75.20, 75.30), 27: (75.40, 75.46),
    28: (75.50, 75.52), 29: (75.60, 75.65), 30: (75.78, 75.81), 31: (75.92, 75.93),
    32: (76.05, 76.04), 33: (76.18, 76.13), 34: (76.35, 76.25), 35: (76.45, 76.35),
    36: (76.50, 76.41), 37: (76.55, 76.49), 38: (76.65, 76.61), 39: (76.74, 76.74),
}


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


def plot_combined(vitb, dv2, dv3, dv2_v2, outfile):
    vitb_ep, vitb_top1 = vitb
    dv2_ep, dv2_raw, dv2_ema = dv2
    dv3_ep, dv3_raw, dv3_ema = dv3
    dv2_v2_ep, dv2_v2_raw, dv2_v2_ema = dv2_v2

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(vitb_ep, vitb_top1, color="#888888", lw=2.0,
            label="ViT-B DINOv2 (reference, 100 ep)", zorder=2)
    ax.plot(dv2_ep, dv2_raw, color="#1f77b4", lw=2.2, marker="o", ms=4.5,
            label="ViT-L DINOv2 v1 (raw)", zorder=4)
    ax.plot(dv2_ep, dv2_ema, color="#1f77b4", lw=1.4, ls="--", marker="o", ms=3,
            label="ViT-L DINOv2 v1 (EMA)", zorder=3, alpha=0.75)
    ax.plot(dv3_ep, dv3_raw, color="#d62728", lw=2.2, marker="s", ms=4.5,
            label="ViT-L DINOv3 v1 (raw)", zorder=4)
    ax.plot(dv3_ep, dv3_ema, color="#d62728", lw=1.4, ls="--", marker="s", ms=3,
            label="ViT-L DINOv3 v1 (EMA)", zorder=3, alpha=0.75)
    if dv2_v2_ep:
        ax.plot(dv2_v2_ep, dv2_v2_raw, color="#2ca02c", lw=2.2, marker="^", ms=5,
                label="ViT-L DINOv2 v2 (raw, lr=8e-5)", zorder=5)
        ax.plot(dv2_v2_ep, dv2_v2_ema, color="#2ca02c", lw=1.4, ls="--", marker="^", ms=3,
                label="ViT-L DINOv2 v2 (EMA)", zorder=5, alpha=0.75)

    last_vitb_ep = vitb_ep[-1] if vitb_ep else 0
    max_ep = max(last_vitb_ep,
                 dv2_ep[-1] if dv2_ep else 0,
                 dv3_ep[-1] if dv3_ep else 0,
                 dv2_v2_ep[-1] if dv2_v2_ep else 0)
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


def plot_raw_only(vitb, dv2, dv3, dv2_v2, outfile):
    vitb_ep, vitb_top1 = vitb
    dv2_ep, dv2_raw, _ = dv2
    dv3_ep, dv3_raw, _ = dv3
    dv2_v2_ep, dv2_v2_raw, _ = dv2_v2

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(vitb_ep, vitb_top1, color="#888888", lw=2.2,
            label="ViT-B DINOv2 (reference, 100 ep)")
    ax.plot(dv2_ep, dv2_raw, color="#1f77b4", lw=2.4, marker="o", ms=5,
            label="ViT-L DINOv2 v1 (lr=5e-5)")
    ax.plot(dv3_ep, dv3_raw, color="#d62728", lw=2.4, marker="s", ms=5,
            label="ViT-L DINOv3 v1 (lr=5e-5)")
    if dv2_v2_ep:
        ax.plot(dv2_v2_ep, dv2_v2_raw, color="#2ca02c", lw=2.4, marker="^", ms=6,
                label="ViT-L DINOv2 v2 (lr=8e-5, resumed from E19)")

    max_ep = max(vitb_ep[-1], dv2_ep[-1], dv3_ep[-1],
                 dv2_v2_ep[-1] if dv2_v2_ep else 0)
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


def fallback_to_tuple(fallback_dict, offset=0):
    eps = sorted(fallback_dict.keys())
    raws = [fallback_dict[e][0] for e in eps]
    emas = [fallback_dict[e][1] for e in eps]
    return [e - offset for e in eps], raws, emas


def main():
    vitb = parse_vitb_summary(VITB_CSV)

    if DINOV2_LOG.exists():
        dv2 = parse_vitl_log(DINOV2_LOG)
    else:
        dv2 = fallback_to_tuple(DINOV2_V1_FALLBACK)
        print(f"  (DINOv2 v1 log missing, using fallback)")

    if DINOV3_LOG.exists():
        dv3 = parse_vitl_log(DINOV3_LOG)
    else:
        dv3 = fallback_to_tuple(DINOV3_V1_FALLBACK)
        print(f"  (DINOv3 v1 log missing, using fallback)")

    raw_by_ep, ema_by_ep = {}, {}
    any_log_found = False
    for log_path in DINOV2_V2_LOGS:
        if not log_path.exists():
            continue
        any_log_found = True
        v2_ep, v2_raw, v2_ema = parse_vitl_log(log_path)
        for e, r, m in zip(v2_ep, v2_raw, v2_ema):
            raw_by_ep[e + DINOV2_V2_OFFSET] = r
            ema_by_ep[e + DINOV2_V2_OFFSET] = m
    # Fill in any missing v2 E0-E19 from fallback
    for e, (r, m) in DINOV2_V2_EARLY_FALLBACK.items():
        if e not in raw_by_ep:
            raw_by_ep[e] = r
            ema_by_ep[e] = m
    if not any_log_found:
        print(f"  (DINOv2 v2 logs all missing, using fallback)")
    dv2_v2_ep_list = sorted(raw_by_ep.keys())
    dv2_v2_raw_list = [raw_by_ep[e] for e in dv2_v2_ep_list]
    dv2_v2_ema_list = [ema_by_ep[e] for e in dv2_v2_ep_list]
    dv2_v2 = (dv2_v2_ep_list, dv2_v2_raw_list, dv2_v2_ema_list)

    print(f"ViT-B reference: {len(vitb[0])} epochs (0..{vitb[0][-1]})")
    print(f"ViT-L DINOv2 v1: {len(dv2[0])} epochs (0..{dv2[0][-1] if dv2[0] else '?'}),"
          f" raw final={dv2[1][-1]:.2f}, ema final={dv2[2][-1]:.2f}")
    print(f"ViT-L DINOv3 v1: {len(dv3[0])} epochs (0..{dv3[0][-1] if dv3[0] else '?'}),"
          f" raw final={dv3[1][-1]:.2f}, ema final={dv3[2][-1]:.2f}")
    if dv2_v2_ep_list:
        print(f"ViT-L DINOv2 v2: {len(dv2_v2_ep_list)} epochs"
              f" (global {dv2_v2_ep_list[0]}..{dv2_v2_ep_list[-1]}),"
              f" raw final={dv2_v2_raw_list[-1]:.2f}, ema final={dv2_v2_ema_list[-1]:.2f}")

    plot_combined(vitb, dv2, dv3, dv2_v2, OUT_DIR / "training_curves.png")
    plot_raw_only(vitb, dv2, dv3, dv2_v2, OUT_DIR / "training_curves_raw.png")


if __name__ == "__main__":
    main()
