"""Generate PNG figures for Final_Pipeline_Methods_Summary.md.

Run from this directory:
    python generate_figures.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _annotate_bars(ax, bars, fmt="{:.4f}", dy=0.002, fontsize=8):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)


def fig_f1_progression():
    """Headline F1 progression across the major pipeline milestones."""
    milestones = [
        ("Exp-3/4\nbaseline", 0.28093, "#aab5be"),
        ("Exp-4\nmax_prob + T=1.5\n+ JPEG85 + temporal", 0.35698, "#4c8cbf"),
        ("Exp-6\nthreshold retune\n(rejected path)", 0.35859, "#d9a441"),
        ("Exp-7 / Exp-8\nmultiscale 4+5+6", 0.40100, "#4c8cbf"),
        ("Exp-9\n+ ViT kNN blend", 0.41754, "#4c8cbf"),
        ("Exp-11 Phase 3\nConvNeXt ensemble", 0.43540, "#4c8cbf"),
        ("Exp-11 FINAL\nw=0.77 α_cnxt=0.30", 0.43790, "#2a7a3a"),
    ]
    labels = [m[0] for m in milestones]
    scores = [m[1] for m in milestones]
    colors = [m[2] for m in milestones]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    bars = ax.bar(labels, scores, color=colors, edgecolor="#333", linewidth=0.5)
    _annotate_bars(ax, bars, fmt="{:.4f}", dy=0.004, fontsize=9)

    ax.set_ylabel("Public-leaderboard F1")
    ax.set_title("PlantCLEF 2026 — F1 Progression Across Experiment Pipeline")
    ax.set_ylim(0.25, 0.47)
    ax.axhline(0.28093, color="#aab5be", ls=":", lw=1)
    ax.axhline(0.43790, color="#2a7a3a", ls=":", lw=1)
    ax.grid(axis="y", ls=":", alpha=0.4)

    ax.annotate("", xy=(6, 0.43790), xytext=(0, 0.28093),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1))
    ax.text(3, 0.3, "Total gain:\n+0.157 F1\n(+55.8% rel.)",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8dc", ec="#b08900"))

    fig.savefig(os.path.join(FIG_DIR, "01_f1_progression.png"))
    plt.close(fig)


def fig_incremental_gain_waterfall():
    """Marginal (incremental) F1 gain from each accepted technique."""
    stages = [
        ("Exp-3 baseline\n(noisy_or, T=2.0)", 0.28093, None),
        ("+ max_prob aggregation\n(T=1.5, orig_only)", 0.03857, "+0.0386"),
        ("+ JPEG85 (q=85, 4:2:2)", 0.01600, "+0.0160"),
        ("+ temporal prior_blend\n(α=0.6)", 0.02148, "+0.0215"),
        ("+ threshold retune\n(0.04 → 0.100)", 0.04402, "+0.0440"),
        ("+ multiscale 4+5+6\n(77 tiles)", 0.00000, "(via above)"),
        ("+ kNN blend (β_m=0.80,\nβ_s=1.00, α=0.15)", 0.01654, "+0.0165"),
        ("+ ConvNeXt ensemble\n(w=0.77, α_cnxt=0.30)", 0.02036, "+0.0204"),
    ]

    running = 0.0
    starts, heights, colors, labels, totals = [], [], [], [], []
    for i, (lab, delta, annot) in enumerate(stages):
        labels.append(lab)
        if i == 0:
            starts.append(0.0)
            heights.append(delta)
            colors.append("#aab5be")
            running = delta
        else:
            starts.append(running)
            heights.append(delta)
            colors.append("#4c8cbf" if delta > 0 else "#c94c4c")
            running += delta
        totals.append(running)

    fig, ax = plt.subplots(figsize=(13, 6))
    xs = np.arange(len(labels))
    ax.bar(xs, heights, bottom=starts, color=colors, edgecolor="#333", linewidth=0.5, width=0.6)

    for x, lab, s, h, tot in zip(xs, labels, starts, heights, totals):
        if s == 0.0 and h > 0.1:
            ax.text(x, h / 2, f"base\n{h:.4f}", ha="center", va="center",
                    fontsize=9, color="#333")
        elif h == 0:
            ax.text(x, tot + 0.003, "(already in\nabove stage)",
                    ha="center", va="bottom", fontsize=8, color="#888")
        else:
            ax.text(x, tot + 0.003, f"+{h:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(x, tot - 0.01, f"F1={tot:.4f}", ha="center", va="top",
                fontsize=8, color="#333")

    ax.plot(xs, totals, "-o", color="#2a7a3a", lw=1.2, ms=4, alpha=0.55, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=8.5)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.48)
    ax.set_title("Waterfall: Incremental F1 Gain by Accepted Technique")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.savefig(os.path.join(FIG_DIR, "02_incremental_waterfall.png"))
    plt.close(fig)


def fig_multiscale_scheme():
    """Exp-7 scheme comparison bar chart."""
    schemes = [
        ("4x4\n(16)",          0.35698, "#aab5be"),
        ("4+5\n(41)",          0.37894, "#4c8cbf"),
        ("4+6\n(52)",          0.37507, "#4c8cbf"),
        ("4+5+2+1\n(46)",      0.37596, "#d9a441"),
        ("4+5+6\n(77) *",      0.38466, "#2a7a3a"),
        ("4+5+6+3\n(86)",      0.38227, "#d9a441"),
        ("4+5+6+7\n(126)",     0.38542, "#d9a441"),
        ("1+2+4\n(21)",        0.35460, "#c94c4c"),
    ]
    labels = [s[0] for s in schemes]
    vals = [s[1] for s in schemes]
    cols = [s[2] for s in schemes]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, vals, color=cols, edgecolor="#333", linewidth=0.5)
    _annotate_bars(ax, bars, fmt="{:.4f}", dy=0.001, fontsize=8)
    ax.axhline(0.35698, color="#aab5be", ls=":", lw=1, label="4x4 baseline (0.35698)")
    ax.set_ylabel("F1 (ViT single-model)")
    ax.set_xlabel("Scale scheme (total tile count)")
    ax.set_ylim(0.33, 0.395)
    ax.set_title("Exp-7: Multiscale Tiling — F1 by Scale Scheme  (ViT-B DINOv2)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.savefig(os.path.join(FIG_DIR, "03_multiscale_scheme.png"))
    plt.close(fig)


def fig_knn_beta_sweep():
    """Exp-9 beta-sweep showing kNN blend characteristics."""
    betas = [0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90, 0.95, 1.00]
    f1 =    [0.32797, 0.34503, 0.36727, 0.38521, 0.40355, 0.41067,
             0.40640, 0.40878, 0.40675, 0.40723, 0.40100]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(betas, f1, "-o", color="#4c8cbf", lw=1.5, ms=5, label="uniform β")
    ax.scatter([0.80], [0.41067], color="#2a7a3a", s=140, zorder=5,
               edgecolor="#333", linewidth=1, label="optimum (β=0.80)")
    ax.axhline(0.40100, color="#aab5be", ls=":", lw=1, label="pure softmax (no kNN)")
    ax.annotate("β=0.80\nF1=0.41067",
                xy=(0.80, 0.41067), xytext=(0.65, 0.415),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#2a7a3a"))

    for b, v in zip(betas, f1):
        ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="#555")

    ax.set_xlabel("β  (= softmax weight;  1-β = kNN weight)")
    ax.set_ylabel("F1")
    ax.set_title("Exp-9: kNN β Blend Sweep — Sharp Optimum at β=0.80")
    ax.set_ylim(0.32, 0.425)
    ax.grid(ls=":", alpha=0.4)
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "04_knn_beta_sweep.png"))
    plt.close(fig)


def fig_knn_differentiated_beta():
    """Exp-9 differentiated β_single with fixed β_multi=0.80."""
    beta_single = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
    f1 =          [0.32972, 0.33607, 0.35128, 0.37229, 0.41153,
                   0.41236, 0.41637, 0.41754]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(beta_single, f1, "-o", color="#8844aa", lw=1.5, ms=5)
    ax.scatter([1.00], [0.41754], color="#2a7a3a", s=140, zorder=5,
               edgecolor="#333", linewidth=1, label="optimum β_s=1.0")
    ax.axhline(0.41153, color="#aab5be", ls=":", lw=1, label="uniform β=0.80 baseline")
    ax.annotate("β_s=1.0 (pure softmax\nfor single-visit) best.",
                xy=(1.0, 0.41754), xytext=(0.7, 0.43),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#2a7a3a"))
    ax.set_xlabel("β_single  (β_multi fixed at 0.80)")
    ax.set_ylabel("F1")
    ax.set_title("Exp-9 Phase 3: Differentiated β — single-visit must use pure softmax")
    ax.set_ylim(0.32, 0.44)
    ax.grid(ls=":", alpha=0.4)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "05_knn_diff_beta.png"))
    plt.close(fig)


def fig_ensemble_weight_sweep():
    """Exp-11 Phase 4 fine weight sweep (w=ViT+kNN share)."""
    ws = [0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.82, 0.85]
    f1 = [0.4287, 0.4358, 0.4366, 0.4354, 0.4350, 0.4352, 0.4279, 0.4223]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ws, f1, "-o", color="#4c8cbf", lw=1.5, ms=5)
    ax.scatter([0.77], [0.4366], color="#2a7a3a", s=140, zorder=5,
               edgecolor="#333", linewidth=1, label="optimum w=0.77")
    ax.axhline(0.41754, color="#aab5be", ls=":", lw=1, label="ViT+kNN only (0.41754)")
    for w, v in zip(ws, f1):
        ax.annotate(f"{v:.4f}", (w, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("w = ViT+kNN weight   (ConvNeXt weight = 1 − w)")
    ax.set_ylabel("F1 (ensemble)")
    ax.set_title("Exp-11 Fine Weight Sweep — α_cnxt=0.05, thr=0.100")
    ax.set_ylim(0.418, 0.441)
    ax.grid(ls=":", alpha=0.4)
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "06_ensemble_weight_sweep.png"))
    plt.close(fig)


def fig_convnext_alpha_sweep():
    """Exp-11 α_cnxt sweep at w=0.77, thr=0.100."""
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    f1     = [0.4366, 0.4356, 0.4375, 0.4374, 0.4378, 0.4379, 0.4360]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(alphas, f1, "-o", color="#d9a441", lw=1.5, ms=5)
    ax.scatter([0.30], [0.4379], color="#2a7a3a", s=160, zorder=5,
               edgecolor="#333", linewidth=1, label="FINAL α=0.30 → F1=0.4379")
    for a, v in zip(alphas, f1):
        ax.annotate(f"{v:.4f}", (a, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("α_cnxt  (ConvNeXt temporal-fusion prior weight)")
    ax.set_ylabel("F1")
    ax.set_title("Exp-11 ConvNeXt α Sweep (w=0.77, thr=0.100) — Round 3 locks α=0.30")
    ax.set_ylim(0.434, 0.439)
    ax.grid(ls=":", alpha=0.4)
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "07_convnext_alpha_sweep.png"))
    plt.close(fig)


def fig_convnext_scales_standalone_vs_ensemble():
    """ConvNeXt: schemes differ for standalone vs ensemble optimum."""
    schemes = ["4+5+6\n(77)", "3+4+5+6\n(86)", "4+5+6+7\n(126)",
               "3+4+5+6+7\n(135)", "2+...+7\n(139)"]
    standalone = [0.3620, 0.3677, 0.3783, 0.3823, 0.3825]
    ensemble   = [0.4354, None,    0.4286, 0.4287, None]

    x = np.arange(len(schemes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 5))
    b1 = ax.bar(x - w/2, standalone, w, color="#d9a441", label="ConvNeXt standalone")
    ens_vals = [v if v is not None else 0 for v in ensemble]
    b2 = ax.bar(x + w/2, ens_vals, w, color="#2a7a3a",
                label="In ensemble with ViT+kNN (w=0.78)")
    for xi, v in zip(x, standalone):
        ax.text(xi - w/2, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    for xi, v in zip(x, ensemble):
        if v is None:
            ax.text(xi + w/2, 0.005, "n/a", ha="center", fontsize=8, color="#888")
        else:
            ax.text(xi + w/2, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_ylabel("F1")
    ax.set_title("ConvNeXt Scale Scheme: Standalone Optimum ≠ Ensemble Optimum\n"
                 "Matching ViT's 4+5+6 gives cleanest complementary signal")
    ax.set_ylim(0, 0.47)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "08_convnext_scales.png"))
    plt.close(fig)


def fig_pipeline_architecture():
    """Boxes-and-arrows final pipeline diagram (matplotlib-drawn)."""
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, face, ec="#222"):
        r = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=ec, lw=1.2,
                          zorder=2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=8.5, zorder=3)

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
                    zorder=1)

    box(2, 87, 20, 9, "Test image (3000×3000)\n2105 SW-Europe quadrats", "#eef3f7")
    box(2, 74, 20, 9, "JPEG round-trip\nq=85, subsampling 4:2:2", "#fff3cd")
    arrow(12, 87, 12, 83)

    box(2, 58, 20, 12, "Multiscale tiling\n4×4 + 5×5 + 6×6 = 77 tiles\nresize Lanczos → 518×518",
        "#eef3f7")
    arrow(12, 74, 12, 70)

    box(28, 63, 22, 9, "ViT-B DINOv2\n(Exp-13 finetuned,\nphase 1+2)", "#cfe2ff")
    box(55, 63, 22, 9, "ConvNeXt-L DINOv3\n(Exp-10 finetuned,\nphase 1+2)", "#f7d4a8")
    arrow(22, 65, 28, 67); arrow(22, 65, 55, 67)

    box(28, 50, 22, 8, "softmax(logits/T),  T=1.5", "#eef3f7")
    box(55, 50, 22, 8, "softmax(logits/T),  T=1.5", "#eef3f7")
    arrow(39, 63, 39, 58); arrow(66, 63, 66, 58)

    box(28, 38, 22, 9, "FAISS-IP  kNN  (k=20, τ=0.07)\nβ_multi=0.80,  β_single=1.00\np = β·softmax + (1−β)·knn",
        "#d4edda")
    arrow(39, 50, 39, 47)

    box(28, 26, 22, 8, "max_prob over 77 tiles\nimage_topk=64", "#eef3f7")
    box(55, 26, 22, 8, "max_prob over 77 tiles\nimage_topk=64", "#eef3f7")
    arrow(39, 38, 39, 34); arrow(66, 50, 66, 34)

    box(28, 14, 22, 9, "temporal prior_blend\nα_vit = 0.15", "#f3e8ff")
    box(55, 14, 22, 9, "temporal prior_blend\nα_cnxt = 0.30", "#f3e8ff")
    arrow(39, 26, 39, 23); arrow(66, 26, 66, 23)

    box(34, 1.5, 32, 9, "ENSEMBLE  (Approach A)\nP = 0.77·P_vit+knn + 0.23·P_cnxt",
        "#fcd5ce")
    arrow(39, 14, 46, 10.5); arrow(66, 14, 58, 10.5)

    box(73, 1.5, 25, 9, "threshold=0.100, topK=10\nF1 = 0.4379", "#b7e4b7")
    arrow(66, 6, 73, 6)

    # legend
    ax.text(80, 87, "Stages", fontsize=10, weight="bold")
    legend_items = [
        ("#fff3cd", "preprocessing"),
        ("#cfe2ff", "ViT branch"),
        ("#f7d4a8", "ConvNeXt branch"),
        ("#d4edda", "kNN (ViT only)"),
        ("#f3e8ff", "temporal fusion"),
        ("#fcd5ce", "ensemble"),
        ("#b7e4b7", "submission"),
    ]
    for i, (c, lab) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((80, 82 - 3*i), 3, 2, facecolor=c, edgecolor="#222"))
        ax.text(84, 83 - 3*i, lab, fontsize=8, va="center")

    ax.set_title("Exp-11 Final Pipeline — F1 = 0.4379  (Approach A: kNN on ViT first, "
                 "then probability-level blend with ConvNeXt)", fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, "09_pipeline_architecture.png"))
    plt.close(fig)


def fig_absolute_gain_stacked():
    """Absolute F1 stacked gain (cumulative)."""
    stages = [
        ("baseline (Exp-3)",                     0.28093),
        ("+ max_prob (T=1.5) ",                  0.31950),
        ("+ JPEG85 + temporal (α=0.6)",          0.35698),
        ("+ multiscale 4+5+6 (77 tiles)",        0.38466),
        ("+ alpha re-tune → 0.05, thr=0.100",    0.40100),
        ("+ kNN β_m=0.80 β_s=1.00 α=0.15",       0.41754),
        ("+ ConvNeXt ensemble (w=0.78 a=0.05)",  0.43540),
        ("+ final tuning (w=0.77 α_cnxt=0.30)",  0.43790),
    ]
    labels = [s[0] for s in stages]
    vals   = [s[1] for s in stages]
    deltas = [vals[0]] + [vals[i] - vals[0] for i in range(1, len(vals))]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(labels))
    b1 = ax.bar(xs, [vals[0]] * len(xs), color="#aab5be", label="Exp-3 baseline",
                edgecolor="#333", linewidth=0.5)
    b2 = ax.bar(xs, [d - vals[0] if i == 0 else d for i, d in enumerate(deltas)],
                bottom=[vals[0] if i > 0 else 0 for i in range(len(xs))],
                color="#4c8cbf", label="absolute gain above baseline",
                edgecolor="#333", linewidth=0.5)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)
        if xi > 0:
            delta = v - vals[0]
            ax.text(xi, vals[0] + (v - vals[0]) / 2, f"+{delta:.3f}",
                    ha="center", va="center", fontsize=8, color="white")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 0.48)
    ax.set_ylabel("F1")
    ax.set_title("Absolute F1 Gain vs. Exp-3 Baseline (cumulative view)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.savefig(os.path.join(FIG_DIR, "10_absolute_gain.png"))
    plt.close(fig)


def fig_rejected_techniques():
    """Rejected techniques and their F1 deltas vs their baseline."""
    rejected = [
        ("TTA (all_tiles, 6-view, hflip+rot90)", -0.008),
        ("top-M avg (m≥2)", -0.028),
        ("noisy_or aggregation", -0.025),
        ("adaptive thr (max_gap, elbow)", -0.100),
        ("tile-level blend before kNN (App B)", -0.021),
        ("retune kNN on blended tiles (App C)", -0.020),
        ("inverse-frequency kNN weighting", -0.018),
        ("confidence-gated kNN", 0.000),
        ("border crop 3%", -0.032),
        ("border crop 5%", -0.043),
        ("border crop 7%", -0.037),
        ("soft distillation (KL loss)", -0.14),
        ("ConvNeXt extra scales (3+7) in ensemble", -0.007),
    ]
    labels = [r[0] for r in rejected]
    vals   = [r[1] for r in rejected]
    colors = ["#c94c4c" if v < -0.02 else "#d9a441" if v < -0.005 else "#aab5be"
              for v in vals]

    fig, ax = plt.subplots(figsize=(11, 6))
    ys = np.arange(len(labels))[::-1]
    ax.barh(ys, vals, color=colors, edgecolor="#333", linewidth=0.5)
    for y, v in zip(ys, vals):
        ax.text(v - 0.002 if v < 0 else 0.001, y, f"{v:+.3f}",
                va="center", ha="right" if v < 0 else "left", fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("F1 delta vs its local baseline")
    ax.set_title("Rejected Techniques — why they did not enter the Exp-11 final pipeline")
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.savefig(os.path.join(FIG_DIR, "11_rejected_techniques.png"))
    plt.close(fig)


def fig_technique_contribution_pie():
    """Contribution share of each accepted technique to the +0.157 F1 total gain."""
    contribs = {
        "max_prob + T=1.5": 0.03857,
        "JPEG85 + temporal prior_blend": 0.03748,
        "threshold + alpha retune": 0.01634,
        "multiscale 4+5+6 (77 tiles)": 0.02400,
        "kNN blend (β_m/β_s differentiated)": 0.01654,
        "ConvNeXt ensemble + final tuning": 0.02036,
    }
    labels = list(contribs.keys())
    vals = list(contribs.values())
    total = sum(vals)
    pcts = [v / total * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4c8cbf", "#d9a441", "#8844aa", "#2a7a3a", "#5db1b1", "#c94c4c"]
    wedges, _, autotexts = ax.pie(vals, labels=labels, colors=colors,
                                   autopct=lambda p: f"{p:.0f}%\n(+{p*total/100:.4f})",
                                   startangle=90, pctdistance=0.72,
                                   textprops=dict(fontsize=9))
    for at in autotexts:
        at.set_fontsize(8.5)
    ax.set_title(f"Contribution Share of Each Accepted Technique\n"
                 f"to Total +{total:.4f} F1 Gain (Exp-3 → Exp-11)")
    fig.savefig(os.path.join(FIG_DIR, "12_contribution_share.png"))
    plt.close(fig)


def main():
    fig_f1_progression()
    fig_incremental_gain_waterfall()
    fig_multiscale_scheme()
    fig_knn_beta_sweep()
    fig_knn_differentiated_beta()
    fig_ensemble_weight_sweep()
    fig_convnext_alpha_sweep()
    fig_convnext_scales_standalone_vs_ensemble()
    fig_pipeline_architecture()
    fig_absolute_gain_stacked()
    fig_rejected_techniques()
    fig_technique_contribution_pie()
    print(f"Saved 12 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
