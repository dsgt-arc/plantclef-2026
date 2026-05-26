"""Habitat-fit diagram — 6in x 4in, all text >=8pt for legibility."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

OUT_PNG = Path(__file__).parent / "habitat_fit_diagram.png"

C_IN = "#3a7bd5"
C_OUT = "#34a853"
C_GEO = "#e08a3c"
C_ALT = "#9b59b6"
C_INK = "#1d1d1f"
C_MUTE = "#36363D"  # darkened from #5b5b62 for sharper print
C_HAIR = "#d2d2d7"


def prob_bars(ax, x0, y0, w, h, values, color, dim_idx=None):
    n = len(values)
    bar_w = w / n * 0.7
    gap = w / n * 0.3
    maxv = values.max()
    for i, v in enumerate(values):
        bx = x0 + i * (bar_w + gap) + gap / 2
        bh = h * v / maxv
        c = "#b0b0b6" if dim_idx is not None and i in dim_idx else color
        ax.add_patch(Rectangle((bx, y0), bar_w, bh,
                               facecolor=c, edgecolor="none"))


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # --- title ---
    ax.text(60, 76, "Habitat fit",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_INK)
    ax.text(60, 71, "rescale fused probabilities by source geography",
            ha="center", va="center",
            fontsize=8.5, color=C_MUTE)

    # --- top row: equation ---
    rng = np.random.default_rng(11)
    vals_in = rng.uniform(0.35, 1.0, size=6)
    f_vals = np.array([1.0, 0.35, 1.0, 0.10, 0.7, 1.0])
    vals_out = vals_in * f_vals
    dim = {i for i, fv in enumerate(f_vals) if fv < 1.0}

    bar_y, bar_h = 50, 11
    bar_w = 22
    prob_bars(ax, 4, bar_y, bar_w, bar_h, vals_in, C_IN)
    prob_bars(ax, 42, bar_y, bar_w, bar_h, f_vals, "#c8c8cd")
    prob_bars(ax, 80, bar_y, bar_w, bar_h, vals_out, C_OUT, dim_idx=dim)

    for bx in (4, 42, 80):
        ax.plot([bx, bx + bar_w], [bar_y - 0.4, bar_y - 0.4],
                color=C_HAIR, lw=0.7)

    # β and 1 indicators on the f-vector
    f_x = 42
    beta_y = bar_y + bar_h * 0.10
    ax.plot([f_x, f_x + bar_w], [bar_y + bar_h, bar_y + bar_h],
            color=C_MUTE, lw=0.6, linestyle=":")
    ax.plot([f_x, f_x + bar_w], [beta_y, beta_y],
            color="#c0392b", lw=0.7, linestyle="--")
    ax.text(f_x + bar_w + 0.8, bar_y + bar_h, "1",
            fontsize=8, color=C_MUTE, va="center")
    ax.text(f_x + bar_w + 0.8, beta_y, r"$\beta$",
            fontsize=8.5, color="#c0392b", va="center", fontweight="bold")

    # labels above
    ax.text(15, bar_y + bar_h + 3.2, r"$P_{\mathrm{fused}}$",
            ha="center", fontsize=11, color=C_IN, fontweight="bold")
    ax.text(53, bar_y + bar_h + 3.2, r"$f_{s,c}$",
            ha="center", fontsize=11, color=C_INK, fontweight="bold")
    ax.text(91, bar_y + bar_h + 3.2, r"$P_{\mathrm{hab}}$",
            ha="center", fontsize=11, color=C_OUT, fontweight="bold")

    # operators
    ax.text(34, bar_y + bar_h / 2, r"$\times$",
            ha="center", va="center", fontsize=18, color=C_INK)
    ax.text(72, bar_y + bar_h / 2, r"$=$",
            ha="center", va="center", fontsize=18, color=C_INK)

    # divider hairline
    ax.plot([8, 112], [42, 42], color=C_HAIR, lw=0.6)

    # ---------- bottom-left: geo ----------
    geo_cx, geo_cy = 18, 20
    geo_r = 12

    ax.text(4, 37, r"$\mathrm{geo}_{s,c}$",
            fontsize=10, color=C_GEO, fontweight="bold")
    ax.text(15, 37, "centroid & radius",
            fontsize=8.5, color=C_MUTE, va="center")

    ax.add_patch(Circle((geo_cx, geo_cy), geo_r,
                        facecolor=C_GEO, alpha=0.10,
                        edgecolor=C_GEO, lw=1.0))
    ax.plot(geo_cx, geo_cy, marker="*", markersize=13,
            color=C_GEO, markeredgecolor="white", markeredgewidth=0.8)

    rng2 = np.random.default_rng(3)
    n_in = 12
    a_in = rng2.uniform(0, 2*np.pi, n_in)
    r_in = np.sqrt(rng2.uniform(0.06, 0.82, n_in)) * geo_r
    ax.scatter(geo_cx + r_in*np.cos(a_in), geo_cy + r_in*np.sin(a_in),
               s=11, color=C_GEO, alpha=0.9, edgecolors="none")
    a_out = rng2.uniform(0, 2*np.pi, 3)
    r_out = rng2.uniform(geo_r * 1.18, geo_r * 1.32, 3)
    ax.scatter(geo_cx + r_out*np.cos(a_out), geo_cy + r_out*np.sin(a_out),
               s=10, color="#bdbdc2", alpha=0.9, edgecolors="none")

    # radius indicator bar + tag
    ax.annotate("", xy=(geo_cx + geo_r, geo_cy),
                xytext=(geo_cx, geo_cy),
                arrowprops=dict(arrowstyle="-", color=C_GEO, lw=1.2))
    ax.text(geo_cx + geo_r / 2, geo_cy + 1.4, "250 km",
            ha="center", fontsize=8.5, color=C_GEO, fontweight="bold")

    # callout text to right of circle
    ax.text(36, 27, r"$n_{\mathrm{near}} \geq 40$",
            fontsize=9.5, color=C_INK, fontweight="bold")
    ax.text(36, 23.5, r"$\rightarrow$  fit $= 1$",
            fontsize=9, color=C_MUTE)
    ax.text(36, 16, r"$n_{\mathrm{near}} = 0$",
            fontsize=9.5, color=C_INK, fontweight="bold")
    ax.text(36, 12.5, r"$\rightarrow$  fit $= \beta$",
            fontsize=9, color=C_MUTE)
    ax.text(36, 6, r"$\beta = 0.10$",
            fontsize=9, color=C_GEO, fontweight="bold")

    # ---------- bottom-right: alt ----------
    ax.text(60, 37, r"$\mathrm{alt}_{s,c}$",
            fontsize=10, color=C_ALT, fontweight="bold")
    ax.text(70, 37, "elevation band",
            fontsize=8.5, color=C_MUTE, va="center")

    # altitude axis
    ax_x = 68
    y_lo, y_hi = 8, 33
    ax.plot([ax_x, ax_x], [y_lo, y_hi], color=C_MUTE, lw=0.9)
    for ay, lab in [(y_lo, "0"), ((y_lo + y_hi)/2, "1.5k"), (y_hi, "3k")]:
        ax.plot([ax_x - 0.7, ax_x + 0.7], [ay, ay], color=C_MUTE, lw=0.9)
        ax.text(ax_x - 1.5, ay, lab, fontsize=8, color=C_MUTE,
                ha="right", va="center")

    # bands
    band_lo, band_hi = 25, 31
    margin_lo = band_lo - (band_hi - band_lo) * 0.5
    margin_hi = band_hi + (band_hi - band_lo) * 0.5
    ax.add_patch(Rectangle((ax_x + 1, margin_lo),
                            42, margin_hi - margin_lo,
                            facecolor=C_ALT, alpha=0.08, edgecolor="none"))
    ax.add_patch(Rectangle((ax_x + 1, band_lo),
                            42, band_hi - band_lo,
                            facecolor=C_ALT, alpha=0.24, edgecolor="none"))
    ax.text(ax_x + 2.5, (band_lo + band_hi)/2,
            r"$[\mathrm{alt}_{\mathrm{lo}},\, \mathrm{alt}_{\mathrm{hi}}]$",
            fontsize=8.5, color=C_ALT, va="center", fontweight="bold")
    ax.text(ax_x + 43, margin_hi - 0.4, r"$\pm 50\%$",
            fontsize=8, color=C_ALT, va="top", ha="right", style="italic")

    # three species dots
    species = [
        (98, 28, "1.0"),
        (105, 20, "0.6"),
        (112, 11, r"$\beta$"),
    ]
    for sx, sy, lab in species:
        ax.plot(sx, sy, marker="o", markersize=8,
                color=C_ALT, markeredgecolor="white", markeredgewidth=1.0)
        ax.text(sx, sy + 2.2, lab, ha="center", fontsize=9,
                color=C_ALT, fontweight="bold")

    ax.text(96, 3.5,
            r"in band $\rightarrow 1$     $>$1 km out $\rightarrow \beta$",
            ha="center", fontsize=8.5, color=C_MUTE)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.005)
    plt.savefig(OUT_PNG, dpi=220, facecolor="white",
                bbox_inches="tight", pad_inches=0.04)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
