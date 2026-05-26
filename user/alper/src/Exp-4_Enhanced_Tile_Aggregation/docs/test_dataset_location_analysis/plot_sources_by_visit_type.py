"""Plot quadrat counts per source, split by multi-visit vs single-visit sites."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(r"C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\PlantCLEF2025_test.csv")
OUT_DIR = Path(__file__).parent
OUT_PNG = OUT_DIR / "sources_by_visit_type.png"

KNOWN_SOURCES = [
    "CBN-PdlC", "CBN-Pla", "CBN-can", "CBN-Pyr",
    "RNNB", "LISAH", "GUARDEN", "OPTMix", "2024-CEV3",
]


def parse_source(qid: str) -> str:
    for src in KNOWN_SOURCES:
        if qid.startswith(src):
            return src
    return qid.split("-")[0]


def parse_site(qid: str) -> str:
    return qid.rsplit("-", 1)[0]


def main() -> None:
    df = pd.read_csv(CSV_PATH, sep=";")
    df["source"] = df["quadrat_id"].map(parse_source)
    df["site"] = df["quadrat_id"].map(parse_site)

    site_visits = df.groupby("site").size()
    multi_sites = set(site_visits[site_visits > 1].index)

    df["visit_type"] = np.where(df["site"].isin(multi_sites), "multi-visit", "single-visit")

    agg = (
        df.groupby(["source", "visit_type"]).size().unstack(fill_value=0)
    )
    for col in ("multi-visit", "single-visit"):
        if col not in agg.columns:
            agg[col] = 0
    agg["total"] = agg["multi-visit"] + agg["single-visit"]
    agg = agg.sort_values("total", ascending=True)

    sources = agg.index.tolist()
    multi = agg["multi-visit"].to_numpy()
    single = agg["single-visit"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    color_multi = "#3b6fb6"
    color_single = "#5aa572"

    bars_multi = ax.barh(sources, multi, color=color_multi, label="multi-visit site")
    bars_single = ax.barh(sources, single, left=multi, color=color_single, label="single-visit site")

    totals = multi + single
    for i, total in enumerate(totals):
        ax.text(total + max(totals) * 0.01, i, f"{int(total)}",
                va="center", fontsize=9, fontweight="bold")

    for i, (m, s) in enumerate(zip(multi, single)):
        if m >= max(totals) * 0.05:
            ax.text(m / 2, i, f"{int(m)}", va="center", ha="center",
                    fontsize=8, color="white")
        if s >= max(totals) * 0.05:
            ax.text(m + s / 2, i, f"{int(s)}", va="center", ha="center",
                    fontsize=8, color="white")

    ax.set_xlabel("Number of quadrats")
    ax.set_title("Quadrats by Source, Split by Multi-visit vs Single-visit Sites")
    ax.set_xlim(0, max(totals) * 1.1)
    ax.legend(loc="lower right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")

    print("\nPer-source breakdown:")
    print(agg[["multi-visit", "single-visit", "total"]].sort_values("total", ascending=False))


if __name__ == "__main__":
    main()
