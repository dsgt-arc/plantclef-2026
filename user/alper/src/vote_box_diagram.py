import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

M_ORANGE="#E67E22"; M_DARK="#333344"; M_BG="#FFFFFF"

fig, ax = plt.subplots(figsize=(3, 2.2), facecolor=M_BG)
ax.set_xlim(0, 3); ax.set_ylim(0, 2.2); ax.set_aspect("equal"); ax.axis("off")

box = mpatches.FancyBboxPatch((0.15, 0.15), 2.7, 1.9, boxstyle="round,pad=0.1",
    facecolor=M_ORANGE+"18", edgecolor=M_ORANGE, lw=2)
ax.add_patch(box)

ax.text(1.5, 1.8, "Species Vote", ha="center", fontsize=13, fontweight="bold", color=M_ORANGE)
ax.text(1.5, 1.3, "Festuca rubra: 60%", ha="center", fontsize=11, color=M_DARK)
ax.text(1.5, 0.9, "Trifolium repens: 20%", ha="center", fontsize=11, color=M_DARK)
ax.text(1.5, 0.5, "Poa pratensis: 10%", ha="center", fontsize=11, color=M_DARK)

fig.savefig(
    r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\vote_box_diagram.png",
    dpi=250, facecolor=M_BG, edgecolor="none", bbox_inches="tight")
plt.close(fig)
print("Saved.")
