"""
Clean diagram showing FAISS-based kNN retrieval: query embedding → index → nearest neighbors → species vote.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

M_TEAL="#009688"; M_BLUE="#2E86C1"; M_ORANGE="#E67E22"; M_PURPLE="#8E44AD"
M_RED="#C0392B"; M_DARK="#333344"; M_GRAY="#888899"; M_LGRAY="#CCCCDD"
M_BG="#FFFFFF"; M_GREEN="#27AE60"

def rbox(ax, x, y, w, h, text, color, fs=9, sub=None):
    b = mpatches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",
        facecolor=color+"18",edgecolor=color,lw=1.8)
    ax.add_patch(b)
    ty = y+h/2+(0.12 if sub else 0)
    ax.text(x+w/2,ty,text,ha="center",va="center",fontsize=fs,
            fontweight="bold",color=M_DARK,linespacing=1.3)
    if sub:
        ax.text(x+w/2,y+h/2-0.18,sub,ha="center",va="center",
                fontsize=fs-2,color=M_GRAY)

def arr(ax, x1,y1,x2,y2, c=M_DARK, lw=1.5):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>",color=c,lw=lw))

def plant_tile(ax, x, y, sz, shade, border=M_LGRAY):
    rect = plt.Rectangle((x,y),sz,sz,facecolor="#D4C5A9",edgecolor=border,lw=0.8)
    ax.add_patch(rect)
    np.random.seed(hash(shade)%200)
    for _ in range(6):
        cx=x+np.random.uniform(0.15,0.85)*sz
        cy=y+np.random.uniform(0.15,0.85)*sz
        r=np.random.uniform(0.04,0.1)*sz
        ax.add_patch(plt.Circle((cx,cy),r,facecolor=shade,edgecolor="none",alpha=0.65))

fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=M_BG)
ax.set_xlim(-0.3, 11.5)
ax.set_ylim(-0.8, 5.0)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5.0, 4.7, "FAISS-based kNN Retrieval",
        ha="center", fontsize=15, fontweight="bold", color=M_DARK)

# ── 1. Query tile ──
plant_tile(ax, 0.0, 1.8, 1.0, "#2ECC71", border=M_BLUE)
ax.text(0.5, 1.55, "Query Tile", ha="center", fontsize=9, fontweight="bold", color=M_BLUE)

# ── 2. DINOv2 ──
arr(ax, 1.15, 2.3, 1.7, 2.3, c=M_DARK)
rbox(ax, 1.75, 1.8, 1.4, 1.0, "DINOv2\nViT-B/14", M_BLUE)
ax.text(2.45, 1.55, "768d embedding", ha="center", fontsize=7, color=M_GRAY)

# ── 3. FAISS index ──
arr(ax, 3.25, 2.3, 3.9, 2.3, c=M_PURPLE, lw=1.8)
ax.text(3.57, 2.55, "cosine\nsearch", ha="center", fontsize=7, color=M_PURPLE, linespacing=1.2)

fx, fy, fw, fh = 4.0, 1.4, 2.2, 1.8
box = mpatches.FancyBboxPatch((fx,fy),fw,fh,boxstyle="round,pad=0.1",
    facecolor=M_PURPLE+"0D",edgecolor=M_PURPLE,lw=2)
ax.add_patch(box)
ax.text(fx+fw/2, fy+fh-0.22, "FAISS Index", ha="center",
        fontsize=11, fontweight="bold", color=M_PURPLE)
ax.text(fx+fw/2, fy+fh-0.5, "IndexFlatIP", ha="center", fontsize=7.5, color=M_GRAY)

# Tiny embedding bars inside
np.random.seed(99)
bar_colors = [M_GREEN, M_TEAL, M_ORANGE, M_BLUE, M_RED, M_PURPLE]
for row in range(3):
    for col in range(7):
        bx = fx + 0.18 + col * 0.28
        by = fy + 0.15 + row * 0.38
        c = bar_colors[np.random.randint(len(bar_colors))]
        ax.add_patch(plt.Rectangle((bx,by),0.14,0.28,
                     facecolor=c+"35",edgecolor=c+"60",lw=0.4))

ax.text(fx+fw/2, fy-0.35, "860K training embeddings",
        ha="center", fontsize=8, fontweight="bold", color=M_PURPLE)

# ── 4. Top-k neighbors ──
arr(ax, 6.3, 2.3, 6.9, 2.3, c=M_PURPLE, lw=1.8)
ax.text(6.6, 2.55, "k = 20", ha="center", fontsize=8, fontweight="bold", color=M_PURPLE)

# Neighbor tiles with labels and similarity scores
nn_data = [
    ("#27AE60", "Festuca rubra",    "0.94"),
    ("#1ABC9C", "Festuca rubra",    "0.91"),
    ("#229954", "Trifolium repens",  "0.87"),
    ("#2ECC71", "Festuca rubra",    "0.84"),
    ("#16A085", "Poa pratensis",    "0.81"),
]

for i, (shade, species, sim) in enumerate(nn_data):
    ty = 3.7 - i * 0.75
    plant_tile(ax, 7.0, ty, 0.5, shade)
    ax.text(7.65, ty + 0.35, species, fontsize=8, color=M_DARK,
            va="center", fontstyle="italic")
    ax.text(7.65, ty + 0.12, f"sim = {sim}", fontsize=7, color=M_GRAY, va="center")

ax.text(8.3, -0.05, "\u22ee  +15 more", ha="center", fontsize=8, color=M_LGRAY)

# ── 5. Vote result ──
arr(ax, 9.4, 2.3, 9.6, 2.3, c=M_ORANGE, lw=1.5)

# Vote summary box
rbox(ax, 9.65, 1.45, 1.9, 1.7, "", M_ORANGE)
ax.text(10.6, 2.85, "Species Vote", ha="center", fontsize=9,
        fontweight="bold", color=M_ORANGE)
ax.text(10.6, 2.35, "Festuca rubra: 60%", ha="center", fontsize=7.5, color=M_DARK)
ax.text(10.6, 2.05, "Trifolium repens: 20%", ha="center", fontsize=7.5, color=M_DARK)
ax.text(10.6, 1.75, "Poa pratensis: 10%", ha="center", fontsize=7.5, color=M_DARK)

fig.savefig(
    r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\knn_diagram.png",
    dpi=250, facecolor=M_BG, edgecolor="none", bbox_inches="tight")
plt.close(fig)
print("Saved.")
