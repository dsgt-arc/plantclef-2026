"""
Generate 4-slide PowerPoint for PlantCLEF 2026 progress report (clean version).
"""

import io, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

SLIDE_W, SLIDE_H = 10.0, 5.625

# ── Colours ─────────────────────────────────────────────────────
BG_WHITE = RGBColor(0xFF,0xFF,0xFF); WHITE = BG_WHITE
BLACK  = RGBColor(0x1A,0x1A,0x2E); DARK = RGBColor(0x33,0x33,0x44)
GRAY   = RGBColor(0x66,0x66,0x77); LGRAY = RGBColor(0xAA,0xAA,0xBB)
TEAL   = RGBColor(0x00,0x96,0x88); ORANGE = RGBColor(0xE6,0x7E,0x22)
BLUE   = RGBColor(0x2E,0x86,0xC1); RED = RGBColor(0xC0,0x39,0x2B)
CARD_BG = RGBColor(0xF5,0xF6,0xFA); CARD_BDR = RGBColor(0xDD,0xDD,0xEE)

M_BG="#FFFFFF"; M_CARD="#F5F6FA"; M_TEAL="#009688"; M_ORANGE="#E67E22"
M_BLUE="#2E86C1"; M_RED="#C0392B"; M_BLACK="#1A1A2E"; M_DARK="#333344"
M_GRAY="#666677"; M_LGRAY="#AAAACC"

# ── Helpers ─────────────────────────────────────────────────────
def set_bg(s): f=s.background.fill; f.solid(); f.fore_color.rgb=BG_WHITE

def tb(s,l,t,w,h,txt,sz=14,b=False,c=BLACK,a=PP_ALIGN.LEFT,fn="Calibri"):
    bx=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf=bx.text_frame; tf.word_wrap=True; p=tf.paragraphs[0]; p.text=txt
    p.font.size=Pt(sz); p.font.bold=b; p.font.color.rgb=c; p.font.name=fn; p.alignment=a
    return tf

def bullets(s,l,t,w,h,items,sz=11,c=DARK,bc=TEAL):
    bx=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf=bx.text_frame; tf.word_wrap=True
    for i,txt in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        rb=p.add_run(); rb.text="\u25b8 "; rb.font.size=Pt(sz); rb.font.bold=True
        rb.font.color.rgb=bc; rb.font.name="Calibri"
        rt=p.add_run(); rt.text=txt; rt.font.size=Pt(sz); rt.font.color.rgb=c
        rt.font.name="Calibri"; p.space_after=Pt(4)

def rich(s,l,t,w,h,runs,a=PP_ALIGN.LEFT):
    bx=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf=bx.text_frame; tf.word_wrap=True; p=tf.paragraphs[0]; p.alignment=a
    for txt,sz,b,c in runs:
        r=p.add_run(); r.text=txt; r.font.size=Pt(sz); r.font.bold=b
        r.font.color.rgb=c; r.font.name="Calibri"

def card(s,l,t,w,h,fill=CARD_BG,bdr=CARD_BDR):
    sh=s.shapes.add_shape(1,Inches(l),Inches(t),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=fill; sh.line.color.rgb=bdr
    sh.line.width=Pt(0.5); sh.shadow.inherit=False

def fig_buf(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=220,bbox_inches="tight",
                facecolor=fig.get_facecolor(),edgecolor="none")
    plt.close(fig); buf.seek(0); return buf

def ax_clean(ax):
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color(M_LGRAY)
    ax.tick_params(colors=M_GRAY,labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

# ── Charts ──────────────────────────────────────────────────────

def chart_domain_gap():
    fig,ax=plt.subplots(figsize=(4.5,1.8),facecolor=M_BG)
    ax.set_facecolor(M_BG); ax.set_xlim(0,9); ax.set_ylim(0,2.2); ax.axis("off")
    r1=plt.Rectangle((0.1,0.2),3.5,1.8,facecolor=M_TEAL+"12",edgecolor=M_TEAL,lw=1.5,ls="--")
    ax.add_patch(r1)
    ax.text(1.85,1.75,"Training Data",ha="center",fontsize=9,color=M_TEAL,fontweight="bold")
    ax.text(1.85,1.15,"1.4M single-plant images\n~800px, 7806 species",ha="center",fontsize=8,color=M_DARK,linespacing=1.4)
    ax.annotate("",xy=(5.3,1.1),xytext=(3.8,1.1),arrowprops=dict(arrowstyle="-|>",color=M_RED,lw=2.5))
    ax.text(4.55,1.6,"Domain\nShift",ha="center",fontsize=8,color=M_RED,fontweight="bold",linespacing=1.2)
    r2=plt.Rectangle((5.4,0.2),3.5,1.8,facecolor=M_ORANGE+"12",edgecolor=M_ORANGE,lw=1.5,ls="--")
    ax.add_patch(r2)
    ax.text(7.15,1.75,"Test Data",ha="center",fontsize=9,color=M_ORANGE,fontweight="bold")
    ax.text(7.15,1.15,"Multi-species vegetation\nplots ~3000px, 50\u00d750cm",ha="center",fontsize=8,color=M_DARK,linespacing=1.4)
    fig.tight_layout(pad=0.1); return fig_buf(fig)

def chart_scale_bars():
    schemes=["A\n4\u00d74\n(16)","D\n4+5\n(41)","F\n4+6\n(52)",
             "E\n4+5+6\n(77)","K\n+7\u00d77\n(126)"]
    f1=[0.35698,0.37894,0.37507,0.38466,0.38542]
    colors=[M_GRAY,M_BLUE,M_BLUE,M_TEAL,M_GRAY]
    fig,ax=plt.subplots(figsize=(4.0,2.4),facecolor=M_BG)
    ax.set_facecolor(M_CARD)
    bars=ax.bar(schemes,f1,color=colors,edgecolor="none",width=0.6)
    bars[3].set_edgecolor(M_TEAL); bars[3].set_linewidth(2)
    ax.axhline(0.35698,color=M_LGRAY,ls="--",lw=0.8,label="Baseline (4\u00d74)")
    ax.set_ylabel("F1",color=M_DARK,fontsize=9)
    ax.set_ylim(0.345,0.395); ax_clean(ax)
    for i,v in enumerate(f1):
        ax.text(i,v+0.001,f"{v:.3f}",ha="center",fontsize=7.5,
                color=M_TEAL if i==3 else M_DARK,fontweight="bold")
    ax.legend(fontsize=7,loc="upper left",facecolor=M_BG,edgecolor=M_LGRAY)
    fig.tight_layout(pad=0.4); return fig_buf(fig)

def chart_alpha():
    alpha=[0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60]
    f1=[0.40031,0.40100,0.39997,0.39929,0.39768,0.39268,0.39004,0.38331,0.38358]
    fig,ax=plt.subplots(figsize=(4.0,2.4),facecolor=M_BG)
    ax.set_facecolor(M_CARD)
    ax.plot(alpha,f1,color=M_ORANGE,lw=2,marker="o",markersize=5,
            markeredgecolor=M_DARK,markeredgewidth=0.4)
    ax.scatter([0.05],[0.40100],c=M_TEAL,s=100,zorder=6,edgecolors=M_BLACK,lw=1,marker="*")
    ax.annotate("\u03b1=0.05, F1=0.401",xy=(0.05,0.40100),xytext=(0.22,0.403),
                fontsize=8,color=M_TEAL,fontweight="bold",
                arrowprops=dict(arrowstyle="->",color=M_TEAL,lw=1))
    ax.set_xlabel("\u03b1 (model weight in temporal fusion)",color=M_DARK,fontsize=8)
    ax.set_ylabel("F1",color=M_DARK,fontsize=9); ax_clean(ax)
    fig.tight_layout(pad=0.4); return fig_buf(fig)

def chart_beta():
    beta=[0.50,0.60,0.70,0.75,0.78,0.80,0.82,0.85,0.90,0.95,1.00]
    f1=[0.32797,0.34503,0.36727,0.38521,0.40355,0.41067,0.40640,0.40878,0.40675,0.40723,0.40100]
    fig,ax=plt.subplots(figsize=(4.0,2.6),facecolor=M_BG)
    ax.set_facecolor(M_CARD)
    ax.plot(beta,f1,color=M_BLUE,lw=2,marker="o",markersize=5,
            markeredgecolor=M_DARK,markeredgewidth=0.4)
    ax.scatter([0.80],[0.41067],c=M_TEAL,s=100,zorder=6,edgecolors=M_BLACK,lw=1,marker="*")
    ax.axhline(0.40100,color=M_LGRAY,ls="--",lw=0.8,label="No kNN (\u03b2=1.0)")
    ax.annotate("\u03b2=0.80, F1=0.411",xy=(0.80,0.41067),xytext=(0.60,0.416),
                fontsize=8,color=M_TEAL,fontweight="bold",
                arrowprops=dict(arrowstyle="->",color=M_TEAL,lw=1))
    ax.set_xlabel("\u03b2  (softmax weight;  1\u2212\u03b2 = kNN weight)",color=M_DARK,fontsize=8)
    ax.set_ylabel("F1",color=M_DARK,fontsize=9); ax.set_ylim(0.32,0.42); ax_clean(ax)
    ax.legend(fontsize=7,loc="lower left",facecolor=M_BG,edgecolor=M_LGRAY)
    fig.tight_layout(pad=0.4); return fig_buf(fig)

def chart_diff_beta():
    beta_s=[0.50,0.60,0.70,0.80,0.90,0.95,1.00]
    f1=[0.33607,0.35128,0.37229,0.41153,0.41236,0.41637,0.41754]
    fig,ax=plt.subplots(figsize=(4.0,2.6),facecolor=M_BG)
    ax.set_facecolor(M_CARD)
    ax.plot(beta_s,f1,color=M_ORANGE,lw=2,marker="s",markersize=5,
            markeredgecolor=M_DARK,markeredgewidth=0.4)
    ax.scatter([1.0],[0.41754],c=M_TEAL,s=100,zorder=6,edgecolors=M_BLACK,lw=1,marker="*")
    ax.annotate("\u03b2_single=1.0 (no kNN)\nF1=0.418",xy=(1.0,0.41754),xytext=(0.68,0.42),
                fontsize=8,color=M_TEAL,fontweight="bold",
                arrowprops=dict(arrowstyle="->",color=M_TEAL,lw=1))
    ax.axhline(0.41153,color=M_LGRAY,ls="--",lw=0.8,label="Uniform \u03b2=0.80")
    ax.set_xlabel("\u03b2_single  (softmax weight for single-visit images)",color=M_DARK,fontsize=8)
    ax.set_ylabel("F1",color=M_DARK,fontsize=9); ax.set_ylim(0.32,0.425); ax_clean(ax)
    ax.legend(fontsize=7,loc="lower right",facecolor=M_BG,edgecolor=M_LGRAY)
    fig.tight_layout(pad=0.4); return fig_buf(fig)

def chart_evolution():
    steps=["Exp-3\nBaseline","Exp-4\nmax_prob","+Temporal\nFusion","+JPEG85",
           "Exp-6\nThr Opt","Exp-7\nMulti-Scale","Exp-8\nRe-tune","Exp-9\nkNN"]
    f1=[0.28093,0.31950,0.33946,0.35517,0.35859,0.38466,0.40100,0.41754]
    colors=[M_GRAY,M_BLUE,M_BLUE,M_BLUE,M_ORANGE,M_TEAL,M_ORANGE,M_TEAL]
    fig,ax=plt.subplots(figsize=(6.5,2.8),facecolor=M_BG)
    ax.set_facecolor(M_CARD)
    bars=ax.bar(range(len(steps)),f1,color=colors,edgecolor="none",width=0.6)
    for i,(b,v) in enumerate(zip(bars,f1)):
        ax.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.3f}",
                ha="center",fontsize=7.5,color=M_BLACK,fontweight="bold")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps,fontsize=7,color=M_DARK)
    ax.set_ylabel("F1",color=M_DARK,fontsize=9); ax.set_ylim(0,0.47); ax_clean(ax)
    fig.tight_layout(pad=0.5); return fig_buf(fig)

# ══════════════════════════════════════════════════════════════════
prs=Presentation()
prs.slide_width=Inches(SLIDE_W); prs.slide_height=Inches(SLIDE_H)
blank=prs.slide_layouts[6]

# ── SLIDE 0: Problem Overview ───────────────────────────────────
s0=prs.slides.add_slide(blank); set_bg(s0)
tb(s0,0.5,0.25,9,0.4,"PlantCLEF 2026 \u2014 Problem Overview",sz=22,b=True,c=BLACK)
tb(s0,0.5,0.65,9,0.25,
   "Multi-label plant species identification in vegetation plot photographs",sz=10,c=GRAY)

# Domain gap (full width)
s0.shapes.add_picture(chart_domain_gap(),Inches(0.5),Inches(1.05),Inches(5.0),Inches(1.9))

# Challenge bullets (right of domain gap)
bullets(s0,5.8,1.1,3.8,1.8,[
    "Train on single-plant close-ups, test on multi-species plots",
    "7,806 candidate species \u2014 extreme long-tail distribution",
    "~2,105 test images; 76% multi-visit, 24% single-visit",
    "Evaluation: multi-label F1 (Kaggle public leaderboard)",
],sz=10,c=DARK)

# Our approach (bottom left)
tb(s0,0.5,3.2,4,0.3,"Our Approach",sz=14,b=True,c=TEAL)
bullets(s0,0.5,3.55,4.5,1.8,[
    "DINOv2 ViT-B/14 pretrained embeddings (no fine-tuning)",
    "Multi-scale tiling (77 tiles) to bridge resolution gap",
    "kNN retrieval + softmax ensemble with temporal location prior",
],sz=10,c=DARK)

# Current result (bottom right)
card(s0,6.5,3.2,3.0,1.8,fill=TEAL,bdr=TEAL)
tb(s0,6.5,3.35,3.0,0.3,"Current Best",sz=12,b=True,c=WHITE,a=PP_ALIGN.CENTER)
tb(s0,6.5,3.75,3.0,0.6,"F1 = 0.418",sz=26,b=True,c=WHITE,a=PP_ALIGN.CENTER)
tb(s0,6.5,4.4,3.0,0.3,"+48.6% over baseline (0.281)",sz=9,c=WHITE,a=PP_ALIGN.CENTER)


# ── SLIDE 1: Exp-6/7 — Threshold + Multi-Scale ─────────────────
s1=prs.slides.add_slide(blank); set_bg(s1)
tb(s1,0.5,0.25,9,0.4,"Threshold Optimization & Multi-Scale Tiling",sz=22,b=True,c=BLACK)
tb(s1,0.5,0.65,9,0.2,"Exp-6 / Exp-7 / Exp-8",sz=10,c=GRAY)

# Scale comparison chart (left)
tb(s1,0.5,0.95,4,0.25,"Tiling Scheme Comparison",sz=12,b=True,c=TEAL)
s1.shapes.add_picture(chart_scale_bars(),Inches(0.3),Inches(1.25),Inches(4.5),Inches(2.5))

# Alpha chart (right)
tb(s1,5.5,0.95,4,0.25,"Location Prior Weight (\u03b1)",sz=12,b=True,c=TEAL)
s1.shapes.add_picture(chart_alpha(),Inches(5.3),Inches(1.25),Inches(4.5),Inches(2.5))

# Key findings (bottom, full width)
tb(s1,0.5,4.0,9,0.25,"Key Findings",sz=12,b=True,c=TEAL)
bullets(s1,0.5,4.3,9,1.2,[
    "Fine scales (4\u00d74, 5\u00d75, 6\u00d76) capture individual plants; coarse scales (1\u00d71\u20133\u00d73) introduce false positives",
    "3 complementary scales outperform any 2 \u2014 optimal ~3 species/image, not 10",
    "77 tiles = more noise \u2192 location prior shifts from 40% to 95% of final prediction   |   F1: 0.357 \u2192 0.401 (+12.3%)",
],sz=10,c=DARK)


# ── SLIDE 2: Exp-9 — kNN Retrieval ─────────────────────────────
s2=prs.slides.add_slide(blank); set_bg(s2)
tb(s2,0.5,0.25,9,0.4,"kNN Retrieval Ensemble",sz=22,b=True,c=BLACK)
tb(s2,0.5,0.65,9,0.2,
   "Exp-9: FAISS-based kNN blended with softmax  |  F1: 0.401 \u2192 0.418 (+4.1%)",sz=10,c=GRAY)

# Beta sweep (left)
tb(s2,0.5,0.95,4,0.25,"Softmax vs kNN Weight",sz=12,b=True,c=TEAL)
s2.shapes.add_picture(chart_beta(),Inches(0.3),Inches(1.2),Inches(4.5),Inches(2.7))

# Differentiated beta (right)
tb(s2,5.5,0.95,4,0.25,"Single-Visit vs Multi-Visit",sz=12,b=True,c=TEAL)
s2.shapes.add_picture(chart_diff_beta(),Inches(5.3),Inches(1.2),Inches(4.5),Inches(2.7))

# Key findings (bottom)
tb(s2,0.5,4.1,9,0.25,"Key Findings",sz=12,b=True,c=TEAL)
bullets(s2,0.5,4.4,9,1.1,[
    "kNN only helps when temporal prior acts as safety net (multi-visit); pure softmax is best for single-visit images",
    "Optimal: \u03b2_multi=0.80 (20% kNN), \u03b2_single=1.00 (no kNN)  \u2014  FAISS IndexFlatIP, 860K embeddings, k=20",
    "Rare species boosting (inverse frequency weighting) hurts \u2014 too few quality training samples for tail species",
],sz=10,c=DARK)


# ── SLIDE 3: Summary ────────────────────────────────────────────
s3=prs.slides.add_slide(blank); set_bg(s3)
tb(s3,0.5,0.25,9,0.4,"Results Summary",sz=22,b=True,c=BLACK)
tb(s3,0.5,0.65,9,0.2,"F1: 0.281 \u2192 0.418 (+48.6% relative)  |  Public Leaderboard",sz=10,c=GRAY)

# Evolution chart (wide)
s3.shapes.add_picture(chart_evolution(),Inches(0.3),Inches(0.95),Inches(7.0),Inches(3.0))

# Technique impact (right)
tb(s3,7.6,0.95,2.2,0.25,"Top Techniques",sz=12,b=True,c=TEAL)
techs=[
    ("Multi-Scale Tiling","+0.044",TEAL),
    ("max_prob Aggregation","+0.039",BLUE),
    ("Temporal Fusion","+0.020",BLUE),
    ("kNN Ensemble","+0.017",ORANGE),
    ("JPEG85 Preproc","+0.016",BLUE),
]
y=1.3
for name,delta,color in techs:
    rich(s3,7.6,y,2.2,0.2,[(delta+"  ",11,True,color),(name,9,False,DARK)])
    y+=0.38

# Config (bottom left)
tb(s3,0.5,4.15,5,0.25,"Final Configuration",sz=12,b=True,c=TEAL)
cfg=["Tiling: 4\u00d74+5\u00d75+6\u00d76 (77 tiles)   Model: DINOv2 ViT-B/14",
     "kNN: FAISS k=20, \u03b2_multi=0.80, \u03b2_single=1.00",
     "Aggregation: max_prob \u2192 temporal fusion (\u03b1=0.15) \u2192 thr=0.100"]
yc=4.45
for line in cfg:
    tb(s3,0.6,yc,5.5,0.2,line,sz=9,c=DARK,fn="Consolas"); yc+=0.28

# Next steps (bottom right)
tb(s3,6.5,4.15,3.2,0.25,"Next Steps",sz=12,b=True,c=ORANGE)
bullets(s3,6.5,4.45,3.2,1.0,[
    "Knowledge distillation / noisy student",
    "Fine-tune DINOv2 (LoRA)",
    "Ensemble multiple architectures",
],sz=9.5,c=DARK,bc=ORANGE)


# ── Save ────────────────────────────────────────────────────────
out=r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\PlantCLEF2026_Progress_Slides.pptx"
prs.save(out); print(f"Saved: {out}")
