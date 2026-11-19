"""CAICP 模拟卷配图：世界模型闭环示意。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

def box(x, y, w, h, text, fc="#e8f0fe"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec="#1a56db", lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

box(0.4, 2.6, 2.0, 1.2, "Environment")
box(3.2, 2.6, 2.2, 1.2, "World Model\n(predict next state)")
box(6.2, 2.6, 2.0, 1.2, "Controller\n(policy)")
box(3.2, 0.4, 2.2, 1.2, "Action\n(a_t)", "#fff4e5")

def arrow(x1, y1, x2, y2, label, off=(0, 0.15)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=14, lw=1.2, color="#333"))
    ax.text((x1+x2)/2 + off[0], (y1+y2)/2 + off[1], label, fontsize=9, ha="center")

arrow(2.4, 3.2, 3.2, 3.2, "observation")
arrow(5.4, 3.2, 6.2, 3.2, "state")
arrow(7.2, 2.6, 4.3, 1.6, "action")
arrow(4.3, 1.6, 2.4, 2.6, "affects")
ax.text(5.0, 4.6, "World-model loop", fontsize=11, ha="center")
fig.tight_layout()
fig.savefig("CAICP&NOAI/daily/2026-11-19/figs/worldmodel-loop.png", dpi=150)
print("saved")
