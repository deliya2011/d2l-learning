"""Sigmoid and its derivative."""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-6, 6, 400)
s = 1 / (1 + np.exp(-x))
ds = s * (1 - s)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(x, s, color="#1f77b4", lw=2, label="sigmoid(x)")
ax.plot(x, ds, color="#d62728", lw=1.6, ls="--", label="sigmoid'(x)")
ax.axhline(0, color="black", lw=0.6)
ax.axhline(0.25, color="#d62728", lw=0.6, ls=":")

# mark the maximum of the derivative
ax.plot([0], [0.25], "o", color="#d62728", ms=4)
ax.annotate("max 0.25", xy=(0, 0.25), xytext=(1.2, 0.31),
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

ax.set_xlabel("x")
ax.set_ylabel("value")
ax.set_title("sigmoid and its derivative")
ax.set_xlim(-6, 6)
ax.set_ylim(-0.05, 1.1)
ax.legend(loc="center right", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("notes/figs/b08-sigmoid.png", dpi=150)
print("saved notes/figs/b08-sigmoid.png")
