import numpy as np
import matplotlib.pyplot as plt

# =========================
# Concept figure: SMA vs EMA
# =========================

# Example discrete PV signal
x = np.arange(1, 9)
p = np.array([4.0, 4.8, 3.9, 5.5, 6.2, 5.7, 7.1, 6.5])

# Window shown for concept
window_idx = np.array([3, 4, 5, 6])  # selected window points
window_x = x[window_idx]
window_p = p[window_idx]

# SMA weights: equal
sma_weights = np.ones(len(window_x)) / len(window_x)

# EMA weights: higher for recent samples
ema_weights = np.array([0.10, 0.18, 0.28, 0.44])
ema_weights = ema_weights / ema_weights.sum()

# Weighted outputs
sma_value = np.sum(window_p * sma_weights)
ema_value = np.sum(window_p * ema_weights)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# -------------------------
# Left: SMA
# -------------------------
ax = axes[0]
ax.plot(x, p, marker="o", linewidth=1.8, label="PV raw power")
ax.scatter(window_x, window_p, s=260, alpha=0.35, label="Samples in window")

# equal weight labels
for xi, yi, wi in zip(window_x, window_p, sma_weights):
    if xi == window_x[-1]:  # titik terakhir di window, biar tidak keluar atas
        ax.text(xi, yi - 0.45, f"w={wi:.2f}", ha="center", va="top", fontsize=9)
    else:
        ax.text(xi, yi + 0.45, f"w={wi:.2f}", ha="center", va="bottom", fontsize=9)

ax.hlines(
    sma_value,
    window_x.min(),
    window_x.max(),
    linestyles="--",
    linewidth=2.2,
    label="SMA output"
)

ax.set_title("SMA: Equal Weighting in Fixed Window", fontsize=12, weight="bold")
ax.set_xlabel("Time step")
ax.set_ylabel("Power")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc="lower right")

# -------------------------
# Right: EMA
# -------------------------
ax = axes[1]
ax.plot(x, p, marker="o", linewidth=1.8, label="PV raw power")

# marker size follows EMA weight
sizes = 700 * ema_weights / ema_weights.max()
ax.scatter(window_x, window_p, s=sizes, alpha=0.45, label="Recent samples weighted more")

# EMA weight labels
for xi, yi, wi in zip(window_x, window_p, ema_weights):
    if xi == window_x[-1]:  # titik terakhir di window, bobot terbesar
        ax.text(xi, yi - 0.45, f"w={wi:.2f}", ha="center", va="top", fontsize=9)
    else:
        ax.text(xi, yi + 0.45, f"w={wi:.2f}", ha="center", va="bottom", fontsize=9)

ax.hlines(
    ema_value,
    window_x.min(),
    window_x.max(),
    linestyles="--",
    linewidth=2.2,
    label="EMA output"
)

ax.set_title("EMA: Higher Weight to Recent Data", fontsize=12, weight="bold")
ax.set_xlabel("Time step")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc="lower right")

# General title
fig.suptitle("Conceptual Difference Between SMA and EMA", fontsize=15, weight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("fig_sma_ema_concept.png", dpi=300, bbox_inches="tight")
plt.savefig("fig_sma_ema_concept.svg", bbox_inches="tight")
plt.show()