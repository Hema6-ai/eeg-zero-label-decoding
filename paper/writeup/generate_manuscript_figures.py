"""Generates manuscript figures from experiment results (progression chart, full method
comparison, per-subject heatmap). Reuses the existing 6 representational-analysis figures
already in paper/writeup/figures/ as-is.

    python -m paper.writeup.generate_manuscript_figures
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SUBJECTS = list(range(1, 10))

# ---------------------------------------------------------------------------
# Figure: accuracy progression across the session's key milestones
# ---------------------------------------------------------------------------
stages = [
    "FgMDM\n(default window)",
    "MEKT-lite\n(default window)",
    "MEKT-lite\n(tuned window)",
    "Multi-window\nMEKT ensemble",
    "8-member\n(window x estimator)\nensemble",
]
stage_acc = [47.45, 48.48, 50.41, 51.16, 52.39]
stage_std = [13.42, 13.49, 14.41, 14.74, 14.92]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(stages)))
bars = ax.bar(stages, stage_acc, yerr=stage_std, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(25, color="gray", linestyle=":", linewidth=1, label="4-class chance (25%)")
ax.axhline(58.68, color="firebrick", linestyle="--", linewidth=1.2, label="Calibrated EEGNet reference (58.68%)")
for bar, acc in zip(bars, stage_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5, f"{acc:.2f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Zero-shot LOSO accuracy (%)")
ax.set_ylim(0, 75)
ax.set_title("Every gain came from signal quality, not method complexity")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
plt.xticks(fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "progression_chart.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# Figure: per-subject accuracy heatmap across representative methods
# ---------------------------------------------------------------------------
methods = [
    "EEGNet\n(calibrated)",
    "Riemannian\nTransformer v2",
    "SPDNet",
    "FgMDM\n(default window)",
    "MEKT-lite\n(tuned window)",
    "8-member\nensemble (best)",
]
data = np.array([
    [70.83, 43.06, 83.68, 38.19, 37.50, 43.75, 69.10, 73.61, 68.40],  # EEGNet
    [51.04, 29.34, 54.86, 35.94, 33.16, 35.07, 34.72, 55.90, 51.39],  # RT v2
    [53.47, 24.13, 64.41, 39.93, 30.21, 35.07, 44.97, 54.69, 50.17],  # SPDNet
    [56.94, 26.04, 67.71, 44.97, 35.42, 37.33, 39.76, 64.93, 53.99],  # FgMDM default window
    [60.59, 32.29, 76.04, 43.92, 35.42, 41.84, 38.89, 65.62, 59.03],  # MEKT tuned window
    [63.02, 31.25, 77.60, 47.40, 36.11, 42.01, 44.44, 69.10, 60.59],  # 8-member ensemble
])

fig, ax = plt.subplots(figsize=(8, 4.8))
im = ax.imshow(data, cmap="RdYlGn", vmin=20, vmax=85, aspect="auto")
ax.set_xticks(range(len(SUBJECTS)))
ax.set_xticklabels([f"S{s}" for s in SUBJECTS])
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=8.5)
for i in range(len(methods)):
    for j in range(len(SUBJECTS)):
        ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                 color="black" if 35 < data[i, j] < 70 else "white")
ax.set_xlabel("Held-out subject (BCI-IV 2a)")
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Accuracy (%)", fontsize=8)
ax.set_title("Subject 2 never crosses ~32% under any zero-shot method")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "per_subject_heatmap.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# Figure: full method comparison, all zero-shot results, sorted
# ---------------------------------------------------------------------------
all_methods = [
    ("MAE-only", 27.84, 2.29, "Learned (raw EEG)"),
    ("MAE+GRL", 27.47, 2.76, "Learned (raw EEG)"),
    ("MAE+CLUB+SCIN+VIB", 26.68, 2.35, "Learned (raw EEG)"),
    ("MAE+EA", 25.87, 1.78, "Learned (raw EEG)"),
    ("Per-class EA+TS+LDA", 25.41, 3.37, "Classical Riemannian"),
    ("RPA (tangent-space Procrustes)", 42.48, 14.82, "Domain adaptation"),
    ("EA+FBCSP+SVM", 39.47, 11.13, "Classical Riemannian"),
    ("Riemannian Transformer v2", 42.38, 10.02, "Learned (geometric)"),
    ("Riemannian Transformer v1", 43.40, 10.33, "Learned (geometric)"),
    ("SPDNet", 44.12, 12.21, "Learned (geometric)"),
    ("EA+MDM", 43.81, 10.97, "Classical Riemannian"),
    ("Band-ensemble TS+LR+EA", 45.37, 12.04, "Classical Riemannian"),
    ("EA+CSP+SVM", 44.62, 10.26, "Classical Riemannian"),
    ("EA+CSP+LDA", 45.18, 11.93, "Classical Riemannian"),
    ("MEKT-lite (untuned)", 46.26, 11.24, "Domain adaptation"),
    ("EA+FgMDM", 47.45, 13.42, "Classical Riemannian"),
    ("MEKT-lite (tuned, k=50)", 48.48, 13.49, "Domain adaptation"),
    ("MEKT-lite (tuned window)", 50.41, 14.41, "Domain adaptation"),
    ("Multi-window MEKT ensemble", 51.16, 14.74, "Ensemble"),
    ("8-member ensemble (best)", 52.39, 14.92, "Ensemble"),
]
all_methods.sort(key=lambda x: x[1])
names = [m[0] for m in all_methods]
accs = [m[1] for m in all_methods]
stds = [m[2] for m in all_methods]
cats = [m[3] for m in all_methods]
cat_colors = {"Learned (raw EEG)": "#d62728", "Learned (geometric)": "#ff7f0e",
              "Classical Riemannian": "#1f77b4", "Domain adaptation": "#2ca02c", "Ensemble": "#9467bd"}
bar_colors = [cat_colors[c] for c in cats]

fig, ax = plt.subplots(figsize=(8, 8))
y_pos = np.arange(len(names))
ax.barh(y_pos, accs, xerr=stds, color=bar_colors, edgecolor="black", linewidth=0.4, capsize=2)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.axvline(25, color="gray", linestyle=":", linewidth=1)
ax.axvline(58.68, color="firebrick", linestyle="--", linewidth=1.2)
ax.set_xlabel("Zero-shot LOSO accuracy (%)")
ax.set_xlim(0, 75)
ax.set_title("Representative zero-shot methods tried, sorted by accuracy")
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=c, label=l) for l, c in cat_colors.items()]
ax.legend(handles=legend_elems, loc="lower right", fontsize=7.5, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "all_methods_comparison.png"), dpi=200)
plt.close()


# ---------------------------------------------------------------------------
# Figure: forest plot of paired statistical tests across the progression
# ---------------------------------------------------------------------------
comparisons = [
    "FgMDM $\\to$ MEKT (default)",
    "MEKT (default) $\\to$ +window",
    "+window $\\to$ +corr estimator",
    "+corr $\\to$ 8-member ensemble",
    "FgMDM (baseline) $\\to$ ensemble (final)",
]
mean_diff = [1.02, 1.93, 0.81, 1.18, 4.94]
ci_lo = [-0.17, 0.41, -0.79, -0.15, 3.38]
ci_hi = [2.26, 3.59, 3.05, 2.30, 6.58]
pvals = [0.250, 0.055, 1.000, 0.129, 0.0039]
significant = [p < 0.05 for p in pvals]

fig, ax = plt.subplots(figsize=(7.5, 4))
y_pos = np.arange(len(comparisons))[::-1]
colors = ["#2ca02c" if s else "#7f7f7f" for s in significant]
for y, lo, hi, m, c in zip(y_pos, ci_lo, ci_hi, mean_diff, colors):
    ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
    ax.plot(m, y, "o", color=c, markersize=8, markeredgecolor="black", markeredgewidth=0.6)
ax.axvline(0, color="black", linestyle=":", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(comparisons, fontsize=9)
for y, m, p in zip(y_pos, mean_diff, pvals):
    ax.text(6.9, y, f"+{m:.2f}pp, p={p:.3f}", va="center", fontsize=8)
ax.set_xlabel("Mean paired accuracy difference (percentage points)")
ax.set_xlim(-1.5, 10)
ax.set_title("Only the cumulative gain is statistically significant at $n{=}9$")
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color="#2ca02c", marker="o", linestyle="-", markersize=7, label="$p<0.05$"),
    Line2D([0], [0], color="#7f7f7f", marker="o", linestyle="-", markersize=7, label="not significant"),
]
ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "statistical_tests_forest.png"), dpi=200, bbox_inches="tight")
plt.close()

print("Figures written to", FIG_DIR)
print(" - progression_chart.png")
print(" - per_subject_heatmap.png")
print(" - all_methods_comparison.png")
print(" - statistical_tests_forest.png")
