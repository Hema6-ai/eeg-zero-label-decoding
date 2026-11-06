"""Regenerates figures/pipeline_overview.png to match Algorithm 1 after the
52.53% confidence-weighted-fusion number was removed pending independent
reproduction (TMLR revision round 2, task 1.3)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

boxes = [
    ("1. Canonicalize",
     "Resample 128 Hz, 22-ch montage,\nper-trial z-score, fix LOSO protocol",
     "#ececec", "#555555"),
    ("2. Sweep baseline families",
     "SSL (raw EEG) / geometric DL / classical\nRiemannian  →  best non-learned: FgMDM 47.45%",
     "#e6edf8", "#3b5fa0"),
    ("3. Diagnose the bottleneck",
     "Disentanglement ablation + CSP probe/t-SNE/Fisher\n→ representation quality is the limiting factor",
     "#e6edf8", "#3b5fa0"),
    ("4. Add domain adaptation",
     "MEKT-lite (JDA, tangent space), default settings\n→ 48.48%",
     "#e8f3e8", "#3f8f4f"),
    ("5. Tune signal-quality levers",
     "Analysis window t*=[0.3,2.5]s (50.41%);\ncorrelation covariance estimator (51.22%)",
     "#fcefe3", "#c96a1e"),
    ("6. Ensemble",
     "4 windows × 2 estimators, soft-voted\n(52.39% — selected on these same subjects, optimistic)",
     "#fbe9e9", "#b23b3b"),
    ("7. Stress-test",
     "10 further extensions (filter-banks, stacking,\npseudo-labeling, …) — none improve on ensemble",
     "#eae8f6", "#5d4f9e"),
    ("8. Validate honestly",
     "Fully nested LOSO (re-search per fold, source\nsubjects only) → 49.94%, 2.4–3.1pp below the\n"
     "in-sample figure; PhysioNet replication (51.95%)",
     "#eae8f6", "#5d4f9e"),
]

n = len(boxes)
base_box_h = 1.15
extra_line_h = 0.32
box_heights = [base_box_h + extra_line_h * max(0, body.count("\n") - 1) for _, body, _, _ in boxes]
gap = 0.42
fig_h = sum(box_heights) + (n - 1) * gap + 1.1
fig_w = 8.0

fig, ax = plt.subplots(figsize=(fig_w, fig_h * 0.72))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

y = fig_h - 1.0
centers = []
for (title, body, face, edge), box_h in zip(boxes, box_heights):
    box = FancyBboxPatch((0.3, y - box_h), fig_w - 0.6, box_h,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.6, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(0.55, y - 0.28, title, fontsize=13, fontweight="bold", va="top", ha="left")
    ax.text(0.55, y - 0.58, body, fontsize=10.3, va="top", ha="left", color="#222222")
    centers.append((y, y - box_h))
    y -= box_h + gap

for (y_top_prev, y_bot_prev), (y_top_next, _) in zip(centers[:-1], centers[1:]):
    ax.annotate("", xy=(fig_w / 2, y_top_next), xytext=(fig_w / 2, y_bot_prev),
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.8))

return_y = centers[-1][1] - 0.55
ax.annotate("", xy=(fig_w / 2, centers[-1][1]), xytext=(fig_w / 2, return_y + 0.05),
            arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.8))
ax.text(fig_w / 2, return_y,
        "Return: 8-member MEKT-lite ensemble\n"
        "49.94% honestly-nested zero-label LOSO accuracy (κ ≈ 0.333)\n"
        "(52.39% selected on these same subjects, optimistic),\n"
        "leakage-checked, cross-dataset replicated",
        fontsize=10.8, style="italic", ha="center", va="top")

ax.set_ylim(return_y - 0.9, fig_h)
plt.tight_layout()
plt.savefig("figures/pipeline_overview.png", dpi=220, bbox_inches="tight")
print("wrote figures/pipeline_overview.png")
