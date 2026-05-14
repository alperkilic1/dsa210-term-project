"""
Regenerate flagged figures from FIGURES-AUDIT.md.
No model re-training -- reconstructs from data with same random_state.
Target: 1200x800 px at 300 DPI for print quality.

Figures regenerated:
  1. 02c_duration_violin_by_impact.png -- clip y>=0, boxplot for small groups
  2. ml_confusion_matrix.png -- diverging colormap (blue=correct, red=incorrect)
  3. h3_severity_updates.png -- grouped bars instead of overlapping histograms
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

FIG_DIR = Path("figures")
DPI = 300
W_INCH = 1200 / DPI
H_INCH = 800 / DPI

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams["figure.dpi"] = DPI

df = pd.read_csv("data/incidents_clean.csv")
print(f"Loaded {len(df)} rows from data/incidents_clean.csv")


# --- Figure 1: Violin plot (fixed negative y-axis) ---
print("\n[1/3] Regenerating 02c_duration_violin_by_impact.png ...")

impact_order = ["none", "minor", "major", "critical"]
existing_impacts = [i for i in impact_order if i in df["impact"].values]

fig, ax = plt.subplots(figsize=(W_INCH * 1.25, H_INCH))

for i, imp in enumerate(existing_impacts):
    subset = df[df["impact"] == imp]["duration_minutes"]
    if len(subset) < 30:
        bp = ax.boxplot(
            subset.values, positions=[i], widths=0.4,
            patch_artist=True, showfliers=True
        )
        color = sns.color_palette("magma", len(existing_impacts))[i]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        parts = ax.violinplot(
            subset.values, positions=[i], showmedians=False,
            showextrema=False
        )
        color = sns.color_palette("magma", len(existing_impacts))[i]
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        q1, med, q3 = subset.quantile([0.25, 0.5, 0.75])
        ax.vlines(i, q1, q3, color="white", linewidth=2)
        ax.scatter([i], [med], color="white", s=30, zorder=5)

sns.stripplot(
    data=df, x="impact", y="duration_minutes",
    order=existing_impacts, ax=ax,
    color="white", size=3, alpha=0.4, jitter=True
)

ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(bottom=0)
ax.set_title("Duration by Impact Severity", fontsize=11, fontweight="bold")
ax.set_xlabel("Impact Level", fontsize=10)
ax.set_ylabel("Duration (minutes)", fontsize=10)
ax.set_xticks(range(len(existing_impacts)))
ax.set_xticklabels([f"{imp.capitalize()}" for imp in existing_impacts])

for i, imp in enumerate(existing_impacts):
    subset = df[df["impact"] == imp]["duration_minutes"]
    med = subset.median()
    n = len(subset)
    ax.annotate(
        f"med={med:.0f}m\nn={n}",
        xy=(i, med), fontsize=7, ha="center", va="bottom",
        color="#ffd93d", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5)
    )

plt.tight_layout()
fig.savefig(FIG_DIR / "02c_duration_violin_by_impact.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved 02c_duration_violin_by_impact.png (y >= 0, boxplot for n<30)")


# --- Figure 2: Confusion matrix (diverging colormap) ---
print("\n[2/3] Regenerating ml_confusion_matrix.png ...")

df["is_business_hours"] = df["created_hour"].between(9, 16).astype(int)
df["is_weekend"] = df["created_weekday"].isin(["Saturday", "Sunday"]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["created_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["created_hour"] / 24)

feature_cols = [
    "service", "created_hour", "created_weekday", "first_hour_updates",
    "is_business_hours", "is_weekend", "hour_sin", "hour_cos",
]
categorical_features = ["service", "created_weekday"]
numeric_features = [
    "created_hour", "first_hour_updates", "is_business_hours",
    "is_weekend", "hour_sin", "hour_cos",
]

X = df[feature_cols]
y = df["duration_class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5
    )),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=["short", "long"])

fig, ax = plt.subplots(figsize=(W_INCH, H_INCH))

cm_display = np.array(cm, dtype=float)
mask_correct = np.eye(2, dtype=bool)
cm_signed = np.where(mask_correct, cm_display, -cm_display)

im = ax.imshow(
    cm_signed, cmap="RdYlBu", aspect="auto",
    vmin=-cm_display.max(), vmax=cm_display.max()
)

for i in range(2):
    for j in range(2):
        color = "white" if abs(cm_signed[i, j]) > cm_display.max() * 0.6 else "black"
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=18, fontweight="bold", color=color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["short", "long"], fontsize=10)
ax.set_yticklabels(["short", "long"], fontsize=10)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
ax.set_title("Confusion Matrix - RF Baseline (Test Set, n=141)", fontsize=11)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Correct (blue) vs Incorrect (red)", fontsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "ml_confusion_matrix.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved ml_confusion_matrix.png (diverging RdYlBu colormap)")


# --- Figure 3: h3_severity_updates (grouped bars) ---
print("\n[3/3] Regenerating h3_severity_updates.png ...")

from scipy.stats import kruskal

severity_order = ["none", "minor", "major", "critical"]
df_h3 = df.dropna(subset=["first_hour_updates"]).copy()
df_h3["first_hour_updates"] = df_h3["first_hour_updates"].astype(int)

groups = {}
for level in severity_order:
    g = df_h3[df_h3["impact"] == level]["first_hour_updates"]
    if len(g) > 0:
        groups[level] = g

group_arrays = [groups[k] for k in groups]
k_groups = len(group_arrays)
n_total = sum(len(g) for g in group_arrays)

if k_groups >= 2:
    h_stat, p_val3 = kruskal(*group_arrays)
    eps_sq = (h_stat - k_groups + 1) / (n_total - k_groups) if n_total > k_groups else 0
else:
    p_val3, eps_sq = 1.0, 0.0

test_colors = {
    "none": "#4a90e2", "minor": "#2ecc71",
    "major": "#f39c12", "critical": "#e74c3c",
}

max_fhu = int(df_h3["first_hour_updates"].max()) if len(df_h3) else 1
bins = np.arange(0, max_fhu + 1)
active_levels = [lv for lv in severity_order if lv in groups]
n_levels = len(active_levels)
bar_width = 0.8 / n_levels

fig, ax = plt.subplots(figsize=(W_INCH * 1.25, H_INCH))

for idx, level in enumerate(active_levels):
    counts, _ = np.histogram(groups[level].values, bins=np.arange(-0.5, max_fhu + 1.5, 1))
    positions = bins + (idx - n_levels / 2 + 0.5) * bar_width
    ax.bar(
        positions, counts, width=bar_width, alpha=0.85,
        label=f"{level.capitalize()} (n={len(groups[level])})",
        color=test_colors[level], edgecolor="white", linewidth=0.5
    )

ax.set_xlabel("Updates in first hour", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_xticks(range(0, max_fhu + 1))
ax.set_title(
    f"H3: First-hour updates by severity (p={p_val3:.4f}, eps-sq={eps_sq:.3f})",
    fontsize=10, fontweight="bold"
)
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(FIG_DIR / "h3_severity_updates.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved h3_severity_updates.png (grouped bars, no overlap)")


# --- Summary ---
print("\n=== Regeneration Complete ===")
import os
for fname in [
    "02c_duration_violin_by_impact.png",
    "ml_confusion_matrix.png",
    "h3_severity_updates.png",
]:
    path = FIG_DIR / fname
    from PIL import Image
    img = Image.open(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {fname}: {img.size[0]}x{img.size[1]}, {size_kb:.0f} KB")
