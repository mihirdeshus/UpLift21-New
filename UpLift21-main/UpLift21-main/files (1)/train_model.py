"""
train_model.py
UpLift21 — Model training pipeline.
Trains RandomForest, XGBoost, LightGBM, CatBoost.
Applies RobustScaler only to non-tree models.
Saves best model + artifacts. Generates evaluation plots.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

Path("models").mkdir(exist_ok=True)
Path("assets").mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#080C12"
CARD    = "#0D1218"
BORDER  = "#1A2332"
ACCENT  = "#00C6A7"
WARN    = "#F5A623"
DANGER  = "#FF4D6A"
TEXT    = "#C8D6E5"
MUTED   = "#4A6A8A"
COLORS  = [ACCENT, WARN, DANGER, "#7B8FFF"]

def style_ax(ax, grid=True):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if grid:
        ax.grid(color=BORDER, linewidth=0.5, linestyle="--", alpha=0.7)

# ── Load data ─────────────────────────────────────────────────────────────────
data = pd.read_csv("data/prenatal_dataset.csv")
X = data.drop("ds_risk_class", axis=1)
y = data["ds_risk_class"]
feature_names = list(X.columns)
classes = [0, 1, 2]
class_labels = ["Low", "Moderate", "High"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaler (for models that need it — not used on tree-based models)
scaler = RobustScaler()
scaler.fit(X_train)

# ── Model definitions ─────────────────────────────────────────────────────────
model_configs = {
    "RandomForest": {
        "model": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "needs_scaling": False,
    },
    "XGBoost": {
        "model": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        ),
        "needs_scaling": False,
    },
    "LightGBM": {
        "model": LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        ),
        "needs_scaling": False,
    },
    "CatBoost": {
        "model": CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        ),
        "needs_scaling": False,
    },
}

# ── Train & evaluate all models ───────────────────────────────────────────────
results = {}
best_model_obj  = None
best_model_name = ""
best_acc        = 0.0

print(f"\n{'Model':<20} {'Accuracy':>10} {'CV Mean':>10} {'CV Std':>8}")
print("-" * 52)

for name, cfg in model_configs.items():
    m   = cfg["model"]
    Xtr = scaler.transform(X_train) if cfg["needs_scaling"] else X_train.values
    Xte = scaler.transform(X_test)  if cfg["needs_scaling"] else X_test.values

    m.fit(Xtr, y_train.values)
    preds = m.predict(Xte)
    probs = m.predict_proba(Xte)
    acc   = accuracy_score(y_test, preds)

    cv = cross_val_score(m, Xtr, y_train.values, cv=StratifiedKFold(3), scoring="accuracy")

    # One-vs-rest AUC
    y_bin = label_binarize(y_test, classes=classes)
    ova_auc = np.mean([
        auc(*roc_curve(y_bin[:, i], probs[:, i])[:2])
        for i in range(len(classes))
    ])

    results[name] = {
        "model":    m,
        "acc":      acc,
        "cv_mean":  cv.mean(),
        "cv_std":   cv.std(),
        "auc":      ova_auc,
        "probs":    probs,
        "preds":    preds,
        "needs_scaling": cfg["needs_scaling"],
    }

    print(f"{name:<20} {acc:>10.4f} {cv.mean():>10.4f} {cv.std():>8.4f}")

    if acc > best_acc:
        best_acc        = acc
        best_model_obj  = m
        best_model_name = name

print(f"\nBest model: {best_model_name}  (accuracy={best_acc:.4f})")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(best_model_obj,  "models/best_model.pkl")
joblib.dump(scaler,          "models/scaler.pkl")
joblib.dump(best_model_name, "models/model_name.pkl")
joblib.dump(results,         "models/all_results.pkl")

# Save comparison table
comp = pd.DataFrame([
    {
        "Model":    n,
        "Accuracy": f"{v['acc']:.4f}",
        "CV Mean":  f"{v['cv_mean']:.4f}",
        "CV Std":   f"{v['cv_std']:.4f}",
        "ROC-AUC":  f"{v['auc']:.4f}",
    }
    for n, v in results.items()
])
comp.to_csv("assets/model_comparison.csv", index=False)

print("Artifacts saved to models/")

# ── Plot 1: ROC Curves (one-vs-rest, best model) ──────────────────────────────
best_probs = results[best_model_name]["probs"]
y_bin      = label_binarize(y_test, classes=classes)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle(f"ROC Curves — {best_model_name}", color=TEXT, fontsize=13, fontweight="600", y=1.02)

for i, (cls_label, color) in enumerate(zip(class_labels, COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], best_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    ax = axes[i]
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=BORDER, linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.07, color=color)
    ax.set_title(f"{cls_label} Risk", fontsize=11)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=MUTED, fontsize=9)
    style_ax(ax)

plt.tight_layout()
plt.savefig("assets/roc_curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("assets/roc_curve.png saved")

# ── Plot 2: Feature Importance ────────────────────────────────────────────────
if hasattr(best_model_obj, "feature_importances_"):
    importances = best_model_obj.feature_importances_
else:
    importances = np.abs(best_model_obj.coef_).mean(axis=0)

importances = importances / importances.sum()
sorted_idx  = np.argsort(importances)
s_names     = [feature_names[i] for i in sorted_idx]
s_vals      = importances[sorted_idx]
bar_colors  = [ACCENT if v == s_vals.max() else "#1E7A6B" for v in s_vals]

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
bars = ax.barh(s_names, s_vals, color=bar_colors, height=0.55)
ax.set_xlabel("Relative Importance")
ax.set_title(f"Feature Importance — {best_model_name}", color=TEXT, fontsize=12, fontweight="600")

for bar, val in zip(bars, s_vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            fontsize=8, color=MUTED,
            fontfamily="monospace")

style_ax(ax)
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("assets/feature_importance.png saved")

# ── Plot 3: Model Comparison Bar Chart ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle("Model Comparison", color=TEXT, fontsize=13, fontweight="600", y=1.02)

model_names_list = list(results.keys())
accs  = [results[n]["acc"] for n in model_names_list]
aucs  = [results[n]["auc"] for n in model_names_list]

for ax, vals, metric in zip(axes, [accs, aucs], ["Accuracy", "ROC-AUC (OvR)"]):
    bar_c = [ACCENT if v == max(vals) else "#1E4A5A" for v in vals]
    brs   = ax.bar(model_names_list, vals, color=bar_c, width=0.5)
    ax.set_ylim(min(vals) - 0.05, 1.02)
    ax.set_title(metric, fontsize=11)
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(brs, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color=MUTED, fontfamily="monospace")
    style_ax(ax)

plt.tight_layout()
plt.savefig("assets/model_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("assets/model_comparison.png saved")

# ── Confusion matrix (best model) ────────────────────────────────────────────
import seaborn as sns
cm  = confusion_matrix(y_test, results[best_model_name]["preds"])
fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlOrRd",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=ax,
    linewidths=0.5,
    linecolor=BG,
    cbar_kws={"shrink": 0.8},
)
ax.set_title(f"Confusion Matrix — {best_model_name}", color=TEXT, fontsize=12, fontweight="600")
ax.set_xlabel("Predicted", color=MUTED)
ax.set_ylabel("Actual", color=MUTED)
ax.tick_params(colors=MUTED)
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("assets/confusion_matrix.png saved")

print("\nTraining complete.")
print(comp.to_string(index=False))
