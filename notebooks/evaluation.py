"""
=================================================================
 EVALUATION DASHBOARD — Visual model & data evaluation
=================================================================
Generates docs/evaluation_dashboard.png summarising data patterns,
model performance, and business-level outputs (ABC classification,
stock evolution, feature importance).
=================================================================
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import json
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3,
})
GREEN, MINT, GOLD, RED, BLUE = "#1B4332", "#52B788", "#F59E0B", "#EF4444", "#3B82F6"


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    training = pd.read_csv(os.path.join(PROC_DIR, "training_dataset.csv"), parse_dates=["date"])
    sales = pd.read_csv(os.path.join(PROC_DIR, "sales_monthly.csv"), parse_dates=["date"])
    gb = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    features = [f for f in meta["features"] if f in training.columns]

    test_df = training[training["year"] == 2020]
    X_test = test_df[features].fillna(0)
    y_test = test_df["quantity_sold"]
    y_pred = np.maximum(0, gb.predict(X_test))

    fig = plt.figure(figsize=(18, 13), facecolor="white")
    fig.suptitle("PharmaPredict — Model & Data Evaluation Dashboard",
                 fontsize=15, fontweight="bold", color=GREEN, y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.40)

    # 1. Revenue by therapeutic class over time
    ax1 = fig.add_subplot(gs[0, :2])
    raw = pd.read_csv(os.path.join(RAW_DIR, "pharma_sales_history.csv"))
    raw = raw[raw["Sales"] > 0]
    month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                 'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    raw["date"] = pd.to_datetime(dict(year=raw["Year"], month=raw["Month"].map(month_map), day=1))
    rev_by_class = raw.groupby(["date", "Product Class"])["Sales"].sum().unstack().fillna(0)
    colors = [GREEN, MINT, GOLD, RED, BLUE, "#8B5CF6"]
    for col, c in zip(rev_by_class.columns, colors):
        ax1.plot(rev_by_class.index, rev_by_class[col] / 1e6, color=c, lw=1.3, label=col)
    ax1.axvline(pd.Timestamp("2020-01-01"), color=RED, ls="--", lw=1.5)
    ax1.set_title("Monthly revenue by therapeutic class (EUR millions)")
    ax1.set_ylabel("Revenue (M)"); ax1.legend(fontsize=7, ncol=3)

    # 2. Top 10 products by total revenue
    ax2 = fig.add_subplot(gs[0, 2])
    top10 = raw.groupby("Product Name")["Sales"].sum().sort_values().tail(10)
    ax2.barh(range(10), top10.values / 1e6,
             color=[GREEN if i >= 7 else MINT if i >= 4 else GOLD for i in range(10)],
             alpha=0.85, edgecolor="white")
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([n[:20] for n in top10.index], fontsize=7)
    ax2.set_title("Top 10 products\n(total revenue 2017-2020)")
    ax2.set_xlabel("Revenue (M)")

    # 3. Predicted vs actual scatter
    ax3 = fig.add_subplot(gs[1, 0])
    lim = max(y_test.max(), y_pred.max()) * 1.05
    ax3.scatter(y_test, y_pred, alpha=0.25, s=10, color=BLUE, edgecolors="none")
    ax3.plot([0, lim], [0, lim], color=RED, ls="--", lw=1.5)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ax3.text(0.05, 0.92, f"R2={r2:.4f}\nMAE={mae:.0f} units/mo", transform=ax3.transAxes,
              fontsize=9, color=GREEN, fontweight="bold", va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MINT, alpha=0.9))
    ax3.set_xlabel("Actual"); ax3.set_ylabel("Predicted")
    ax3.set_title("Gradient Boosting\nPredicted vs Actual (2020 test set)")
    ax3.set_xlim(0, lim); ax3.set_ylim(0, lim)

    # 4. MAE by therapeutic class
    ax4 = fig.add_subplot(gs[1, 1])
    rows = []
    for cls in test_df["Product Class"].unique():
        mask = test_df["Product Class"] == cls
        rows.append({"cls": cls, "mae": mean_absolute_error(y_test[mask], y_pred[mask])})
    class_df = pd.DataFrame(rows).sort_values("mae")
    bar_colors = [RED if v > 100 else MINT for v in class_df["mae"]]
    bars = ax4.barh(class_df["cls"], class_df["mae"], color=bar_colors, alpha=0.85, edgecolor="white")
    ax4.axvline(class_df["mae"].mean(), color=GOLD, ls="--", lw=1.5)
    for bar, v in zip(bars, class_df["mae"]):
        ax4.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.0f}", va="center", fontsize=7.5)
    ax4.set_title("MAE by therapeutic class\n(units/month)"); ax4.set_xlabel("MAE")

    # 5. Model comparison
    ax5 = fig.add_subplot(gs[1, 2])
    m = meta["metrics"]
    names = ["Random\nForest", "Gradient\nBoosting"]
    r2s = [m["random_forest"]["r2"], m["gradient_boosting"]["r2"]]
    maes = [m["random_forest"]["mae"], m["gradient_boosting"]["mae"]]
    x = np.arange(2); w = 0.35
    ax5b = ax5.twinx()
    ax5.bar(x - w/2, r2s, w, color=GREEN, alpha=0.75, edgecolor="white", label="R2")
    ax5b.bar(x + w/2, maes, w, color=GOLD, alpha=0.75, edgecolor="white", label="MAE")
    ax5.set_xticks(x); ax5.set_xticklabels(names)
    ax5.set_ylabel("R2", color=GREEN); ax5b.set_ylabel("MAE (units/mo)", color=GOLD)
    ax5.set_title("Model comparison"); ax5.set_ylim(0, 1.1); ax5b.set_ylim(0, 200)
    ax5.spines["top"].set_visible(False); ax5b.spines["top"].set_visible(False)
    h1, l1 = ax5.get_legend_handles_labels(); h2, l2 = ax5b.get_legend_handles_labels()
    ax5.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    # 6. Feature importance
    ax6 = fig.add_subplot(gs[2, 0])
    importance = pd.Series(meta["feature_importance"]).sort_values()
    ax6.barh(importance.index, importance.values,
             color=[GREEN if v > 0.05 else MINT for v in importance.values],
             alpha=0.85, edgecolor="white")
    ax6.set_title("Feature importance\n(Random Forest)"); ax6.set_xlabel("Importance")

    # 7. Revenue by country & channel
    ax7 = fig.add_subplot(gs[2, 1])
    by_country_channel = raw.groupby(["Country", "Channel"])["Sales"].sum().unstack().fillna(0) / 1e6
    by_country_channel.plot(kind="bar", ax=ax7, color=[GREEN, MINT], alpha=0.85, edgecolor="white", width=0.65)
    ax7.set_title("Revenue by country & channel\n(EUR millions)")
    ax7.set_ylabel("Revenue (M)"); ax7.set_xticklabels(by_country_channel.index, rotation=0)
    ax7.legend(fontsize=8)

    # 8. ABC classification pie chart
    ax8 = fig.add_subplot(gs[2, 2])
    stats = pd.read_csv(os.path.join(PROC_DIR, "product_stats.csv"))
    stats = stats.sort_values("avg_monthly_demand", ascending=False)
    cum = stats["avg_monthly_demand"].cumsum() / stats["avg_monthly_demand"].sum() * 100
    classes = ["A" if c <= 70 else "B" if c <= 90 else "C" for c in cum]
    counts = [sum(1 for c in classes if c == x) for x in ["A", "B", "C"]]
    ax8.pie(counts,
            labels=[f"A - High priority\n({counts[0]} products)",
                    f"B - Medium\n({counts[1]})",
                    f"C - Low\n({counts[2]})"],
            colors=[GREEN, MINT, GOLD], autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax8.set_title(f"ABC classification\n({sum(counts)} products)")

    out_path = os.path.join(DOCS_DIR, "evaluation_dashboard.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=140)
    print(f"Dashboard saved to {out_path}")


if __name__ == "__main__":
    main()
