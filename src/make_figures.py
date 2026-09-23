"""Figure generation. Palette: dataviz skill's validated categorical set
(8 hues, PASS on all hard gates, light mode) + marker-shape redundancy for
model family (9th family gets a distinct marker, not a 9th generated hue) so
identity survives grayscale/print reproduction, not just color."""
import sys, json
sys.path.insert(0, "/home/claude/xai_aq/src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from navigation import pareto_mask

FIGS_DIR = "/home/claude/xai_aq/figs"
RESULTS_DIR = "/home/claude/xai_aq/results"

CAT_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#4a3aa7", "#e34948"]
MODEL_FAMILIES = ["logreg", "dtree", "rforest", "extratrees", "xgboost",
                   "lightgbm", "knn", "nb", "mlp"]
FAMILY_COLOR = {f: CAT_PALETTE[i % len(CAT_PALETTE)] for i, f in enumerate(MODEL_FAMILIES)}
FAMILY_MARKER = {f: m for f, m in zip(MODEL_FAMILIES,
                  ["o", "s", "^", "D", "v", "P", "X", "*", "h"])}

MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SEC_INK = "#52514e"


def _style_ax(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SEC_INK, labelsize=9)


def plot_pareto_front(station, seed=0):
    df = pd.read_csv(f"{RESULTS_DIR}/{station}_nsga2_seed{seed}.csv")
    mask = pareto_mask(df[["mcc", "phi"]].values)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=200)
    for fam in df["model_family"].unique():
        sub = df[df["model_family"] == fam]
        ax.scatter(sub["mcc"], sub["phi"], s=26, alpha=0.55,
                   color=FAMILY_COLOR.get(fam, MUTED), marker=FAMILY_MARKER.get(fam, "o"),
                   label=fam, linewidths=0, zorder=3)
    front = df[mask].sort_values("mcc")
    ax.plot(front["mcc"], front["phi"], color=INK, linewidth=1.2, zorder=4, alpha=0.6)
    ax.scatter(front["mcc"], front["phi"], s=90, facecolors="none",
               edgecolors=INK, linewidths=1.4, zorder=5, label="Pareto front")

    ax.set_xlabel("MCC (predictive performance)", color=INK, fontsize=10)
    ax.set_ylabel("Φ (explanation faithfulness)", color=INK, fontsize=10)
    ax.set_title(f"{station} — NSGA-II search (seed {seed})", color=INK, fontsize=11, loc="left")
    _style_ax(ax)
    ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIGS_DIR}/{station}_pareto_front.pdf")
    fig.savefig(f"{FIGS_DIR}/{station}_pareto_front.png")
    plt.close(fig)


def plot_hypervolume_comparison():
    df = pd.read_csv(f"{RESULTS_DIR}/hypervolume_by_station.csv")
    means = df.groupby("station")[["hv_nsga2", "hv_random", "hv_tpe"]].mean()
    stations = means.index.tolist()
    x = np.arange(len(stations))
    w = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
    methods = [("hv_nsga2", "NSGA-II (2-obj)", CAT_PALETTE[0]),
               ("hv_random", "Random search (2-obj)", CAT_PALETTE[2]),
               ("hv_tpe", "TPE (MCC-only)", CAT_PALETTE[1])]
    for i, (col, label, color) in enumerate(methods):
        ax.bar(x + (i - 1) * w, means[col], width=w, label=label, color=color, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(stations, fontsize=9, color=SEC_INK)
    ax.set_ylabel("2D hypervolume (mean across seeds)", color=INK, fontsize=10)
    ax.set_title("Search-method comparison by station", color=INK, fontsize=11, loc="left")
    _style_ax(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGS_DIR}/hypervolume_comparison.pdf")
    fig.savefig(f"{FIGS_DIR}/hypervolume_comparison.png")
    plt.close(fig)


def plot_noise_stress():
    df = pd.read_csv(f"{RESULTS_DIR}/transfer_eval_results.csv")
    own = df[df["target_type"] == "own"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), dpi=200)

    ax = axes[0]
    rules = own["rule"].unique()
    for i, rule in enumerate(rules):
        sub = own[own["rule"] == rule]
        ax.scatter(sub["phi_clean"], sub["phi_noisy_mean"], s=70,
                   color=CAT_PALETTE[i % len(CAT_PALETTE)], label=rule, zorder=3)
    lims = [0, max(own["phi_clean"].max(), own["phi_noisy_mean"].max()) * 1.1]
    ax.plot(lims, lims, "--", color=MUTED, linewidth=1, zorder=2, label="y = x (no degradation)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Φ, clean", fontsize=10, color=INK)
    ax.set_ylabel("Φ, under sensor noise", fontsize=10, color=INK)
    ax.set_title("Faithfulness under sensor noise", fontsize=10, loc="left", color=INK)
    _style_ax(ax)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for i, rule in enumerate(rules):
        sub = own[own["rule"] == rule]
        ax.scatter(sub["source_station"], sub["rank_stability_mean"], s=70,
                   color=CAT_PALETTE[i % len(CAT_PALETTE)], label=rule, zorder=3)
    ax.axhline(0.8, linestyle="--", color=MUTED, linewidth=1, zorder=2)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("Rank stability (Spearman ρ, clean vs. noisy)", fontsize=9, color=INK)
    ax.set_title("Attribution rank stability under noise", fontsize=10, loc="left", color=INK)
    _style_ax(ax)
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(f"{FIGS_DIR}/noise_stress.pdf")
    fig.savefig(f"{FIGS_DIR}/noise_stress.png")
    plt.close(fig)


def plot_transfer_heatmap():
    df = pd.read_csv(f"{RESULTS_DIR}/transfer_eval_results.csv")
    sub = df[df["target_type"].isin(["own", "primary_transfer", "heldout_transfer", "crosscity_transfer"])].copy()
    sub["mcc_val"] = sub["own_test_mcc"].where(sub["target_type"] == "own", sub["transfer_mcc"])
    piv = sub[sub["rule"] == "knee_point"].pivot_table(
        index="source_station", columns="eval_target", values="mcc_val", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=200)
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=max(0.05, np.nanmax(piv.values)), aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title("Knee-point model: MCC, source station → eval target", fontsize=10, loc="left", color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIGS_DIR}/transfer_heatmap.pdf")
    fig.savefig(f"{FIGS_DIR}/transfer_heatmap.png")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="Dongsi,Huairou,Wanshouxigong,Dingling")
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    stations = args.stations.split(",")

    if args.which in ("all", "pareto"):
        for st in stations:
            try:
                plot_pareto_front(st)
                print("OK pareto:", st)
            except Exception as e:
                print("SKIP pareto", st, e)
    if args.which in ("all", "hv"):
        try:
            plot_hypervolume_comparison(); print("OK hv comparison")
        except Exception as e:
            print("SKIP hv comparison", e)
    if args.which in ("all", "noise"):
        try:
            plot_noise_stress(); print("OK noise stress")
        except Exception as e:
            print("SKIP noise stress", e)
    if args.which in ("all", "transfer"):
        try:
            plot_transfer_heatmap(); print("OK transfer heatmap")
        except Exception as e:
            print("SKIP transfer heatmap", e)
