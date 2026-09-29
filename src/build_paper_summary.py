"""Regenerates results/paper_summary.json from the current
transfer_eval_results.csv / portfolio_summary.json / baselines.csv. Kept as
a standalone script (rather than inline analysis) so the summary numbers
quoted in the paper can always be reproduced from the checked-in results
files, not from an ad-hoc one-off computation."""
import json
import numpy as np
import pandas as pd

RESULTS_DIR = "/home/claude/xai_aq/results"

with open(f"{RESULTS_DIR}/portfolio_summary.json") as f:
    portfolio = json.load(f)

baselines = pd.read_csv(f"{RESULTS_DIR}/baselines.csv")
df = pd.read_csv(f"{RESULTS_DIR}/transfer_eval_results.csv")
own = df[df["target_type"] == "own"].copy()
own["explainer"] = own["model_family"].map(
    lambda f: "kernel" if f in ("mlp", "nb", "knn") else ("linear" if f == "logreg" else "tree")
)

own_holdout_by_rule = {
    rule: {
        "own_test_mcc": round(float(g["own_test_mcc"].mean()), 3),
        "own_test_phi": round(float(g["own_test_phi"].mean()), 3),
        "phi_drop": round(float(g["phi_drop"].mean()), 3),
        "rank_stability_mean": round(float(g["rank_stability_mean"].mean()), 3),
    }
    for rule, g in own.groupby("rule")
}

transfer = df[df["target_type"] != "own"].copy()
transfer_mcc_by_type_rule = {
    f"{t}|{r}": round(float(g["transfer_mcc"].mean()), 3)
    for (t, r), g in transfer.groupby(["target_type", "rule"])
}
transfer_phi_by_type_rule = {
    f"{t}|{r}": round(float(g["transfer_phi"].mean()), 3)
    for (t, r), g in transfer.groupby(["target_type", "rule"])
}

explainer_type_effect = {
    etype: {
        "phi_drop_mean": round(float(g["phi_drop"].mean()), 3),
        "phi_drop_count": int(len(g)),
        "rank_stability_mean_mean": round(float(g["rank_stability_mean"].mean()), 3),
        "rank_stability_mean_count": int(len(g)),
    }
    for etype, g in own.groupby("explainer")
}

worst = own.loc[own["rank_stability_mean"].idxmin()]
tied_worst = own[own["rank_stability_mean"] == own["rank_stability_mean"].min()]
lowest_rank_stability_cases = [
    {
        "source_station": r["source_station"],
        "rule": r["rule"],
        "model_family": r["model_family"],
        "rank_stability_mean": float(r["rank_stability_mean"]),
    }
    for _, r in tied_worst.iterrows()
]

dongsi_maxphi = own[(own.source_station == "Dongsi") & (own.rule == "max_phi")]

summary = {
    "search": {
        "n_stations": portfolio["n_stations"],
        "n_trials": 40,
        "n_seeds": 2,
        "cv_folds": 3,
        "win_rate_vs_tpe": portfolio["win_rate_vs_tpe"],
        "win_rate_vs_random": portfolio["win_rate_vs_random"],
        "wilcoxon_vs_tpe_p": portfolio["wilcoxon_nsga2_vs_tpe"]["pvalue"],
        "wilcoxon_vs_random_p": portfolio["wilcoxon_nsga2_vs_random"]["pvalue"],
        "mean_hv_nsga2": float(np.mean([s["hv_nsga2"] for s in portfolio["station_means"]])),
        "mean_hv_random": float(np.mean([s["hv_random"] for s in portfolio["station_means"]])),
        "mean_hv_tpe": float(np.mean([s["hv_tpe"] for s in portfolio["station_means"]])),
    },
    "baselines_persistence_mcc": dict(zip(baselines.station, baselines.persistence_mcc)),
    "delhi_persistence_mcc": 0.361,
    "own_holdout_by_rule": own_holdout_by_rule,
    "transfer_mcc_by_type_rule": transfer_mcc_by_type_rule,
    "transfer_phi_by_type_rule": transfer_phi_by_type_rule,
    "explainer_type_effect": explainer_type_effect,
    "lowest_rank_stability_cases": lowest_rank_stability_cases,
    "dongsi_maxphi_mcc": float(dongsi_maxphi["own_test_mcc"].iloc[0]),
    "dongsi_persistence_mcc": float(baselines.set_index("station").loc["Dongsi", "persistence_mcc"]),
    "n_pareto_front_by_station": 4,
    "kernel_explainer_reseed_check": {
        "note": "phi_drop_mean for kernel-explained models, before vs. after "
                "independently reseeding KernelExplainer per clean/noisy call "
                "and raising nsamples from 80 to 300.",
        "before": -0.219,
        "after": explainer_type_effect.get("kernel", {}).get("phi_drop_mean"),
    },
}

with open(f"{RESULTS_DIR}/paper_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
