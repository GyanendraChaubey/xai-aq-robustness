"""
Aggregates all per-station search results:
  - 2D hypervolume per (station, method, seed)
  - TPE's post-hoc Phi (TPE never optimizes Phi during search, so we take
    its best-MCC trial per seed, refit under the same blocked-CV folds, and
    measure Phi the same way NSGA-II/random trials were scored -- giving a
    single comparable (mcc, phi) "TPE point" per station/seed, exactly the
    role MOXEC's get_tpe_point() plays for its 3-objective baselines)
  - Wilcoxon signed-rank tests (station-level, n=4): NSGA-II vs TPE,
    NSGA-II vs random search
  - Per-family win counts, front sizes
Writes results/portfolio_summary.json and results/hypervolume_by_station.csv
"""
import sys, json, time
sys.path.insert(0, "/home/claude/xai_aq/src")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import matthews_corrcoef

from features import build_dataset, load_beijing_daily
from xai_core import blocked_cv_splits, build_preprocessing_steps, suggest_model, faithfulness
from navigation import hypervolume_2d, pareto_mask

RESULTS_DIR = "/home/claude/xai_aq/results"
PRIMARY_STATIONS = ["Dongsi", "Huairou", "Wanshouxigong", "Dingling"]
N_SEEDS = 2
CV_FOLDS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tpe_post_hoc_phi(station, seed, X, y, faith_n=40, faith_steps=6):
    """Refit TPE's best-MCC trial's pipeline under the same blocked folds
    used during search, measuring Phi the same way NSGA-II/random trials
    were (ungrouped, averaged across folds) -- for a fair (mcc, phi)
    comparison point."""
    tpe_df = pd.read_csv(f"{RESULTS_DIR}/{station}_tpe_seed{seed}.csv")
    best = tpe_df.loc[tpe_df["mcc"].idxmax()]
    params = {k.replace("param_", ""): v for k, v in best.items()
              if k.startswith("param_") and pd.notna(v)}
    for k in list(params.keys()):
        if isinstance(params[k], float) and params[k] == int(params[k]) and \
           k not in ("lr_C", "xgb_lr", "lgb_lr", "mlp_alpha"):
            params[k] = int(params[k])

    n_classes = len(np.unique(y))
    folds = blocked_cv_splits(len(X), n_splits=CV_FOLDS)
    mccs, phis = [], []
    for tr_idx, te_idx in folds:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        fixed = optuna.trial.FixedTrial(params)
        pre_steps = build_preprocessing_steps(fixed, X.shape[1])
        family, model = suggest_model(fixed, n_classes, random_state=seed)
        pipe = ImbPipeline(pre_steps + [("clf", model)])
        try:
            pipe.fit(X_tr, y_tr)
        except Exception:
            continue
        pred = pipe.predict(X_te)
        mccs.append(matthews_corrcoef(y_te, pred))
        fitted = pipe.named_steps["clf"]
        def pre_t(Xr):
            Xp = Xr
            for name, step in pipe.steps[:-1]:
                if name != "smote":
                    Xp = step.transform(Xp)
            return Xp
        X_tr_pre, X_te_pre = pre_t(X_tr), pre_t(X_te)
        n_f = min(faith_n, len(X_te_pre))
        idx = np.random.RandomState(seed).choice(len(X_te_pre), n_f, replace=False)
        try:
            phi, _, _ = faithfulness(fitted, family, X_tr_pre, X_te_pre[idx], k_steps=faith_steps)
        except Exception:
            phi = 0.0
        phis.append(phi)
    return float(np.mean(mccs)), float(np.mean(phis)), best["model_family"]


def main():
    bj = load_beijing_daily()
    rows = []
    for station in PRIMARY_STATIONS:
        sdf = bj[bj.station == station].reset_index(drop=True)
        Xd, y, meta = build_dataset(sdf)
        X = Xd.values
        for seed in range(N_SEEDS):
            nsga2 = pd.read_csv(f"{RESULTS_DIR}/{station}_nsga2_seed{seed}.csv")
            rand = pd.read_csv(f"{RESULTS_DIR}/{station}_random_seed{seed}.csv")

            hv_nsga2 = hypervolume_2d(nsga2[["mcc", "phi"]].values)
            hv_rand = hypervolume_2d(rand[["mcc", "phi"]].values)

            log(f"{station} seed{seed}: computing TPE post-hoc phi...")
            tpe_mcc, tpe_phi, tpe_family = tpe_post_hoc_phi(station, seed, X, y)
            hv_tpe = tpe_mcc * tpe_phi  # single point vs (0,0) ref

            front_mask = pareto_mask(nsga2[["mcc", "phi"]].values)
            rows.append({
                "station": station, "seed": seed,
                "hv_nsga2": hv_nsga2, "hv_random": hv_rand, "hv_tpe": hv_tpe,
                "n_front_nsga2": int(front_mask.sum()),
                "best_mcc_nsga2": nsga2["mcc"].max(), "best_phi_nsga2": nsga2["phi"].max(),
                "tpe_mcc": tpe_mcc, "tpe_phi": tpe_phi, "tpe_family": tpe_family,
            })
            log(f"  hv_nsga2={hv_nsga2:.4f} hv_random={hv_rand:.4f} hv_tpe={hv_tpe:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/hypervolume_by_station.csv", index=False)

    # station-level (average across seeds) for the paired Wilcoxon test, n=4
    station_means = df.groupby("station")[["hv_nsga2", "hv_random", "hv_tpe"]].mean()
    log("\nStation-level mean hypervolume:\n" + station_means.to_string())

    w_vs_tpe = wilcoxon(station_means["hv_nsga2"], station_means["hv_tpe"])
    w_vs_random = wilcoxon(station_means["hv_nsga2"], station_means["hv_random"])
    win_vs_tpe = int((station_means["hv_nsga2"] > station_means["hv_tpe"]).sum())
    win_vs_random = int((station_means["hv_nsga2"] > station_means["hv_random"]).sum())

    summary = {
        "n_stations": len(station_means),
        "wilcoxon_nsga2_vs_tpe": {"statistic": float(w_vs_tpe.statistic), "pvalue": float(w_vs_tpe.pvalue)},
        "wilcoxon_nsga2_vs_random": {"statistic": float(w_vs_random.statistic), "pvalue": float(w_vs_random.pvalue)},
        "win_rate_vs_tpe": f"{win_vs_tpe}/{len(station_means)}",
        "win_rate_vs_random": f"{win_vs_random}/{len(station_means)}",
        "station_means": station_means.reset_index().to_dict(orient="records"),
    }
    with open(f"{RESULTS_DIR}/portfolio_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
