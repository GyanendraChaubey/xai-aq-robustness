"""
Main search driver. Per-station design (mirrors MOXEC's per-dataset design,
which is what its Wilcoxon/Friedman machinery expects): for each of the 4
primary stations, independently run

  - NSGA-II, 2-objective (MCC x Phi), N_SEEDS seeds x N_TRIALS trials
  - TPE, single-objective (MCC only) baseline -- same budget
  - Random search, 2-objective -- same budget

all under BLOCKED (non-shuffled) temporal CV. Search-phase Phi uses the
UNGROUPED faithfulness metric (comparable across all three search methods,
matches TPE's need for a scalar-compatible pipeline); the grouped/
noise-stress/transfer evaluations are applied POST-HOC to just the 3
navigation-rule points per front (cheap: 3 x 4 stations = 12 refits, not
40 x 2 x 4 = 320) in run_transfer_eval.py.

Resumable: every (station, method, seed) is its own Optuna study in a
shared sqlite DB; re-running this script skips studies that already have
enough completed trials.
"""
import sys, time, json, argparse, traceback
sys.path.insert(0, "/home/claude/xai_aq/src")
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler, TPESampler, RandomSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import matthews_corrcoef

from features import build_dataset, load_beijing_daily, COMMON_FEATURES
from xai_core import blocked_cv_splits, build_preprocessing_steps, suggest_model, faithfulness

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = "/home/claude/xai_aq/results"
LOG_PATH = "/home/claude/xai_aq/logs/run_search.log"
STUDY_DB = f"sqlite:////home/claude/xai_aq/results/studies.db"
PRIMARY_STATIONS = ["Dongsi", "Huairou", "Wanshouxigong", "Dingling"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def make_objective_2obj(X, y, cv_folds, seed, faith_n=40, faith_steps=6, single_obj=False):
    n_classes = len(np.unique(y))
    folds = blocked_cv_splits(len(X), n_splits=cv_folds)

    def objective(trial):
        pre_steps = build_preprocessing_steps(trial, X.shape[1])
        family, model = suggest_model(trial, n_classes, random_state=seed)
        mccs, phis = [], []
        for tr_idx, te_idx in folds:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if len(np.unique(y_tr)) < 2:
                raise optuna.TrialPruned()
            pipe = ImbPipeline(pre_steps + [("clf", model)])
            try:
                pipe.fit(X_tr, y_tr)
            except Exception:
                raise optuna.TrialPruned()
            pred = pipe.predict(X_te)
            mccs.append(matthews_corrcoef(y_te, pred))

            if not single_obj:
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

        trial.set_user_attr("model_family", family)
        if single_obj:
            return float(np.mean(mccs))
        return float(np.mean(mccs)), float(np.mean(phis))

    return objective


def run_study(name, sampler, directions, X, y, cv_folds, seed, n_trials, single_obj=False):
    study = optuna.create_study(study_name=name, directions=directions, sampler=sampler,
                                 storage=STUDY_DB, load_if_exists=True)
    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        log(f"  {name}: running {remaining} new trials ({len(study.trials)} done)")
        study.optimize(make_objective_2obj(X, y, cv_folds, seed, single_obj=single_obj),
                        n_trials=remaining, show_progress_bar=False,
                        catch=(Exception,))
    else:
        log(f"  {name}: already complete ({len(study.trials)} trials)")
    return study


def trials_to_df(study, single_obj=False):
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        row = {"trial": t.number, "model_family": t.user_attrs.get("model_family", "unknown"),
               **{f"param_{k}": v for k, v in t.params.items()}}
        if single_obj:
            row["mcc"] = t.values[0]
        else:
            row["mcc"], row["phi"] = t.values
        rows.append(row)
    return pd.DataFrame(rows)


def run_station(station, n_trials, n_seeds, cv_folds):
    log(f"=== Station: {station} ===")
    bj = load_beijing_daily()
    station_df = bj[bj.station == station].reset_index(drop=True)
    X_df, y, meta = build_dataset(station_df)
    X = X_df.values
    log(f"  n={len(X)}, class dist={pd.Series(y).value_counts().sort_index().to_dict()}")

    for seed in range(n_seeds):
        # NSGA-II, 2-objective
        s = run_study(f"{station}_nsga2_seed{seed}", NSGAIISampler(seed=seed),
                       ["maximize", "maximize"], X, y, cv_folds, seed, n_trials)
        trials_to_df(s).to_csv(f"{RESULTS_DIR}/{station}_nsga2_seed{seed}.csv", index=False)

        # Random search, 2-objective
        s = run_study(f"{station}_random_seed{seed}", RandomSampler(seed=seed),
                       ["maximize", "maximize"], X, y, cv_folds, seed, n_trials)
        trials_to_df(s).to_csv(f"{RESULTS_DIR}/{station}_random_seed{seed}.csv", index=False)

        # TPE, single-objective (MCC only)
        s = run_study(f"{station}_tpe_seed{seed}", TPESampler(seed=seed),
                       ["maximize"], X, y, cv_folds, seed, n_trials, single_obj=True)
        trials_to_df(s, single_obj=True).to_csv(f"{RESULTS_DIR}/{station}_tpe_seed{seed}.csv", index=False)

    log(f"=== Station {station} complete ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default=",".join(PRIMARY_STATIONS))
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--cv-folds", type=int, default=3)
    args = ap.parse_args()

    stations = args.stations.split(",")
    log(f"Starting run: stations={stations} n_trials={args.n_trials} n_seeds={args.n_seeds} cv_folds={args.cv_folds}")
    t0 = time.time()
    for station in stations:
        try:
            run_station(station, args.n_trials, args.n_seeds, args.cv_folds)
        except Exception as e:
            log(f"STATION {station} FAILED: {e}\n{traceback.format_exc()}")
    log(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
