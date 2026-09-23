"""
Post-hoc evaluation of the 3 navigation-rule points (knee / max-MCC / max-Phi)
from each station's seed-0 NSGA-II Pareto front:

  1. Sensor-noise stress test (grouped faithfulness, on the training
     station's own held-out slice)
  2. Spatial transfer: refit-once-per-source-station model applied to
     (a) the other 3 primary stations, (b) the 8 held-out Beijing stations
  3. Cross-city transfer: same model applied to Delhi (real missing WSPM,
     real sensor gaps -- no synthetic noise needed there, the domain gap
     IS the stress test)

All heavy (fit + SHAP) work happens exactly once per (station, nav-rule) =
4 x 3 = 12 refits; every downstream evaluation reuses that fitted pipeline
for pure inference + explanation, which is cheap.
"""
import sys, time, json, argparse
sys.path.insert(0, "/home/claude/xai_aq/src")
import numpy as np
import pandas as pd
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import matthews_corrcoef

from features import (build_dataset, load_beijing_daily, load_delhi_daily,
                       COMMON_FEATURES, FEATURE_GROUPS)
from xai_core import build_preprocessing_steps, suggest_model, faithfulness, noise_stress_test
from navigation import pareto_mask, build_navigation_table

RESULTS_DIR = "/home/claude/xai_aq/results"
PRIMARY_STATIONS = ["Dongsi", "Huairou", "Wanshouxigong", "Dingling"]
FEATURE_IDX_BY_GROUP = {g: [COMMON_FEATURES.index(c) for c in cols] for g, cols in FEATURE_GROUPS.items()}
# Only ACTUALLY-SENSED columns get injected noise/missingness -- pollutants
# + raw meteorology, not the engineered calendar features (month/dow are
# computed facts, never "sensed", so noise there has no physical meaning).
SENSOR_COLS = sorted(FEATURE_IDX_BY_GROUP["pollutants"] + FEATURE_IDX_BY_GROUP["pollutants_roll"]
                      + FEATURE_IDX_BY_GROUP["meteorology"])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refit_pipeline(params, X_tr, y_tr, n_classes, seed):
    fixed = optuna.trial.FixedTrial(params)
    pre_steps = build_preprocessing_steps(fixed, X_tr.shape[1])
    family, model = suggest_model(fixed, n_classes, random_state=seed)
    pipe = ImbPipeline(pre_steps + [("clf", model)])
    pipe.fit(X_tr, y_tr)
    return family, pipe


def pre_transform(pipe, Xraw):
    Xp = Xraw
    for name, step in pipe.steps[:-1]:
        if name != "smote":
            Xp = step.transform(Xp)
    return Xp


def eval_on(pipe, family, X_bg_pre, X_raw, y):
    """MCC + grouped Phi on a fresh (raw) dataset, through the fitted pipeline."""
    X_pre = pre_transform(pipe, X_raw)
    pred = pipe.named_steps["clf"].predict(X_pre)
    mcc = matthews_corrcoef(y, pred) if len(np.unique(y)) > 1 else np.nan
    n_f = min(60, len(X_pre))
    idx = np.random.RandomState(0).choice(len(X_pre), n_f, replace=False)
    try:
        phi, _, _ = faithfulness(pipe.named_steps["clf"], family, X_bg_pre, X_pre[idx],
                                  k_steps=6, feature_groups=FEATURE_IDX_BY_GROUP)
    except Exception:
        phi = np.nan
    return mcc, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default=",".join(PRIMARY_STATIONS))
    args = ap.parse_args()
    stations = args.stations.split(",")

    bj = load_beijing_daily()
    all_bj_stations = sorted(bj.station.unique())
    held_out_stations = [s for s in all_bj_stations if s not in PRIMARY_STATIONS]
    delhi = load_delhi_daily()
    X_delhi_df, y_delhi, meta_delhi = build_dataset(delhi)
    log(f"Held-out Beijing stations ({len(held_out_stations)}): {held_out_stations}")
    log(f"Delhi rows: {len(X_delhi_df)}")

    per_station_data = {}
    for st in all_bj_stations:
        sdf = bj[bj.station == st].reset_index(drop=True)
        Xd, y, meta = build_dataset(sdf)
        per_station_data[st] = (Xd.values, y)

    all_results = []
    navigation_rows = []

    for station in stations:
        front_path = f"{RESULTS_DIR}/{station}_nsga2_seed0.csv"
        try:
            front_all = pd.read_csv(front_path)
        except FileNotFoundError:
            log(f"SKIP {station}: {front_path} not found yet")
            continue
        mask = pareto_mask(front_all[["mcc", "phi"]].values)
        front = front_all[mask].reset_index(drop=True)
        log(f"{station}: {len(front_all)} trials, {len(front)} on Pareto front")
        nav = build_navigation_table(front)
        for row in nav:
            row["station"] = station
        navigation_rows.extend(nav)

        X, y = per_station_data[station]
        n_classes = len(np.unique(y))
        n = len(X)
        split = int(n * 0.75)  # chronological: first 75% train, last 25% held-out test
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]

        for rule_row in nav:
            trial_row = front[front["trial"] == rule_row["trial"]].iloc[0]
            params = {k.replace("param_", ""): v for k, v in trial_row.items()
                      if k.startswith("param_") and pd.notna(v)}
            # cast int-typed params back from float (CSV round-trip artifact)
            for k in list(params.keys()):
                if isinstance(params[k], float) and params[k] == int(params[k]) and \
                   k not in ("lr_C", "xgb_lr", "lgb_lr", "mlp_alpha", "kbest_frac"):
                    params[k] = int(params[k])

            log(f"  refit rule={rule_row['rule']} family={rule_row['model_family']}")
            try:
                family, pipe = refit_pipeline(params, X_tr, y_tr, n_classes, seed=0)
            except Exception as e:
                log(f"    REFIT FAILED: {e}")
                continue

            X_tr_pre = pre_transform(pipe, X_tr)

            # 1. sensor-noise stress test on own held-out slice
            try:
                stress = noise_stress_test(pipe, family, X_tr_pre, X_te, FEATURE_IDX_BY_GROUP,
                                            n_reps=5, seed=0, k_steps=6, sensor_cols=SENSOR_COLS)
            except Exception as e:
                log(f"    STRESS TEST FAILED: {e}")
                stress = {}

            own_mcc, own_phi = eval_on(pipe, family, X_tr_pre, X_te, y_te)

            base_row = {
                "source_station": station, "rule": rule_row["rule"],
                "model_family": family, "search_mcc": rule_row["mcc"], "search_phi": rule_row["phi"],
                "own_test_mcc": own_mcc, "own_test_phi": own_phi, **stress,
            }
            all_results.append({**base_row, "eval_target": "own_holdout", "target_type": "own"})

            # 2a. spatial transfer to other primary stations
            for other in PRIMARY_STATIONS:
                if other == station:
                    continue
                Xo, yo = per_station_data[other]
                mcc_o, phi_o = eval_on(pipe, family, X_tr_pre, Xo, yo)
                all_results.append({**base_row, "eval_target": other, "target_type": "primary_transfer",
                                     "transfer_mcc": mcc_o, "transfer_phi": phi_o})

            # 2b. spatial transfer to held-out (unseen) stations
            for other in held_out_stations:
                Xo, yo = per_station_data[other]
                mcc_o, phi_o = eval_on(pipe, family, X_tr_pre, Xo, yo)
                all_results.append({**base_row, "eval_target": other, "target_type": "heldout_transfer",
                                     "transfer_mcc": mcc_o, "transfer_phi": phi_o})

            # 3. cross-city transfer to Delhi
            mcc_d, phi_d = eval_on(pipe, family, X_tr_pre, X_delhi_df.values, y_delhi)
            all_results.append({**base_row, "eval_target": "delhi_ncr", "target_type": "crosscity_transfer",
                                 "transfer_mcc": mcc_d, "transfer_phi": phi_d})

            log(f"    own_mcc={own_mcc:.3f} own_phi={own_phi:.3f} "
                f"phi_drop={stress.get('phi_drop', float('nan')):.3f} "
                f"rank_stab={stress.get('rank_stability_mean', float('nan')):.3f}")

    pd.DataFrame(all_results).to_csv(f"{RESULTS_DIR}/transfer_eval_results.csv", index=False)
    pd.DataFrame(navigation_rows).to_csv(f"{RESULTS_DIR}/navigation_points.csv", index=False)
    log(f"Saved {len(all_results)} rows -> transfer_eval_results.csv")


if __name__ == "__main__":
    main()
