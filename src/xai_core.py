"""
Core search + explanation-evaluation library. Adapted from MOXEC
(~/Documents/MOXEC/full_pipeline/moxec_full_pipeline.py) with three changes
load-bearing for this paper's claim, plus one budget cut the user asked for:

  1. COST/LATENCY OBJECTIVE REMOVED. Search is 2-objective (MCC x Phi), not
     3-objective. NSGA-II directions=["maximize","maximize"].
  2. StratifiedKFold -> BLOCKED TEMPORAL CV. Air-quality time series are
     autocorrelated; a shuffled K-fold leaks future into past within a
     station. blocked_cv_splits() yields contiguous, chronologically-ordered
     folds instead.
  3. Faithfulness (Phi) gets a GROUPED variant for correlated meteorological
     drivers (temp/pressure/RH/wind move together) -- single-feature
     insertion/deletion misattributes credit among them.
  4. NEW: noise_stress_test() -- perturbs the explain set with sensor-like
     noise/missingness and re-measures Phi + rank-stability of the
     attribution ordering. Not present in MOXEC at all.

Model families, preprocessing search space, SHAP dispatch logic, and the
faithfulness insertion/deletion mechanics are carried over near-verbatim
from MOXEC (that machinery is domain-agnostic and already validated across
38 datasets); this module is the fork point.
"""
import numpy as np
import pandas as pd
import shap
import optuna
from optuna.samplers import NSGAIISampler, TPESampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr

optuna.logging.set_verbosity(optuna.logging.WARNING)

MLP_ARCHS = {"32": (32,), "64": (64,), "64_32": (64, 32)}
MODEL_FAMILIES = ["logreg", "dtree", "rforest", "extratrees", "xgboost",
                   "lightgbm", "knn", "nb", "mlp"]


# ---------------------------------------------------------------- CV -----
def blocked_cv_splits(n_samples, n_splits=3, min_train_frac=0.4):
    """Contiguous, chronologically-ordered (blocked) CV. Fold i trains on
    everything before its block and tests on the block itself (walk-forward
    style), never shuffling. min_train_frac guarantees the first fold still
    has a usable training set."""
    splits = []
    start = int(n_samples * min_train_frac)
    block_bounds = np.linspace(start, n_samples, n_splits + 1, dtype=int)
    for i in range(n_splits):
        train_idx = np.arange(0, block_bounds[i])
        test_idx = np.arange(block_bounds[i], block_bounds[i + 1])
        if len(train_idx) > 10 and len(test_idx) > 5:
            splits.append((train_idx, test_idx))
    return splits


def station_blocked_cv_splits(meta, n_splits=3, min_train_frac=0.4):
    """Blocked CV applied independently WITHIN each station (so a fold never
    trains on station A's future to predict station A's past, and never mixes
    station boundaries into one contiguous block), then unions the resulting
    train/test index sets across stations for that fold number.
    `meta` is the per-row DataFrame with a 'station' column, index-aligned to X."""
    meta = meta.reset_index(drop=True)
    per_station_splits = {}
    for station, g in meta.groupby("station", sort=False):
        idx = g.index.to_numpy()
        local_splits = blocked_cv_splits(len(idx), n_splits, min_train_frac)
        per_station_splits[station] = [(idx[tr], idx[te]) for tr, te in local_splits]
    n_folds = min(len(v) for v in per_station_splits.values())
    folds = []
    for f in range(n_folds):
        tr = np.concatenate([per_station_splits[s][f][0] for s in per_station_splits])
        te = np.concatenate([per_station_splits[s][f][1] for s in per_station_splits])
        folds.append((np.sort(tr), np.sort(te)))
    return folds


# ------------------------------------------------------- preprocessing ---
def build_preprocessing_steps(trial, n_features):
    steps = [("impute", SimpleImputer(strategy="median"))]  # NEW vs MOXEC:
    # air-quality data has real sensor gaps (Delhi rain/WSPM); MOXEC's UCI/
    # KEEL tables had none, so it never needed this step.

    scaler_choice = trial.suggest_categorical("scaler", ["none", "standard", "minmax", "robust"])
    if scaler_choice == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_choice == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_choice == "robust":
        steps.append(("scaler", RobustScaler()))

    # NOTE vs MOXEC: no SelectKBest step. Grouped/correlated-driver
    # faithfulness (this paper's core evaluation) needs feature COLUMN
    # IDENTITY to stay fixed and known post-preprocessing (feature_groups
    # indexes into the transformed matrix). A variable-width feature
    # selector breaks that invariant, so the search space is deliberately
    # narrowed to scaler x imbalance-handling x model family/hyperparams;
    # feature identity is fixed at the domain-engineered 20-column schema.

    imb_choice = trial.suggest_categorical("imbalance", ["none", "smote"])
    if imb_choice == "smote":
        steps.append(("smote", SMOTE(random_state=0, k_neighbors=3)))

    return steps


def suggest_model(trial, n_classes, random_state=0):
    family = trial.suggest_categorical("model_family", MODEL_FAMILIES)

    if family == "logreg":
        C = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        # saga (not liblinear): liblinear's OvR-only scheme errors out on our
        # >2-class AQI target in current sklearn; saga supports l1/l2 with a
        # true multinomial loss for n_classes > 2.
        model = LogisticRegression(C=C, penalty=penalty, solver="saga",
                                    max_iter=3000, random_state=random_state)
    elif family == "dtree":
        depth = trial.suggest_int("dt_max_depth", 2, 20)
        min_split = trial.suggest_int("dt_min_samples_split", 2, 20)
        model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split,
                                        random_state=random_state)
    elif family == "rforest":
        n_est = trial.suggest_int("rf_n_estimators", 20, 200)
        depth = trial.suggest_int("rf_max_depth", 3, 18)
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                        n_jobs=-1, random_state=random_state)
    elif family == "extratrees":
        n_est = trial.suggest_int("et_n_estimators", 20, 200)
        depth = trial.suggest_int("et_max_depth", 3, 18)
        model = ExtraTreesClassifier(n_estimators=n_est, max_depth=depth,
                                      n_jobs=-1, random_state=random_state)
    elif family == "xgboost":
        n_est = trial.suggest_int("xgb_n_estimators", 20, 200)
        depth = trial.suggest_int("xgb_max_depth", 2, 10)
        lr = trial.suggest_float("xgb_lr", 1e-3, 0.5, log=True)
        model = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                   objective="multi:softprob", num_class=n_classes,
                                   eval_metric="mlogloss", n_jobs=-1,
                                   random_state=random_state)
    elif family == "lightgbm":
        n_est = trial.suggest_int("lgb_n_estimators", 20, 200)
        leaves = trial.suggest_int("lgb_num_leaves", 7, 100)
        lr = trial.suggest_float("lgb_lr", 1e-3, 0.5, log=True)
        model = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=leaves, learning_rate=lr,
                                    n_jobs=-1, random_state=random_state, verbosity=-1)
    elif family == "knn":
        k = trial.suggest_int("knn_k", 3, 31)
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    elif family == "nb":
        model = GaussianNB()
    elif family == "mlp":
        arch_key = trial.suggest_categorical("mlp_units", list(MLP_ARCHS.keys()))
        alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True)
        model = MLPClassifier(hidden_layer_sizes=MLP_ARCHS[arch_key], alpha=alpha,
                               max_iter=400, random_state=random_state)
    return family, model


# ----------------------------------------------------------- SHAP/Phi ----
TREE_FAMILIES = {"dtree", "rforest", "extratrees", "xgboost", "lightgbm"}


def get_shap_values(model, family, X_background, X_explain, n_classes):
    if family in TREE_FAMILIES:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain, check_additivity=False)
    elif family == "logreg":
        explainer = shap.LinearExplainer(model, X_background)
        sv = explainer.shap_values(X_explain)
    else:
        bg = shap.kmeans(X_background, min(25, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(X_explain, nsamples=80, silent=True)

    if isinstance(sv, list):
        return np.array(sv)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return np.transpose(sv, (2, 0, 1))
    if n_classes == 2:
        return np.stack([-sv, sv])
    # some explainers collapse a >2-class output to (n,d) for the top class only;
    # broadcast defensively rather than crash a whole trial.
    return np.stack([sv] * n_classes)


def faithfulness(model, family, X_background, X_explain, k_steps=8, feature_groups=None):
    """Phi = InsertionAUC - DeletionAUC, per-instance on the predicted class.
    If feature_groups (dict[name] -> list[col idx]) is given, insertion/
    deletion operate on GROUPS of columns at a time (all-in or all-out),
    with each group's |SHAP| summed to rank it -- this is the correlated-
    driver-aware variant used for the meteorological block."""
    n, d = X_explain.shape
    background_vec = np.median(X_background, axis=0)
    proba_full = model.predict_proba(X_explain)
    target_cls = proba_full.argmax(axis=1)
    n_classes_local = proba_full.shape[1]

    sv_all = get_shap_values(model, family, X_background, X_explain, n_classes_local)
    sv = np.stack([sv_all[target_cls[i], i, :] for i in range(n)])

    if feature_groups is None:
        units = [[j] for j in range(d)]
    else:
        units = list(feature_groups.values())
    n_units = len(units)
    unit_importance = np.stack([np.abs(sv[:, u]).sum(axis=1) for u in units], axis=1)  # (n, n_units)
    order = np.argsort(-unit_importance, axis=1)

    steps = np.unique(np.linspace(0, n_units, k_steps, dtype=int))
    x = steps / n_units

    del_curves = np.zeros((n, len(steps)))
    ins_curves = np.zeros((n, len(steps)))
    for si, k in enumerate(steps):
        Xdel = X_explain.copy()
        Xins = np.tile(background_vec, (n, 1))
        for i in range(n):
            unit_idx = order[i, :k]
            cols = np.concatenate([units[u] for u in unit_idx]) if k > 0 else np.array([], dtype=int)
            if len(cols):
                Xdel[i, cols] = background_vec[cols]
                Xins[i, cols] = X_explain[i, cols]
        del_curves[:, si] = model.predict_proba(Xdel)[np.arange(n), target_cls]
        ins_curves[:, si] = model.predict_proba(Xins)[np.arange(n), target_cls]

    del_auc = np.array([np.trapezoid(del_curves[i], x) for i in range(n)])
    ins_auc = np.array([np.trapezoid(ins_curves[i], x) for i in range(n)])
    phi = float((ins_auc - del_auc).mean())
    return phi, sv, order  # return sv/order too, reused by the stress test


# --------------------------------------------------- sensor-noise stress -
def inject_sensor_noise(X, rng, rel_noise_std=0.15, missing_frac=0.15, noisy_cols=None):
    """Simulates a low-cost-sensor-grade version of X: multiplicative
    Gaussian noise (rel_noise_std, matching the ~10-20% relative error
    commonly reported for low-cost PM/gas sensors vs. reference monitors)
    plus MCAR missingness (filled by the pipeline's own median imputer at
    inference, exactly as a real deployment would), applied ONLY to
    actually-sensed columns (pollutants + raw meteorology). noisy_cols MUST
    be passed explicitly by the caller for this reason -- the default
    (all columns) is provided only as a fallback and should not be used
    with the calendar-feature columns in this project's schema, since
    calendar features are computed facts, not sensor readings, and
    "sensor noise" on them has no physical meaning."""
    Xn = X.copy()
    cols = noisy_cols if noisy_cols is not None else np.arange(X.shape[1])
    noise = rng.normal(1.0, rel_noise_std, size=(X.shape[0], len(cols)))
    Xn[:, cols] = Xn[:, cols] * noise
    miss_mask = rng.random(size=(X.shape[0], len(cols))) < missing_frac
    for j_local, j in enumerate(cols):
        Xn[miss_mask[:, j_local], j] = np.nan
    return Xn


def noise_stress_test(pipe, family, X_background_pre, X_explain_raw, feature_idx_by_group,
                       n_reps=5, seed=0, k_steps=8, sensor_cols=None):
    """Applies inject_sensor_noise() to X_explain n_reps times (through the
    FULL fitted pipeline, so imputation/scaling react exactly as they would
    on a real noisy input), and reports:
      - phi_noisy_mean, phi_drop = phi_clean - phi_noisy_mean
      - rank_stability: mean Spearman rho between the CLEAN feature-group
        importance ranking and each noisy replicate's ranking (1.0 = the
        explanation points at the same drivers even under noise)
    """
    rng = np.random.RandomState(seed)
    fitted_model = pipe.named_steps["clf"]

    def transform_through_pre(Xraw):
        Xp = Xraw
        for name, step in pipe.steps[:-1]:
            if name != "smote":
                Xp = step.transform(Xp)
        return Xp

    X_explain_pre = transform_through_pre(X_explain_raw)
    phi_clean, sv_clean, _ = faithfulness(fitted_model, family, X_background_pre, X_explain_pre,
                                           k_steps=k_steps, feature_groups=feature_idx_by_group)
    units = list(feature_idx_by_group.values())
    clean_rank = -np.stack([np.abs(sv_clean[:, u]).sum(axis=1) for u in units], axis=1).mean(0)
    clean_order = np.argsort(clean_rank)

    phis, rhos = [], []
    for r in range(n_reps):
        Xn_raw = inject_sensor_noise(X_explain_raw, np.random.RandomState(seed * 100 + r),
                                      noisy_cols=sensor_cols)
        Xn_pre = transform_through_pre(Xn_raw)
        phi_n, sv_n, _ = faithfulness(fitted_model, family, X_background_pre, Xn_pre,
                                       k_steps=k_steps, feature_groups=feature_idx_by_group)
        noisy_rank = -np.stack([np.abs(sv_n[:, u]).sum(axis=1) for u in units], axis=1).mean(0)
        noisy_order = np.argsort(noisy_rank)
        rho, _ = spearmanr(clean_order, noisy_order)
        phis.append(phi_n)
        rhos.append(rho if not np.isnan(rho) else 0.0)

    return {
        "phi_clean": phi_clean,
        "phi_noisy_mean": float(np.mean(phis)),
        "phi_noisy_std": float(np.std(phis)),
        "phi_drop": phi_clean - float(np.mean(phis)),
        "rank_stability_mean": float(np.mean(rhos)),
        "rank_stability_std": float(np.std(rhos)),
    }
