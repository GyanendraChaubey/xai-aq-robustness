"""Fast end-to-end smoke test: one model, one fold, on a small Beijing subset.
Verifies the whole chain (blocked CV -> pipeline -> faithfulness -> grouped
faithfulness -> noise stress test) runs without error before committing to
the full NSGA-II run."""
import sys, time
sys.path.insert(0, "/home/claude/xai_aq/src")
import numpy as np
from features import build_dataset, load_beijing_daily, COMMON_FEATURES, FEATURE_GROUPS
from xai_core import (station_blocked_cv_splits, build_preprocessing_steps, suggest_model,
                       faithfulness, noise_stress_test)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import matthews_corrcoef
import optuna

t0 = time.time()
bj = load_beijing_daily()
primary_stations = ["Dongsi", "Huairou", "Wanshouxigong", "Dingling"]
bj_primary = bj[bj.station.isin(primary_stations)].reset_index(drop=True)
X_df, y, meta = build_dataset(bj_primary)
print(f"Primary pool: {X_df.shape}, stations={sorted(meta.station.unique())}, t={time.time()-t0:.1f}s")

feature_idx_by_group = {g: [COMMON_FEATURES.index(c) for c in cols] for g, cols in FEATURE_GROUPS.items()}
print("feature groups (by index):", feature_idx_by_group)

folds = station_blocked_cv_splits(meta, n_splits=3)
print(f"n folds: {len(folds)}, fold sizes: {[(len(tr), len(te)) for tr, te in folds]}")

tr_idx, te_idx = folds[-1]  # use last (largest-train) fold for the smoke test
X = X_df.values
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]
n_classes = len(np.unique(y))
print(f"train={X_tr.shape} test={X_te.shape} n_classes={n_classes}")

study = optuna.create_study(directions=["maximize", "maximize"])
trial = study.ask()
pre_steps = build_preprocessing_steps(trial, X.shape[1])
family, model = suggest_model(trial, n_classes, random_state=0)
print("sampled family:", family, "pre_steps:", [s[0] for s in pre_steps])

pipe = ImbPipeline(pre_steps + [("clf", model)])
t1 = time.time()
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
mcc = matthews_corrcoef(y_te, pred)
print(f"fit+predict OK in {time.time()-t1:.1f}s, MCC={mcc:.3f}")

fitted_model = pipe.named_steps["clf"]
def pre_transform(Xraw):
    Xp = Xraw
    for name, step in pipe.steps[:-1]:
        if name != "smote":
            Xp = step.transform(Xp)
    return Xp

X_tr_pre = pre_transform(X_tr)
X_te_pre = pre_transform(X_te)
n_faith = min(40, len(X_te_pre))
idx = np.random.RandomState(0).choice(len(X_te_pre), n_faith, replace=False)

t2 = time.time()
phi, sv, order = faithfulness(fitted_model, family, X_tr_pre, X_te_pre[idx], k_steps=6)
print(f"faithfulness (ungrouped) OK in {time.time()-t2:.1f}s, phi={phi:.4f}")

t3 = time.time()
phi_g, sv_g, order_g = faithfulness(fitted_model, family, X_tr_pre, X_te_pre[idx], k_steps=6,
                                     feature_groups=feature_idx_by_group)
print(f"faithfulness (grouped) OK in {time.time()-t3:.1f}s, phi_grouped={phi_g:.4f}")

t4 = time.time()
stress = noise_stress_test(pipe, family, X_tr_pre, X_te[idx], feature_idx_by_group,
                            n_reps=3, seed=0, k_steps=6)
print(f"noise_stress_test OK in {time.time()-t4:.1f}s: {stress}")

print(f"\nTOTAL smoke test time: {time.time()-t0:.1f}s")
