"""2D hypervolume + front-navigation rules (knee-point / max-MCC / max-Phi).
Replaces MOXEC's 3D pymoo hypervolume with a closed-form 2D version now that
cost is dropped; replaces its 3-rule (knee/cost-constrained/weighted-MES)
navigation with 3 rules that make sense without a cost axis."""
import numpy as np


def hypervolume_2d(points, ref=(0.0, 0.0)):
    """points: array (n,2) of (mcc, phi), BOTH maximized. ref is the
    dominated corner (default (0,0), below which nothing in this study
    falls). HV = area of the union of rectangles [ref_x, mcc_i] x [ref_y, phi_i]
    for points on the non-dominated front."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])
    pts = pts[order]
    hv = 0.0
    max_phi_so_far = ref[1]
    prev_mcc = ref[0]
    # sweep from highest MCC down; only points that raise phi beyond
    # everything seen so far (i.e., are non-dominated) contribute area
    front = []
    best_phi = -np.inf
    for p in pts:
        if p[1] > best_phi:
            front.append(p)
            best_phi = p[1]
    front = np.array(front)  # sorted descending MCC, ascending... need proper sweep
    front = front[np.argsort(front[:, 0])]  # ascending MCC
    hv = 0.0
    prev_x = ref[0]
    for x, y in front:
        hv += (x - prev_x) * (y - ref[1])
        prev_x = x
    return float(hv)


def pareto_mask(points):
    """points: (n,2), maximize both. Returns boolean mask of non-dominated points."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and \
               (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]):
                mask[i] = False
                break
    return mask


def knee_point(df, mcc_col="mcc", phi_col="phi"):
    """Point on the front closest to the ideal corner (max mcc, max phi) in
    normalized space."""
    mcc = df[mcc_col].values
    phi = df[phi_col].values
    mcc_n = (mcc - mcc.min()) / (mcc.max() - mcc.min() + 1e-12)
    phi_n = (phi - phi.min()) / (phi.max() - phi.min() + 1e-12)
    dist = np.sqrt((1 - mcc_n) ** 2 + (1 - phi_n) ** 2)
    return df.iloc[[int(np.argmin(dist))]]


def max_mcc_point(df, mcc_col="mcc"):
    return df.iloc[[int(df[mcc_col].values.argmax())]]


def max_phi_point(df, phi_col="phi"):
    return df.iloc[[int(df[phi_col].values.argmax())]]


def build_navigation_table(front_df):
    rules = {
        "knee_point": knee_point(front_df),
        "max_mcc": max_mcc_point(front_df),
        "max_phi": max_phi_point(front_df),
    }
    rows = []
    for name, sub in rules.items():
        r = sub.iloc[0]
        rows.append({"rule": name, "trial": r["trial"], "model_family": r["model_family"],
                      "mcc": r["mcc"], "phi": r["phi"]})
    return rows
