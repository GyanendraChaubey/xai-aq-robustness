"""Methodology / framework overview diagram for the Materials and Methods
section. Same validated categorical palette and typographic conventions as
make_figures.py, drawn as a simple box-and-arrow flowchart (no external
diagramming dependency)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

FIGS_DIR = "/home/claude/xai_aq/figs"

INK = "#0b0b0b"
SEC_INK = "#52514e"
BLUE = "#2a78d6"
BLUE_FILL = "#e8f1fc"
ORANGE = "#eb6834"
ORANGE_FILL = "#fdece3"
GREEN = "#1baf7a"
GREEN_FILL = "#e5f7f0"
MUTED = "#898781"


def box(ax, xy, w, h, text, edge=INK, fill="white", fontsize=8.6, weight="normal", lw=1.1):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, weight=weight, zorder=4, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=MUTED, lw=1.2, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                         linewidth=lw, color=color, zorder=2, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def bottom_mid(b):
    x, y, w, h = b
    return (x + w / 2, y)


def top_mid(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def right_mid(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left_mid(b):
    x, y, w, h = b
    return (x, y + h / 2)


fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=220)
ax.set_xlim(0, 11.5)
ax.set_ylim(0.7, 7.6)
ax.axis("off")

# --- Row 1: data sources -------------------------------------------------
d1 = box(ax, (0.3, 6.55), 3.2, 0.85,
         "Beijing Multi-Site Air Quality\n12 stations, 2013–2017\n(hourly → daily aggregation)",
         edge=BLUE, fill=BLUE_FILL)
d2 = box(ax, (3.8, 6.55), 3.2, 0.85,
         "Delhi/NCR CPCB station\n2021–2023\n(daily, real sensor gaps)",
         edge=BLUE, fill=BLUE_FILL)

# --- Row 2: feature engineering ------------------------------------------
feat = box(ax, (1.4, 5.35), 5.0, 0.72,
           "Common 20-feature schema + next-day AQI-category target\n"
           "(station identity excluded)",
           edge=INK, fill="white")
arrow(ax, bottom_mid(d1), (bottom_mid(d1)[0], 5.35 + 0.72), color=BLUE)
arrow(ax, bottom_mid(d2), (bottom_mid(d2)[0], 5.35 + 0.72), color=BLUE)

# --- Row 3: two-objective search -----------------------------------------
search = box(ax, (0.9, 4.05), 5.9, 0.85,
             "Two-objective NSGA-II search per primary station\n"
             "9 model families $\\times$ scaling/resampling choices, blocked (walk-forward) CV",
             edge=ORANGE, fill=ORANGE_FILL, fontsize=9)
arrow(ax, bottom_mid(feat), top_mid(search), color=INK)

mcc_box = box(ax, (0.9, 3.05), 2.75, 0.6, "$f_1$ = MCC\n(predictive performance)", fontsize=8.2)
phi_box = box(ax, (4.05, 3.05), 2.75, 0.6, "$f_2$ = $\\Phi$\n(SHAP insertion/deletion faithfulness)", fontsize=8.2)
arrow(ax, bottom_mid((search[0] + 0.2, search[1], 0.1, search[3])), top_mid(mcc_box), color=ORANGE)
arrow(ax, bottom_mid((search[0] + search[2] - 0.3, search[1], 0.1, search[3])), top_mid(phi_box), color=ORANGE)

pareto = box(ax, (2.3, 1.95), 3.1, 0.62, "Pareto front\n(per station, per seed)", edge=INK, fill="white")
arrow(ax, bottom_mid(mcc_box), (pareto[0] + 0.5, pareto[1] + pareto[3]), color=MUTED)
arrow(ax, bottom_mid(phi_box), (pareto[0] + pareto[2] - 0.5, pareto[1] + pareto[3]), color=MUTED)

# --- Navigation rules ------------------------------------------------------
nav_y = 0.95
n1 = box(ax, (0.15, nav_y), 2.15, 0.62, "max-MCC\n(accuracy first)", edge=GREEN, fill=GREEN_FILL, fontsize=8.2)
n2 = box(ax, (2.45, nav_y), 2.15, 0.62, "knee-point\n(balanced)", edge=GREEN, fill=GREEN_FILL, fontsize=8.2)
n3 = box(ax, (4.75, nav_y), 2.15, 0.62, "max-$\\Phi$\n(explainability first)", edge=GREEN, fill=GREEN_FILL, fontsize=8.2)
for nb in (n1, n2, n3):
    arrow(ax, bottom_mid(pareto), top_mid(nb), color=MUTED)

# --- Right column: evaluation ---------------------------------------------
ev_title = box(ax, (7.5, 4.6), 3.7, 0.5, "Post-hoc evaluation\n(no retraining)", edge=INK, fill="#f2f1ec", weight="bold", fontsize=8.6)
e1 = box(ax, (7.5, 3.55), 3.7, 0.72,
         "Sensor-noise / missingness stress test\n(multiplicative noise + MCAR masking, rank stability)",
         edge=BLUE, fill=BLUE_FILL, fontsize=7.8)
e2 = box(ax, (7.5, 2.55), 3.7, 0.72,
         "Spatial transfer\n(8 held-out Beijing stations)",
         edge=BLUE, fill=BLUE_FILL, fontsize=8.0)
e3 = box(ax, (7.5, 1.55), 3.7, 0.72,
         "Cross-city transfer\n(Delhi/NCR, unseen pollution regime)",
         edge=BLUE, fill=BLUE_FILL, fontsize=8.0)

# single connector: the three navigation-point models collectively feed
# the post-hoc evaluation column (routed as one curved arrow to avoid
# crossing back through the MCC/Phi/Pareto boxes)
conn = FancyArrowPatch(right_mid(pareto), left_mid(ev_title),
                        connectionstyle="arc3,rad=0.28", arrowstyle="-|>",
                        mutation_scale=9, linewidth=1.2, color=MUTED,
                        zorder=2, shrinkA=4, shrinkB=4, linestyle=(0, (4, 2)))
ax.add_patch(conn)
ax.text(6.85, 3.95, "selected\nnavigation-point\nmodels", ha="center", va="center",
        fontsize=6.6, color=SEC_INK, style="italic", linespacing=1.15)

arrow(ax, bottom_mid(ev_title), top_mid(e1), color=INK, lw=0.9)
arrow(ax, bottom_mid(e1), top_mid(e2), color=INK, lw=0.9)
arrow(ax, bottom_mid(e2), top_mid(e3), color=INK, lw=0.9)

fig.tight_layout()
fig.savefig(f"{FIGS_DIR}/framework_diagram.pdf", bbox_inches="tight")
fig.savefig(f"{FIGS_DIR}/framework_diagram.png", bbox_inches="tight")
plt.close(fig)
print("OK framework diagram")
