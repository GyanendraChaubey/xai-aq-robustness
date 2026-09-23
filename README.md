# Faithfulness Is Not Robustness: Rethinking Explainable Model Selection for Air Quality Forecasting

Code and experimental results for a two-objective explainable-AI model-selection
benchmark, submitted to *Frontiers in Artificial Intelligence* (Machine Learning
and Artificial Intelligence section), Research Topic "Explainable Machine
Learning for Air Quality & Meteorological Prediction."

## Abstract

Model selection under explainability constraints is commonly treated as a
single scalar trade-off between predictive accuracy and explanation
faithfulness. This work casts it instead as a two-objective search — Matthews
Correlation Coefficient (MCC) versus SHAP insertion/deletion faithfulness
(Φ) — solved with NSGA-II under temporally blocked, leave-one-station-out
cross-validation, on the UCI Beijing Multi-Site Air Quality dataset (12
stations, 2013–2017) for next-day AQI-category forecasting. Three
navigation-rule points per station (max-MCC, max-Φ, knee-point) are selected
from each resulting Pareto front and stress-tested under (i) synthetic sensor
noise and missingness, (ii) spatial transfer to held-out Beijing stations,
and (iii) cross-city transfer to an independent Delhi/NCR CPCB station
(2021–2023).

The central finding is that predictive accuracy and explanation faithfulness
are optimized, degraded, and transferred independently of one another. A
model selected purely for high explanation faithfulness can underperform a
naive persistence baseline, and a faithfulness score can remain stable under
distribution shift even as the underlying predictions collapse toward
chance — faithfulness, in other words, is not a proxy for robustness.

## Key Results

- Two-objective (MCC × Φ) search beats single-objective TPE at 4/4 stations
  (Wilcoxon p = 0.125, the maximum attainable significance at n = 4); NSGA-II
  vs. random search is a coin flip (2/4, p = 0.875) — the formulation drives
  the gain, not the sampler.
- Raw faithfulness is an unreliable robustness signal: TreeExplainer-based
  models show ≈0 change in Φ under sensor noise (+0.003), while attribution
  rank stability (Spearman ρ) for the same perturbation drops as low as 0.20.
- Predictive skill and explanation faithfulness transfer independently. The
  max-MCC model transfers best predictively (0.28 held-out, 0.18 Delhi); the
  max-Φ model's faithfulness barely moves under transfer (0.378 → 0.366–0.375)
  even as its own MCC collapses toward chance at the Delhi transfer site
  (0.048).
- At Dongsi, the max-Φ (decision tree) model's own-test MCC (0.048) falls
  below that station's naive persistence baseline (0.168).

## Repository Structure

```
src/        full pipeline — feature engineering, NSGA-II search, stress
            test, transfer evaluation, statistics, figure generation
results/    raw run outputs (per-station trial tables, hypervolume,
            transfer-eval CSVs, paper_summary.json)
figs/       all figures (PDF + PNG), including the methodology/framework
            diagram
```

## Installation

```bash
pip install -r requirements.txt   # optuna, shap, imbalanced-learn, xgboost, lightgbm, scikit-learn
```

## Reproducing the Results

```bash
python src/run_search.py --stations Dongsi,Huairou,Wanshouxigong,Dingling \
    --n-trials 40 --n-seeds 2 --cv-folds 3
python src/run_transfer_eval.py
python src/aggregate_results.py
python src/make_figures.py
```

## Data

- **Beijing Multi-Site Air Quality** — UCI Machine Learning Repository,
  DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G).
- **Delhi/NCR CPCB station data** — source/license to be confirmed before
  final submission.

## Citation

Citation details will be added once the manuscript is accepted and assigned
a DOI.

## License

MIT — see [LICENSE](LICENSE).
