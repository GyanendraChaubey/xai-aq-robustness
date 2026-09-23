# Faithfulness Is Not Robustness: Rethinking Explainable Model Selection for Air Quality Forecasting

Code, results, and manuscript for a two-objective (accuracy x explanation
faithfulness) model-selection benchmark for next-day AQI-category
forecasting, submitted to Frontiers in AI (ML section), Research Topic
"Explainable Machine Learning for Air Quality & Meteorological Prediction".

## What this is

Model selection is cast as a two-objective search (Matthews Correlation
Coefficient x SHAP insertion/deletion faithfulness) solved with NSGA-II
under temporally blocked cross-validation, over the UCI Beijing Multi-Site
Air Quality dataset (12 stations, 2013-2017). Three navigation-rule points
per station (max-MCC, max-faithfulness, knee-point) are selected from each
resulting Pareto front and stress-tested under (i) synthetic sensor noise
and missingness, (ii) spatial transfer to held-out Beijing stations, and
(iii) cross-city transfer to an independent Delhi/NCR CPCB station
(2021-2023).

Headline result: predictive accuracy and explanation faithfulness are
optimized, degraded, and transferred independently of one another -- a
model selected purely for high explanation faithfulness can underperform a
naive persistence baseline, and a faithfulness score can stay stable under
transfer even as the underlying predictions collapse toward chance.

## Repository structure

```
paper/      manuscript source (main.tex, references.bib, main.pdf)
src/        full pipeline: feature engineering, NSGA-II search, stress
            test, transfer evaluation, statistics, figure generation
results/    raw run outputs (per-station trial tables, hypervolume,
            transfer-eval CSVs, paper_summary.json)
figs/       all figures (PDF + PNG), including the framework diagram
```

## Reproducing the results

```
pip install -r requirements.txt   # optuna, shap, imbalanced-learn, xgboost, lightgbm, scikit-learn
python src/run_search.py --stations Dongsi,Huairou,Wanshouxigong,Dingling --n-trials 40 --n-seeds 2 --cv-folds 3
python src/run_transfer_eval.py
python src/aggregate_results.py
python src/make_figures.py
```

Compiling the manuscript:

```
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Data

- Beijing Multi-Site Air Quality: UCI Machine Learning Repository, DOI 10.24432/C5RK5G.
- Delhi/NCR CPCB station data: source/license to be confirmed before final submission.

## License

MIT (see LICENSE).

## Citation

Citation details will be added once the manuscript is accepted / assigned a DOI.
