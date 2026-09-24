# Geopolitical Risk and Option Mid-Quote Forecasting — ITA US

Replication materials for the master's dissertation that tests whether the Caldara–Iacoviello Geopolitical Risk (GPR) index improves out-of-sample forecasts of the five-trading-day-ahead mid-quote of American call options on the iShares U.S. Aerospace & Defense ETF (ITA US).

The identifying comparison is **Baseline+VIX** against **Baseline+VIX+GPR**, estimated with the same algorithm, the same sample and the same hyperparameter search. The conclusion rests on the difference in out-of-sample error between these two configurations, not on the ranking of models.

## Contents

All inputs sit in the same folder as the script, and all outputs are written to `output_horse_race/` inside that folder.

```
.
├── horse_race_forecasting_fixed.py   main pipeline: data, features, estimation, predictions
├── analise_resultados.py             evaluation: metrics and tests of Section 3.7, from the saved predictions
├── requirements.txt                  exact package versions of the reported execution
├── PROVENIENCIA.txt                  source, transformation, coverage and download date of each input
├── dados_finais.xlsx                 option data (Bloomberg Spreadsheet Builder)
├── data_gpr_daily_recent.xls         daily GPR index (Caldara & Iacoviello)
├── ita_vix.csv                       VIX, daily close / 100
├── ita_risk_free.csv                 13-week T-bill (^IRX), daily close / 100
├── ita_dividend_yield.csv            ITA trailing 12-month dividends / close
└── output_horse_race/                created by the run
    ├── sample_flow.csv               sample counts at each step
    ├── versions.txt                  package versions and run parameters
    └── ITA/
        ├── predictions_ITA_<config>.csv   test-set forecasts of every model
        ├── results_ITA_<config>.csv       RMSE, MAE, R², Theil's U
        ├── importance_ITA_<config>.csv    variable importance
        └── dataset_ITA_common.csv         common sample used by the four configurations
```

`<config>` is one of `sem_indices` (Baseline), `so_GPR` (Baseline+GPR), `so_VIX` (Baseline+VIX) and `GPR_e_VIX` (Baseline+VIX+GPR).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

The reported execution used Python 3.14.0 with the exact versions pinned in `requirements.txt`.

## Running

The analysis has two steps, run in this order. No path needs to be edited: both scripts locate their inputs in their own folder.

### 1. Estimation

```bash
python horse_race_forecasting_fixed.py
```

Builds the sample and the features, tunes and fits every model under the four configurations, and writes the test-set forecasts to `output_horse_race/`. The run is fully offline (`OFFLINE_ONLY = True`). It stops if the option file or any market series is missing or does not cover the option sample; if the GPR file cannot be read, the error is reported at the end of the run and the configurations containing the index are not valid.

The script also prints diagnostic Diebold–Mariano tables with Holm adjustment across all fitted models. These are not the tables reported in the dissertation, which use the families of Section 3.7.2 and are produced in step 2.

### 2. Evaluation

```bash
python analise_resultados.py
```

Re-estimates nothing. It reads the prediction files written in step 1 and computes every statistic reported in the dissertation:

- RMSE, MAE, out-of-sample R² relative to the random walk and Theil's U, on the full test sample and on the liquidity subsample (volume > 0, open interest > 0, relative spread ≤ 50%);
- Diebold–Mariano tests on date-aggregated losses with the Harvey–Leybourne–Newbold correction, under squared and absolute loss;
- Holm-adjusted p-values within the families of Section 3.7.2: the CRR benchmark and the four main-set estimators against the random walk (J = 5), the four main-set estimators between configurations (J = 4), and all eight estimators in the robustness set (J = 8);
- the central comparison (Baseline+VIX+GPR against Baseline+VIX) and the auxiliary comparisons between configurations;
- the minimum detectable effect, the loss differential by test date, errors by moneyness and the share of variable importance received by the GPR block.

Because it works only from the saved predictions, readers can verify the reported results without re-estimating the models.

## Design summary

| Element | Specification |
|---|---|
| Target | Mid-quote of the same contract on the 5th following ITA trading date |
| Sample | 1,678 contract-day observations, 142 trading dates (Nov 2025 – Jun 2026) |
| Split | 80/20 by trading date, with purging of training observations whose target falls in the test window: 1,082 training / 506 test observations |
| GPR availability | 7-calendar-day lag, backward as-of merge, no backfilling (`*_pit` variables) |
| Hyperparameters | `RandomizedSearchCV`, 20 candidates, 3 expanding purged folds over the second half of the training dates |
| Benchmarks | Naïve random walk; Cox–Ross–Rubinstein roll-down (200 steps, American call with dividend yield) |
| Models | OLS, Ridge, Elastic Net on the price change, XGBoost (main set); Random Forest, LightGBM, CatBoost, MLP (robustness set) |
| Seed | 42, shared by all estimators and searches |

Setting `SCALE_N_ITER = True` reproduces the robustness exercise with 20 candidates per tuned hyperparameter. Setting `GPR_AVAILABILITY_LAG = 0` reproduces the ex post contemporaneous specification. Both write to the same `output_horse_race/` folder, so rename it before running a variant.

## Data sources

- **Options:** Bloomberg Terminal. Bloomberg data are subject to the terminal's licence terms and are included only for replication of the dissertation.
- **GPR:** Caldara, D. & Iacoviello, M. (2022), "Measuring Geopolitical Risk", *American Economic Review*, 112(4), 1194–1225. Daily file, updated every Monday; the file included is the vintage used in the study.
- **VIX, risk-free rate and ITA dividends:** Yahoo Finance via `yfinance`.

Details for each file are in `PROVENIENCIA.txt`.
