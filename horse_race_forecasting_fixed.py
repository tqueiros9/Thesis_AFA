#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast of ITA US option mid-quotes five trading days ahead.

Tests whether the Caldara-Iacoviello geopolitical risk index adds information
beyond the characteristics of the option, the underlying ETF and the VIX. It
compares eight models against two benchmarks: the random walk and the CRR
binomial tree.

Inputs (all in this script's folder; the run does not access the internet):
    dados_finais.xlsx              Bloomberg options
    data_gpr_daily_recent.xls      GPR index
    ita_vix.csv                    VIX
    ita_risk_free.csv              risk-free rate (^IRX)
    ita_dividend_yield.csv         ITA dividend yield

Outputs in output_horse_race/, inside this script's folder:
    ITA/results_*.csv       metrics by configuration
    ITA/predictions_*.csv   test-set predictions
    sample_flow.csv         sample counts at each step
    versions.txt            package versions and run parameters

Run with:  python horse_race_forecasting_fixed.py
"""

import warnings
warnings.filterwarnings("ignore")
import tempfile

import re
import numpy as np
import pandas as pd
import yfinance as yf

from pathlib import Path
from scipy.stats import randint, uniform, t as student_t
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_SEED      = 42
np.random.seed(RANDOM_SEED)

# Folder containing the script: inputs and outputs all live here
PROJECT_DIR      = Path(__file__).resolve().parent

OPTIONS_FILE     = PROJECT_DIR / "dados_finais.xlsx"
GPR_FILE         = PROJECT_DIR / "data_gpr_daily_recent.xls"
OUTPUT_DIR       = PROJECT_DIR / "output_horse_race"

# Market data stored locally
DIV_YIELD_FILE   = PROJECT_DIR / "ita_dividend_yield.csv"
VIX_FILE         = PROJECT_DIR / "ita_vix.csv"
RF_FILE          = PROJECT_DIR / "ita_risk_free.csv"

RISK_FREE_TICKER = "^IRX"
VIX_TICKER       = "^VIX"

# Run without internet access: requires the CSV files in the script folder.
# Set to False only to regenerate missing series.
OFFLINE_ONLY     = True
PROVENANCE_FILE  = PROJECT_DIR / "PROVENIENCIA.txt"

TRAIN_RATIO      = 0.80

# Hyperparameter optimisation
TUNE_HYPERPARAMS = True
N_ITER_SEARCH    = 20     # candidate combinations (base)
CV_SPLITS        = 3

# True: n_iter = N_ITER_SEARCH * number of hyperparameters of the model.
# Tested: median change of 0.00% in RMSE. The fixed budget is kept.
SCALE_N_ITER     = False

# Rolling origin (--origem-movel): temporal robustness added after the
# primary specification. Expanding window, one test block per calendar month
# from ORIGEM_PRIMEIRO_MES, hyperparameters re-tuned at each origin using only
# data before the block. Main set + benchmarks.
ORIGEM_MOVEL         = False
ORIGEM_PRIMEIRO_MES  = "2026-03"
ORIGEM_MODELOS       = ["OLS", "Ridge", "XGBoost"]   # + ElasticNet (delta)

# Main horizon (trading days)
MAIN_HORIZON_DAYS = 5

# GPR availability lag (calendar days)
GPR_AVAILABILITY_LAG = 7

# Liquidity filter: 0 = accept all observations
MIN_LIQUID_VOL = 0
MIN_LIQUID_OI  = 0


# =============================================================================
# SECTION 1 — LOADING THE OPTIONS EXCEL FILE
# =============================================================================

def parse_contract_name(name: str):
    """Extracts strike, expiry date and underlying from the contract name."""
    name = str(name).strip()
    m = re.search(r'\b([CP])(\d+(?:\.\d+)?)\b', name)
    strike = float(m.group(2)) if m else None

    all_dates = re.findall(r'\d{2}/\d{2}/\d{2}', name)
    expiry_date = None
    if all_dates:
        try:
            expiry_date = pd.to_datetime(all_dates[-1], format="%m/%d/%y")
        except Exception:
            pass

    underlying = None
    if len(all_dates) >= 2:
        m2 = re.search(r'\d{2}/\d{2}/\d{2}\s+(.+?)\s+\d{2}/\d{2}/\d{2}', name)
        if m2:
            underlying = m2.group(1).strip()
    if underlying is None:
        parts = re.split(r'\d{2}/\d{2}/\d{2}', name)
        candidate = parts[0].strip()
        if candidate:
            underlying = candidate

    return strike, expiry_date, underlying


def load_options_excel(path: Path) -> pd.DataFrame:
    """Reads the Bloomberg file: 8 columns per contract, data from row 2."""
    raw    = pd.read_excel(path, header=None, dtype=str)
    n_cols = raw.shape[1]
    h0     = raw.iloc[0].tolist()
    h1     = raw.iloc[1].tolist()   # noqa: F841
    data   = raw.iloc[2:].reset_index(drop=True)

    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")

    spot_spx = pd.to_numeric(data.iloc[:, -2], errors="coerce")
    spot_ita = pd.to_numeric(data.iloc[:, -1], errors="coerce")

    base_df = pd.DataFrame({
        "date"              : dates,
        "spot_spx_index"    : spot_spx,
        "spot_ita_us_equity": spot_ita,
    })

    n_option_cols = n_cols - 1 - 2
    n_contracts   = n_option_cols // 8

    CONTRACT_COLS = [
        "PX_LAST", "PX_BID", "PX_ASK", "MATURITY",
        "IVOL_MID", "PX_VOLUME", "OPEN_INT", "EOD_TIME_TO_EXPIRY_LAST",
    ]

    option_records = []

    for i in range(n_contracts):
        base  = 1 + i * 8
        label = str(h0[base]).strip()
        if label in {"nan", "", "None"}:
            continue

        grp = data.iloc[:, base : base + 8].copy()
        grp.columns = CONTRACT_COLS

        for col in CONTRACT_COLS:
            grp[col] = pd.to_numeric(grp[col], errors="coerce")

        grp["mid_price"] = (grp["PX_BID"] + grp["PX_ASK"]) / 2.0
        grp["price"]     = grp["mid_price"]
        grp["iv"]        = grp["IVOL_MID"] / 100.0

        full = pd.concat([base_df.reset_index(drop=True),
                          grp.reset_index(drop=True)], axis=1)
        full = full.dropna(subset=["date"]).copy()

        has_data = full["price"].notna() | full["iv"].notna()
        full = full[has_data].copy()
        if full.empty:
            continue

        strike, expiry_date, underlying_name = parse_contract_name(label)

        full["contract"]        = label
        full["opt_type"]        = "call"
        full["strike"]          = strike
        full["expiry_date"]     = expiry_date
        full["underlying_name"] = underlying_name

        if expiry_date is not None:
            computed_dtm = (expiry_date - full["date"]).dt.days
            eod_dtm      = full["EOD_TIME_TO_EXPIRY_LAST"].round()
            full["days_to_maturity"] = eod_dtm.fillna(computed_dtm)
        else:
            full["days_to_maturity"] = full["EOD_TIME_TO_EXPIRY_LAST"].round()

        full = full.rename(columns={
            "PX_LAST"                : "px_last",
            "PX_BID"                 : "px_bid",
            "PX_ASK"                 : "px_ask",
            "MATURITY"               : "maturity_raw",
            "IVOL_MID"               : "ivol_mid_pct",
            "PX_VOLUME"              : "px_volume",
            "OPEN_INT"               : "open_int",
            "EOD_TIME_TO_EXPIRY_LAST": "eod_tte",
        })

        option_records.append(full)

    if not option_records:
        raise ValueError("Nenhum contrato encontrado.")

    options_df = pd.concat(option_records, ignore_index=True)
    options_df = options_df.sort_values(["contract", "date"]).reset_index(drop=True)
    return options_df


# =============================================================================
# SECTION 2 — LOADING THE GPR
# =============================================================================

def load_gpr(path: Path) -> pd.DataFrame:
    """Loads the GPR index and computes the derived variables."""
    raw = pd.read_excel(path)
    raw.columns = raw.columns.str.strip()

    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], dayfirst=True, errors="coerce")
    elif "DATE" in raw.columns:
        raw["date"] = pd.to_datetime(raw["DATE"], dayfirst=True, errors="coerce")
    elif "DAY" in raw.columns:
        raw["date"] = pd.to_datetime(
            raw["DAY"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        raise ValueError("Coluna de data (date/DATE/DAY) nao encontrada no GPR.")

    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    rename_map = {
        "GPRD"       : "gpr",
        "GPRD_AC"    : "gpr_ac",
        "GPRDACT"    : "gpr_ac",
        "GPRD_ACT"   : "gpr_ac",
        "GPRD_TH"    : "gpr_th",
        "GPRDTHREAT" : "gpr_th",
        "GPRD_THREAT": "gpr_th",
        "GPRD_MA7"   : "gpr_ma7",
        "GPRD_MA30"  : "gpr_ma30",
        "N10D"       : "gpr_n10d",  # count of newspaper articles on GPR over the last 10 days
    }
    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    base_cols = ["gpr", "gpr_ac", "gpr_th", "gpr_ma7", "gpr_ma30", "gpr_n10d"]
    keep      = ["date"] + [c for c in base_cols if c in raw.columns]
    gpr       = raw[keep].copy()

    for c in base_cols:
        if c not in gpr.columns:
            continue
        if gpr[c].dtype == object:
            s = (gpr[c].astype(str)
                 .str.strip()
                 .str.replace(".", "", regex=False)
                 .str.replace(",", ".", regex=False))
            gpr[c] = pd.to_numeric(s, errors="coerce")
        else:
            gpr[c] = pd.to_numeric(gpr[c], errors="coerce")

    if "gpr_ma7" not in gpr.columns and "gpr" in gpr.columns:
        gpr["gpr_ma7"]  = gpr["gpr"].rolling(7,  min_periods=1).mean()
    if "gpr_ma30" not in gpr.columns and "gpr" in gpr.columns:
        gpr["gpr_ma30"] = gpr["gpr"].rolling(30, min_periods=1).mean()

    for base in ["gpr", "gpr_ac", "gpr_th"]:
        if base not in gpr.columns:
            continue
        gpr[f"{base}_change"]     = gpr[base].diff().fillna(0.0)
        gpr[f"{base}_pct_change"] = (
            gpr[base].pct_change()
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )

    if "gpr" in gpr.columns:
        if "gpr_ma7"  in gpr.columns:
            gpr["gpr_ma_diff7"]  = gpr["gpr"] - gpr["gpr_ma7"]
        if "gpr_ma30" in gpr.columns:
            gpr["gpr_ma_diff30"] = gpr["gpr"] - gpr["gpr_ma30"]

    return gpr


# =============================================================================
# SECTION 3 — MARKET DATA (with local save/load)
# =============================================================================

def _descarregar(ticker: str, start_date, end_date, col_name: str) -> pd.DataFrame:
    """Downloads the close of a ticker. Only called with OFFLINE_ONLY=False."""
    df = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=5),
                     progress=False, auto_adjust=False, multi_level_index=False)
    if df.empty:
        raise ValueError(f"{ticker}: descarga vazia.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df["date"]   = pd.to_datetime(df[date_col]).dt.normalize()
    df[col_name] = pd.to_numeric(df["close"], errors="coerce")
    return (df[["date", col_name]].dropna().drop_duplicates("date")
              .sort_values("date").reset_index(drop=True))


def _verificar_cobertura(df, nome, start_date, end_date):
    """Confirms that the series covers the whole option period."""
    if df.empty:
        raise ValueError(f"{nome}: ficheiro vazio.")
    ini, fim = df["date"].min(), df["date"].max()
    if ini > pd.Timestamp(start_date) or fim < pd.Timestamp(end_date):
        raise ValueError(
            f"{nome} cobre {ini.date()} a {fim.date()}, insuficiente para o "
            f"periodo das opcoes ({pd.Timestamp(start_date).date()} a "
            f"{pd.Timestamp(end_date).date()})."
        )
    return ini, fim


def verificar_dados_mercado(start_date, end_date):
    """
    Checks the three market series before any of them is used, so that
    all problems are reported at once and not one per run.
    """
    series = [
        (RF_FILE,        "taxa sem risco", RISK_FREE_TICKER),
        (VIX_FILE,       "VIX",            VIX_TICKER),
        (DIV_YIELD_FILE, "dividend yield", None),
    ]
    em_falta, incompletos = [], []

    for path, nome, ticker in series:
        if not path.exists():
            origem = f"descarregar de {ticker}" if ticker else "pre-calculado"
            em_falta.append(f"    {path.name:<24} {nome:<16} ({origem})")
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = df["date"].dt.normalize()
            _verificar_cobertura(df, path.name, start_date, end_date)
        except Exception as exc:
            incompletos.append(f"    {exc}")

    if not em_falta and not incompletos:
        return

    linhas = [f"Nao e possivel correr: {len(em_falta) + len(incompletos)} "
              f"de {len(series)} series de mercado com problemas.", ""]
    if em_falta:
        linhas += ["  Em falta na pasta do script:"] + em_falta + [""]
    if incompletos:
        linhas += ["  Nao cobrem o periodo das opcoes:"] + incompletos + [""]
    linhas += [
        "  Coloque os ficheiros na pasta do script. As series com ticker podem",
        "  ser descarregadas definindo OFFLINE_ONLY=False; o dividend yield e",
        "  pre-calculado (dividendos de 12 meses / preco de fecho) e tem de ser",
        "  gerado a parte.",
    ]
    raise FileNotFoundError("\n".join(linhas))


def _registar_proveniencia(ticker, path, start_date, end_date):
    """Records ticker, period and download date."""
    PROVENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE_FILE, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M}  {ticker:<6} -> {path.name}  "
                f"periodo {pd.Timestamp(start_date).date()} a "
                f"{pd.Timestamp(end_date).date()}\n")


def _serie_mercado(path, col, ticker, start_date, end_date):
    """Reads the series from the local CSV; only downloads if OFFLINE_ONLY is False."""
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.normalize()
        ini, fim = _verificar_cobertura(df, path.name, start_date, end_date)
        print(f"    {path.name}: {len(df)} linhas, {ini.date()} a {fim.date()}")
        return df[["date", col]]

    if OFFLINE_ONLY:
        raise FileNotFoundError(
            f"{path} nao existe e OFFLINE_ONLY=True.\n"
            f"    Coloque o ficheiro na pasta do script, ou defina OFFLINE_ONLY=False "
            f"para o descarregar de {ticker}."
        )

    df = _descarregar(ticker, start_date, end_date, col)
    df[col] = df[col] / 100.0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _registar_proveniencia(ticker, path, start_date, end_date)
    print(f"    {path.name}: descarregado e guardado ({len(df)} linhas)")
    return df


def get_risk_free_rate(start_date, end_date) -> pd.DataFrame:
    """Risk-free rate (^IRX), already in decimal."""
    return _serie_mercado(RF_FILE, "risk_free_rate", RISK_FREE_TICKER, start_date, end_date)


def get_vix(start_date, end_date) -> pd.DataFrame:
    """VIX, already in decimal."""
    return _serie_mercado(VIX_FILE, "vix", VIX_TICKER, start_date, end_date)


def get_dividend_yield(start_date, end_date) -> pd.DataFrame:
    """ITA dividend yield (12-month sum over the closing price)."""
    if not DIV_YIELD_FILE.exists():
        raise FileNotFoundError(
            f"{DIV_YIELD_FILE} nao existe.\n"
            "    Esta serie e pre-calculada (dividendos de 12 meses / preco de fecho) "
            "e nao pode ser descarregada diretamente."
        )
    df = pd.read_csv(DIV_YIELD_FILE, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    df = df[["date", "dividend_yield"]].dropna().sort_values("date").reset_index(drop=True)
    ini, fim = _verificar_cobertura(df, DIV_YIELD_FILE.name, start_date, end_date)
    print(f"    {DIV_YIELD_FILE.name}: {len(df)} linhas, {ini.date()} a {fim.date()}")
    return df


# =============================================================================
# SECTION 4 — SUPERVISOR'S HELPER FUNCTIONS
# =============================================================================

def add_exact_horizon_target(df, master_calendar, horizon=5,
                             contract_col="contract", date_col="date",
                             price_col="price"):
    """Attaches the price of the same contract on the h-th following trading date."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(master_calendar)
    ).normalize().unique().sort_values()

    if horizon < 1 or horizon >= len(calendar):
        raise ValueError("Horizonte fora dos limites do calendario.")

    target_map = pd.Series(calendar[horizon:].values, index=calendar[:-horizon])
    out["target_date"] = out[date_col].map(target_map)

    future = out[[contract_col, date_col, price_col]].copy()
    if future.duplicated([contract_col, date_col]).any():
        raise ValueError("Observacoes repetidas para o mesmo contrato e data.")
    future = future.rename(columns={date_col: "target_date", price_col: "target_h"})

    out = out.merge(
        future,
        on=[contract_col, "target_date"],
        how="left",
        validate="many_to_one",
    )
    out["naive_forecast"] = out[price_col]
    return out


def merge_gpr_with_availability_lag(options_df, gpr_df, feature_cols,
                                    availability_lag_days=7, date_col="date"):
    """Merges the GPR with an availability lag (the file is published weekly)."""
    left = options_df.copy()
    left[date_col] = pd.to_datetime(left[date_col]).dt.normalize()
    left = left.sort_values(date_col)

    right = gpr_df[[date_col] + list(feature_cols)].copy()
    right[date_col] = pd.to_datetime(right[date_col]).dt.normalize()
    right["available_date"] = right[date_col] + pd.Timedelta(days=availability_lag_days)
    right = right.drop(columns=[date_col]).sort_values("available_date")
    right = right.rename(columns={c: f"{c}_pit" for c in feature_cols})

    merged = pd.merge_asof(
        left,
        right,
        left_on=date_col,
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.drop(columns=["available_date"])


def purged_holdout_split(df, train_ratio=0.80, date_col="date",
                         target_date_col="target_date"):
    """Splits train and test without letting training targets enter the test period."""
    frame = df.sort_values(date_col).copy()
    dates = pd.DatetimeIndex(frame[date_col].dropna().unique()).sort_values()
    cut = int(len(dates) * train_ratio)
    if cut <= 0 or cut >= len(dates):
        raise ValueError("Proporcao invalida para split.")

    first_test_date = pd.Timestamp(dates[cut])
    train = frame[
        (frame[date_col] < first_test_date)
        & (frame[target_date_col] < first_test_date)
    ].copy()
    test = frame[frame[date_col] >= first_test_date].copy()
    return train, test, first_test_date


def purged_date_cv(df, n_splits=3, initial_train_fraction=0.50,
                   date_col="date", target_date_col="target_date"):
    """Expanding folds by date for RandomizedSearchCV (no look-ahead)."""
    frame = df.reset_index(drop=True).copy()
    dates = pd.DatetimeIndex(frame[date_col].dropna().unique()).sort_values()
    first_validation = int(len(dates) * initial_train_fraction)
    validation_dates = dates[first_validation:]
    blocks = [b for b in np.array_split(validation_dates, n_splits) if len(b)]

    folds = []
    for block in blocks:
        validation_start = pd.Timestamp(block[0])
        train_mask = (
            (frame[date_col] < validation_start)
            & (frame[target_date_col] < validation_start)
        )
        validation_mask = frame[date_col].isin(block)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx   = np.flatnonzero(validation_mask.to_numpy())
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx))

    if len(folds) != n_splits:
        raise ValueError("Nao foi possivel construir o numero pedido de folds.")
    return folds


def crr_american_call(S, K, r, q, sigma, T, steps=200):
    """Price of an American call via the Cox-Ross-Rubinstein tree."""
    values = np.asarray([S, K, r, q, sigma, T], dtype=float)
    if not np.isfinite(values).all() or S <= 0 or K <= 0 or sigma <= 0:
        return np.nan
    if T <= 0:
        return max(S - K, 0.0)

    dt       = T / steps
    up       = np.exp(sigma * np.sqrt(dt))
    down     = 1.0 / up
    prob     = (np.exp((r - q) * dt) - down) / (up - down)
    if not 0.0 <= prob <= 1.0:
        return np.nan
    discount = np.exp(-r * dt)

    j      = np.arange(steps + 1)
    stock  = S * up ** j * down ** (steps - j)
    option = np.maximum(stock - K, 0.0)

    for level in range(steps - 1, -1, -1):
        continuation = discount * (
            prob * option[1:level + 2]
            + (1.0 - prob) * option[:level + 1]
        )
        j        = np.arange(level + 1)
        stock    = S * up ** j * down ** (level - j)
        exercise = np.maximum(stock - K, 0.0)
        option   = np.maximum(exercise, continuation)
    return float(option[0])


def crr_roll_down_forecast(row, steps=200):
    """CRR price keeping spot, IV, rate and dividends at t and reducing the maturity."""
    elapsed = (pd.Timestamp(row["target_date"]) - pd.Timestamp(row["date"])).days
    future_T = max(float(row["T"]) - elapsed / 365.0, 0.0)
    q = float(row["dividend_yield"]) if pd.notna(row.get("dividend_yield")) else 0.0
    return crr_american_call(
        S     = float(row["spot"]),
        K     = float(row["strike"]),
        r     = float(row["risk_free_rate"]),
        q     = q,
        sigma = float(row["iv"]),
        T     = future_T,
        steps = steps,
    )


def diebold_mariano_by_date(dates, y_true, prediction_a, prediction_b,
                            horizon=5, loss="squared"):
    """
    Diebold-Mariano test with the loss aggregated by date, using the
    Harvey, Leybourne and Newbold (1997) small-sample correction.
    A negative statistic favours forecast A.
    """
    y = np.asarray(y_true,        dtype=float)
    a = np.asarray(prediction_a,  dtype=float)
    b = np.asarray(prediction_b,  dtype=float)

    if loss == "squared":
        diff = (y - a) ** 2 - (y - b) ** 2
    elif loss == "absolute":
        diff = np.abs(y - a) - np.abs(y - b)
    else:
        raise ValueError("loss deve ser 'squared' ou 'absolute'.")

    frame = pd.DataFrame({"date": pd.to_datetime(dates), "d": diff}).dropna()
    daily = frame.groupby("date", sort=True)["d"].mean().to_numpy()
    n = len(daily)

    if n <= max(10, horizon + 1):
        return {"statistic": np.nan, "p_value": np.nan,
                "n_dates": n, "mean_loss_diff": np.nan}

    mean_d   = daily.mean()
    centered = daily - mean_d
    gamma0   = np.mean(centered ** 2)
    autocovs = [np.mean(centered[lag:] * centered[:-lag]) for lag in range(1, horizon)]
    lrv      = gamma0 + 2.0 * sum(autocovs)

    if lrv <= 0:
        return {"statistic": np.nan, "p_value": np.nan,
                "n_dates": n, "mean_loss_diff": mean_d}

    raw_stat   = mean_d / np.sqrt(lrv / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic  = raw_stat * correction
    p_value    = 2.0 * student_t.sf(abs(statistic), df=n - 1)

    return {
        "statistic"     : float(statistic),
        "p_value"       : float(p_value),
        "n_dates"       : int(n),
        "mean_loss_diff": float(mean_d),
    }


def holm_adjust(p_values):
    """Sequential Holm-Bonferroni adjustment (with monotonicity enforced)."""
    p      = np.asarray(p_values, dtype=float)
    order  = np.argsort(p)
    ordered = p[order]
    multipliers = len(p) - np.arange(len(p))
    adj_ordered = np.maximum.accumulate(ordered * multipliers)
    adj_ordered = np.minimum(adj_ordered, 1.0)
    adjusted = np.empty_like(adj_ordered)
    adjusted[order] = adj_ordered
    return adjusted


# =============================================================================
# SECTION 5 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the explanatory variables with information available at t."""
    df = df.sort_values(["contract", "date"]).copy()
    g  = df.groupby("contract", sort=False)

    df["spread"]         = df["px_ask"] - df["px_bid"]
    df["volume_log"]     = np.log1p(df["px_volume"].fillna(0))
    df["open_int_log"]   = np.log1p(df["open_int"].fillna(0))
    df["T"]              = (df["days_to_maturity"] / 365.0).clip(lower=0.0)

    df["price_lag1"]      = g["price"].transform(lambda x: x.shift(1))
    df["price_return_1d"] = (
        g["price"].transform(lambda x: x.pct_change())
        .replace([np.inf, -np.inf], np.nan)
    )
    df["iv_lag1"]         = g["iv"].transform(lambda x: x.shift(1))
    df["iv_change"]       = g["iv"].transform(lambda x: x.diff())
    df["spread_lag1"]     = g["spread"].transform(lambda x: x.shift(1))
    df["volume_log_lag1"] = g["volume_log"].transform(lambda x: x.shift(1))

    def _pick_spot(row):
        name = str(row.get("underlying_name", "")).upper()
        if "ITA" in name:
            return row.get("spot_ita_us_equity", np.nan)
        elif "SPXW" in name or "SPX" in name:
            return row.get("spot_spx_index", np.nan)
        for sc in ["spot_ita_us_equity", "spot_spx_index"]:
            val = row.get(sc, np.nan)
            if pd.notna(val):
                return val
        return np.nan

    spot_cols_present = [c for c in df.columns if c.startswith("spot_")]
    if spot_cols_present:
        df["spot"] = df.apply(_pick_spot, axis=1)

    if "spot" in df.columns and "strike" in df.columns:
        df["moneyness"]      = df["spot"] / df["strike"]
        df["log_moneyness"]  = np.log(df["moneyness"].replace(0, np.nan))
        df["intrinsic_call"] = (df["spot"] - df["strike"]).clip(lower=0.0)
        df["time_value_t"]   = (df["price"] - df["intrinsic_call"]).clip(lower=0.0)

        # Approximate intrinsic value at t+h under the risk-neutral measure.
        if "risk_free_rate" in df.columns:
            df["spot_fwd"] = df["spot"] * np.exp(
                df["risk_free_rate"].fillna(0.045) * MAIN_HORIZON_DAYS / 365.0
            )
        else:
            df["spot_fwd"] = df["spot"]
        df["intrinsic_fwd_approx"] = (df["spot_fwd"] - df["strike"]).clip(lower=0.0)

        gs = df.groupby("contract", sort=False)
        df["spot_return_1d"] = (
            gs["spot"].transform(lambda x: x.pct_change())
            .replace([np.inf, -np.inf], np.nan)
        )
        df["spot_ma5"] = gs["spot"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    return df


# =============================================================================
# SECTION 6 — TARGET AND SPLIT
# =============================================================================

def build_forecast_target(df: pd.DataFrame, master_calendar,
                          horizon: int = 5) -> pd.DataFrame:
    """Builds the target on the h-th following trading date."""
    df = df.sort_values(["contract", "date"]).copy()
    df = df[df["days_to_maturity"] > horizon].copy()

    df = add_exact_horizon_target(df, master_calendar, horizon=horizon)

    # Rename for compatibility with the rest of the pipeline
    df = df.rename(columns={"target_h": f"target_{horizon}d"})
    df["target_t1"] = df[f"target_{horizon}d"]   # working column

    df = df.dropna(subset=["target_t1", "target_date"]).copy()
    return df


# =============================================================================
# SECTION 7 — MODELS
# =============================================================================

def build_models() -> dict:
    """Horse-race models. Linear models and MLP get a StandardScaler; trees do not."""
    return {
        "OLS": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=RANDOM_SEED,
            verbosity=0,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=RANDOM_SEED,
            verbose=0,
            train_dir=tempfile.gettempdir(),
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=False,
                random_state=RANDOM_SEED,
            )),
        ]),
    }


def build_elastic_net_model():
    """ElasticNet with the supervisor's hyperparameters (suited to correlated GPR features)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNet(
            alpha=0.01,
            l1_ratio=0.50,
            max_iter=20000,
            random_state=RANDOM_SEED,
        )),
    ])


# =============================================================================
# SECTION 8 — EVALUATION AND DIEBOLD-MARIANO
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = "",
             naive_rmse: float = None) -> dict:
    """RMSE, MAE, R2, Theil_U for option price forecasting."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)

    theil_u = round(rmse / naive_rmse, 4) if (naive_rmse and naive_rmse > 0) else np.nan

    return {
        "model"  : name,
        "RMSE"   : round(rmse, 6),
        "MAE"    : round(mae,  6),
        "R2"     : round(r2,   6),
        "Theil_U": theil_u,
    }


def get_feature_importance(fitted_model, feature_names: list):
    est = fitted_model
    if hasattr(fitted_model, "named_steps"):
        est = fitted_model.named_steps.get("model", fitted_model)
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=feature_names)
    if hasattr(est, "coef_"):
        return pd.Series(np.abs(est.coef_), index=feature_names)
    return None


def liquidity_report(df: pd.DataFrame) -> pd.Series:
    vol = df["px_volume"].fillna(0) if "px_volume" in df.columns else pd.Series(0, index=df.index)
    oi  = df["open_int"].fillna(0)  if "open_int"  in df.columns else pd.Series(0, index=df.index)
    n   = len(df)

    print(f"\n  -- Analise de Liquidez ({n} obs totais) --")
    for label, series, thresh in [
        ("Volume = 0",  vol, 0),
        ("Volume < 10", vol, 10),
        ("OI = 0",      oi,  0),
        ("OI < 50",     oi,  50),
    ]:
        count = (series <= thresh).sum()
        print(f"    {label:<16}: {count:>5} obs  ({100*count/n:5.1f}%)")

    # Liquidity subset recommended by the supervisor
    spread_ratio = (df["spread"] / df["price"]).fillna(np.inf) if "spread" in df.columns else pd.Series(np.inf, index=df.index)
    liquid_strict = (vol > 0) & (oi > 0) & (spread_ratio <= 0.50)
    print(f"    Robusto (vol>0 & OI>0 & spread<50%): "
          f"{liquid_strict.sum():>5} obs  ({100*liquid_strict.sum()/n:5.1f}%)")

    liquid = (vol > MIN_LIQUID_VOL) & (oi > MIN_LIQUID_OI)
    print(f"    Principal (vol>{MIN_LIQUID_VOL} & OI>{MIN_LIQUID_OI}): "
          f"{liquid.sum():>5} obs  ({100*liquid.sum()/n:5.1f}%)")
    return liquid


# =============================================================================
# SECTION 8b — HYPERPARAMETERS
# =============================================================================

def get_param_grids() -> dict:
    return {
        "Ridge": {
            "model__alpha": uniform(loc=0.01, scale=99.99),
        },
        "ElasticNet": {
            "model__alpha":    uniform(1e-4, 0.2),
            "model__l1_ratio": uniform(0.1, 0.9),
        },
        "RandomForest": {
            "n_estimators":     randint(100, 600),
            "max_depth":        randint(3, 14),
            "min_samples_leaf": randint(2, 16),
        },
        "XGBoost": {
            "n_estimators":     randint(200, 1200),
            "max_depth":        randint(3, 8),
            "learning_rate":    uniform(0.005, 0.145),
            "subsample":        uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": randint(3, 20),
        },
        "LightGBM": {
            "n_estimators":      randint(200, 1200),
            "max_depth":         randint(3, 8),
            "learning_rate":     uniform(0.005, 0.145),
            "subsample":         uniform(0.6, 0.4),
            "colsample_bytree":  uniform(0.5, 0.5),
            "min_child_samples": randint(3, 20),
        },
        "CatBoost": {
            "iterations":    randint(200, 1200),
            "depth":         randint(3, 8),
            "learning_rate": uniform(0.005, 0.145),
        },
        "MLP": {
            "model__hidden_layer_sizes": [(64,), (128,), (128, 64), (128, 64, 32), (256, 128, 64)],
            "model__alpha":              uniform(1e-5, 0.1),
            "model__learning_rate_init": uniform(1e-4, 0.01),
        },
    }


def tune_model(name, model, X_train, y_train, param_grids, cv_folds=None):
    """
    Randomised hyperparameter search over purged folds.
    No grid (OLS) or no folds: trains with the default values.
    """
    grid = param_grids.get(name)
    if grid is None or cv_folds is None:
        model.fit(X_train, y_train)
        return model, 0

    n_iter = N_ITER_SEARCH * len(grid) if SCALE_N_ITER else N_ITER_SEARCH

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=grid,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, n_iter


# =============================================================================
# SECTION 9 — AUXILIARY HORSE RACE
# =============================================================================

def _print_dm_table(dates_test, y_test, predictions, naive_pred, h=5):
    """Table of the DM test by date, with Holm correction."""
    print(f"\n  Diebold-Mariano vs Naive (por data, h={h}d, HLN 1997)")
    print(f"  Perda agregada por data: {pd.DatetimeIndex(dates_test).nunique()} datas no teste")
    print(f"  Correcao Holm-Bonferroni (familia={len(predictions)} testes)")
    print(f"  DM_stat > 0 = modelo PIOR; < 0 = modelo MELHOR")
    print(f"  {'Modelo':<28} {'DM_stat':>10} {'N_datas':>8} {'p_bruto':>10} {'p_holm':>10}  Sig(Holm)")
    print(f"  {'-'*76}")

    dm_results = []
    for name, preds in predictions.items():
        res = diebold_mariano_by_date(dates_test, y_test, preds, naive_pred, horizon=h)
        if not np.isnan(res["statistic"]):
            dm_results.append((name, res["statistic"], res["n_dates"], res["p_value"]))

    if not dm_results:
        return

    p_values = [r[3] for r in dm_results]
    p_adj    = holm_adjust(p_values)

    for (name, dm_stat, n_dates, p_raw), p_h in zip(dm_results, p_adj):
        stars = ""
        if p_h < 0.01:  stars = "***"
        elif p_h < 0.05: stars = "**"
        elif p_h < 0.10: stars = "*"
        print(f"  {name:<28} {dm_stat:>10.4f} {n_dates:>8} {p_raw:>10.4f} {p_h:>10.4f}  {stars}")


def _print_moneyness_breakdown(test_df, y_test, predictions, naive_pred):
    if "moneyness" not in test_df.columns:
        return
    m = test_df["moneyness"].values
    groups = {
        "ITM  (m > 1.05)":          m > 1.05,
        "ATM  (0.95 <= m <= 1.05)": (m >= 0.95) & (m <= 1.05),
        "OTM  (m < 0.95)":          m < 0.95,
    }
    model_names = [k for k in predictions][:7]
    header_models = "".join(f"  {mn[:9]:>9}" for mn in model_names)
    print(f"\n  {'Grupo':<28} {'N':>5}  {'Naive':>8}{header_models}")
    print("  " + "-" * (44 + 11 * len(model_names)))

    for gname, mask in groups.items():
        n_g = int(mask.sum())
        if n_g < 5:
            print(f"  {gname:<28} {n_g:>5}  [omitido: <5 obs]")
            continue
        naive_r = np.sqrt(mean_squared_error(y_test[mask], naive_pred[mask]))
        row_str = f"  {gname:<28} {n_g:>5}  {naive_r:>8.4f}"
        for mn in model_names:
            preds = predictions.get(mn, np.full(len(y_test), np.nan))
            p_g   = preds[mask]
            valid = ~np.isnan(p_g)
            if valid.sum() >= 5:
                rmse_g = np.sqrt(mean_squared_error(y_test[mask][valid], p_g[valid]))
                row_str += f"  {rmse_g:>9.4f}"
            else:
                row_str += f"  {'N/A':>9}"
        print(row_str)


def _print_results_table(results_df, title):
    has_theil = "Theil_U" in results_df.columns
    width = 90 if has_theil else 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    hdr = f"  {'#':<4} {'Modelo':<30} {'RMSE':>10} {'MAE':>10} {'R2':>10}"
    if has_theil:
        hdr += f" {'Theil_U':>9}"
    print(hdr)
    print(f"  {'-' * (width - 2)}")
    for rank, row in results_df.iterrows():
        marker = "  <-- MELHOR" if rank == 1 else ""
        line = (
            f"  {rank:<4} {row['model']:<30} "
            f"{row['RMSE']:>10.4f} {row['MAE']:>10.4f} "
            f"{row['R2']:>10.4f}"
        )
        if has_theil:
            tu = row.get("Theil_U", float("nan"))
            line += f" {tu:>9.4f}" if not (isinstance(tu, float) and (tu != tu)) else f" {'nan':>9}"
        line += marker
        print(line)
    print(f"{'=' * width}")


def _print_gpr_impact(res_com_gpr, res_sem_gpr, title):
    print(f"\n{'=' * 72}")
    print(f"  IMPACTO DO GPR — {title}")
    print(f"  Delta RMSE = RMSE(sem GPR) - RMSE(com GPR)   [+ = GPR ajuda]")
    print(f"{'=' * 72}")
    print(f"  {'Modelo':<28} {'com GPR':>10} {'sem GPR':>10} {'Delta':>8} {'Delta%':>8}")
    print(f"  {'-' * 68}")
    r_com = res_com_gpr.set_index("model")["RMSE"]
    r_sem = res_sem_gpr.set_index("model")["RMSE"]
    rows = []
    for m in r_com.index.intersection(r_sem.index):
        delta = r_sem[m] - r_com[m]
        pct   = delta / r_com[m] * 100
        rows.append((m, r_com[m], r_sem[m], delta, pct))
    rows.sort(key=lambda x: x[3], reverse=True)
    for m, com, sem, delta, pct in rows:
        sign = "+" if delta >= 0 else ""
        print(f"  {m:<28} {com:>10.4f} {sem:>10.4f} {sign}{delta:>7.4f} {sign}{pct:>6.2f}%")
    print(f"{'=' * 72}")


def _print_comparison_table(resultados: dict):
    configs_cmp = ["so_GPR", "so_VIX", "GPR_e_VIX"]
    available   = [c for c in configs_cmp if c in resultados]
    if len(available) < 2:
        return

    lookup = {}
    for cfg in available:
        df = resultados[cfg]
        lookup[cfg] = {row["model"]: row for _, row in df.iterrows()}

    all_models = []
    for cfg in available:
        for m in lookup[cfg]:
            if m not in all_models:
                all_models.append(m)

    metrics = ["RMSE", "MAE", "R2"]
    col_w   = 9
    ncols   = len(available) * len(metrics)
    width   = 30 + ncols * (col_w + 1) + 4
    cfg_labels = {"so_GPR": "GPR", "so_VIX": "VIX", "GPR_e_VIX": "GPR+VIX"}

    print(f"\n{'=' * width}")
    print(f"  Comparacao so_GPR vs so_VIX vs GPR_e_VIX")
    print(f"{'=' * width}")

    grp_hdr = f"  {'':30}"
    for cfg in available:
        span = len(metrics) * (col_w + 1)
        grp_hdr += f" {cfg_labels.get(cfg, cfg):^{span-1}}"
    print(grp_hdr)

    col_hdr = f"  {'Modelo':<30}"
    for cfg in available:
        for met in metrics:
            col_hdr += f" {met:>{col_w}}"
    print(col_hdr)
    print(f"  {'-' * (width - 2)}")

    def fmtv(v):
        if isinstance(v, float) and (v != v):
            return f"{'nan':>{col_w}}"
        return f"{v:>{col_w}.4f}"

    for mname in all_models:
        line = f"  {mname:<30}"
        for cfg in available:
            row = lookup[cfg].get(mname, {})
            for met in metrics:
                v = row.get(met, float("nan")) if isinstance(row, dict) else row.get(met, float("nan"))
                line += " " + fmtv(v)
        print(line)
    print(f"{'=' * width}")


def _run_horse_race(train_df, test_df, feature_cols, label, out_dir,
                    horizon: int = 5):
    """Runs the benchmarks and the models in one configuration and returns the results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_cols].values
    y_train = train_df["target_t1"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target_t1"].values

    results     = []
    importances = {}
    predictions = {}

    # Purged CV folds for tuning
    cv_folds = None
    if TUNE_HYPERPARAMS and "target_date" in train_df.columns:
        try:
            cv_folds = purged_date_cv(
                train_df, n_splits=CV_SPLITS,
                date_col="date", target_date_col="target_date"
            )
        except Exception as e:
            print(f"      [AVISO] purged_date_cv falhou ({e}), usando CV=3")

    # -------------------------------------------------------------------------
    # Benchmarks
    # Benchmarks
    naive_pred = test_df["naive_forecast"].values
    naive_rmse = float(np.sqrt(mean_squared_error(y_test, naive_pred)))
    results.append(evaluate(y_test, naive_pred, "Naive (Random Walk)", naive_rmse=naive_rmse))
    predictions["Naive (Random Walk)"] = naive_pred

    # CRR benchmark
    crr_col = "crr_forecast"
    if crr_col in test_df.columns:
        crr_pred = test_df[crr_col].values
        crr_mask = ~np.isnan(crr_pred)
        if crr_mask.sum() > 10:
            results.append(evaluate(
                y_test[crr_mask], crr_pred[crr_mask],
                "CRR binomial (roll-down)",
                naive_rmse=float(np.sqrt(mean_squared_error(y_test[crr_mask], naive_pred[crr_mask]))),
            ))
            predictions["CRR binomial (roll-down)"] = crr_pred

    # -------------------------------------------------------------------------
    # Tabular models (price level)
    # -------------------------------------------------------------------------
    param_grids = get_param_grids()
    if TUNE_HYPERPARAMS:
        modo = (f"{N_ITER_SEARCH} x n_hiperparametros" if SCALE_N_ITER
                else f"{N_ITER_SEARCH} fixas")
        print(f"      [HP tuning: {modo} combinacoes x {CV_SPLITS} folds purged]")

    for name, model in build_models().items():
        print(f"      {name:<22}", end="", flush=True)
        try:
            if TUNE_HYPERPARAMS:
                model, n_it = tune_model(name, model, X_train, y_train,
                                         param_grids, cv_folds=cv_folds)
                # OLS has no grid
                print(f"({n_it} comb.) " if n_it else "(sem tuning) ", end="", flush=True)
            else:
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            ev    = evaluate(y_test, preds, name, naive_rmse=naive_rmse)
            results.append(ev)
            predictions[name] = preds
            imp = get_feature_importance(model, feature_cols)
            if imp is not None:
                importances[name] = imp
            print("OK")
        except Exception as exc:
            print(f"ERRO ({exc})")

    # -------------------------------------------------------------------------
    # ElasticNet on the price change (delta model)
    # -------------------------------------------------------------------------
    print(f"      {'ElasticNet (delta)':<22}", end="", flush=True)
    try:
        en_model = build_elastic_net_model()
        delta_y_train = y_train - train_df["price"].values
        if TUNE_HYPERPARAMS:
            en_model, en_it = tune_model("ElasticNet", en_model, X_train,
                                         delta_y_train, param_grids,
                                         cv_folds=cv_folds)
            print(f"({en_it} comb.) " if en_it else "(sem tuning) ", end="", flush=True)
        else:
            en_model.fit(X_train, delta_y_train)
        delta_pred = en_model.predict(X_test)
        en_preds   = test_df["price"].values + delta_pred
        ev_en      = evaluate(y_test, en_preds, "ElasticNet (delta)", naive_rmse=naive_rmse)
        results.append(ev_en)
        predictions["ElasticNet (delta)"] = en_preds
        print("OK")
    except Exception as exc:
        print(f"ERRO ({exc})")

    # -------------------------------------------------------------------------
    # Diebold-Mariano by date and breakdown by moneyness
    # -------------------------------------------------------------------------
    # DM vs Naive for all models and for the CRR
    tabular_preds = {
        k: v for k, v in predictions.items()
        if k != "Naive (Random Walk)"
    }
    if tabular_preds:
        _print_dm_table(
            test_df["date"].values, y_test,
            tabular_preds, naive_pred, h=horizon
        )

    if "moneyness" in test_df.columns and tabular_preds:
        print(f"\n  Breakdown por Moneyness (RMSE):")
        _print_moneyness_breakdown(test_df, y_test, tabular_preds, naive_pred)

    # -------------------------------------------------------------------------
    # Evaluation on liquid observations
    # -------------------------------------------------------------------------
    if "volume_log" in test_df.columns:
        liq_mask = test_df["volume_log"].values > 0
        if liq_mask.sum() >= 10:
            y_liq         = y_test[liq_mask]
            naive_liq     = naive_pred[liq_mask]
            liq_naive_rmse = float(np.sqrt(mean_squared_error(y_liq, naive_liq)))
            print(f"\n  Avaliacao em obs liquidas (volume>0): {int(liq_mask.sum())} obs")
            for lname, lpreds in predictions.items():
                if lname in ("Naive (Random Walk)", "CRR binomial (roll-down)"):
                    continue
                p_liq = lpreds[liq_mask]
                valid = ~np.isnan(p_liq)
                if valid.sum() >= 5:
                    liq_rmse = float(np.sqrt(mean_squared_error(y_liq[valid], p_liq[valid])))
                    print(f"    {lname:<30}  RMSE={liq_rmse:.4f}  "
                          f"Theil={liq_rmse/liq_naive_rmse:.4f}")

    # -------------------------------------------------------------------------
    # Saving results
    # -------------------------------------------------------------------------
    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )
    results_df.index += 1

    slug = label.replace(" ", "_").replace("+", "com").replace("-", "sem")
    results_df.to_csv(out_dir / f"results_{slug}.csv", index=True)

    preds_out = test_df[["date", "contract", "target_date", "target_t1",
                          "naive_forecast"]].copy()
    if crr_col in test_df.columns:
        preds_out[crr_col] = test_df[crr_col].values
    for mname, preds in predictions.items():
        col = "pred_" + mname.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if len(preds) == len(preds_out):
            preds_out[col] = preds
    preds_out.to_csv(out_dir / f"predictions_{slug}.csv", index=False)

    if importances:
        imp_df = pd.DataFrame(importances)
        imp_df.index.name = "feature"
        imp_df.to_csv(out_dir / f"importance_{slug}.csv")

    return results_df, predictions, test_df


# =============================================================================
# SECTION 9b — ROLLING ORIGIN (TEMPORAL ROBUSTNESS)
# =============================================================================

def _prever_origem(train_df, test_df, feature_cols):
    """Predictions of one origin: benchmarks and main set, with tuning on the training set."""
    X_train = train_df[feature_cols].values
    y_train = train_df["target_t1"].values
    X_test  = test_df[feature_cols].values

    preds = {"Naive (Random Walk)": test_df["naive_forecast"].values}
    if "crr_forecast" in test_df.columns:
        preds["CRR binomial (roll-down)"] = test_df["crr_forecast"].values

    cv_folds = purged_date_cv(train_df, n_splits=CV_SPLITS,
                              date_col="date", target_date_col="target_date")
    param_grids = get_param_grids()

    modelos = build_models()
    for nome in ORIGEM_MODELOS:
        modelo, _ = tune_model(nome, modelos[nome], X_train, y_train,
                               param_grids, cv_folds=cv_folds)
        preds[nome] = modelo.predict(X_test)

    en_model = build_elastic_net_model()
    delta_train = y_train - train_df["price"].values
    en_model, _ = tune_model("ElasticNet", en_model, X_train, delta_train,
                             param_grids, cv_folds=cv_folds)
    preds["ElasticNet (delta)"] = test_df["price"].values + en_model.predict(X_test)
    return preds


def _run_origem_movel(df_common, configs, out_dir):
    """
    Rolling origin with an expanding window and monthly test blocks.
    At each origin the training set is the history before the first day of the
    block, without the observations whose target falls inside the block (same
    purge as the holdout).
    """
    import time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meses  = df_common["date"].dt.to_period("M")
    blocos = [m for m in sorted(meses.unique())
              if m >= pd.Period(ORIGEM_PRIMEIRO_MES, "M")]
    print(f"\n  ORIGEM MOVEL: {len(blocos)} blocos mensais "
          f"({', '.join(str(b) for b in blocos)}) | janela expansiva")

    acumulado = {cfg: [] for cfg, _, _ in configs}
    fluxo = []
    for k, mes in enumerate(blocos, 1):
        inicio = time.time()
        for cfg_label, feat_cols, _ in configs:
            feat_ok = [c for c in feat_cols if c in df_common.columns]
            keep_cols = list(dict.fromkeys(
                ["date", "contract", "target_t1", "target_date",
                 "naive_forecast", "price", "moneyness", "volume_log"]
                + (["crr_forecast"] if "crr_forecast" in df_common.columns else [])
                + feat_ok
            ))
            mdf = (
                df_common[[c for c in keep_cols if c in df_common.columns]]
                .dropna(subset=feat_ok + ["target_t1"])
                .sort_values("date")
                .reset_index(drop=True)
            )
            no_bloco   = mdf["date"].dt.to_period("M") == mes
            first_test = mdf.loc[no_bloco, "date"].min()
            test_df  = mdf[no_bloco].copy()
            train_df = mdf[(mdf["date"] < first_test)
                           & (mdf["target_date"] < first_test)].copy()
            n_antes  = int((mdf["date"] < first_test).sum())

            preds = _prever_origem(train_df, test_df, feat_ok)

            out = test_df[["date", "contract", "target_date", "target_t1",
                           "naive_forecast"]].copy()
            if "crr_forecast" in test_df.columns:
                out["crr_forecast"] = test_df["crr_forecast"].values
            for nome, p in preds.items():
                col = "pred_" + nome.lower().replace(" ", "_").replace("(", "").replace(")", "")
                out[col] = p
            out["origem"] = str(mes)
            acumulado[cfg_label].append(out)

            if cfg_label == configs[0][0]:
                fluxo.append(dict(
                    origem=str(mes), primeiro_dia_teste=first_test.date(),
                    datas_treino=train_df["date"].nunique(),
                    obs_treino=len(train_df), obs_purgadas=n_antes - len(train_df),
                    obs_teste=len(test_df), datas_teste=test_df["date"].nunique(),
                ))
        f = fluxo[-1]
        print(f"    [{k}/{len(blocos)}] {mes}: treino {f['obs_treino']} obs "
              f"({f['datas_treino']} datas), purgadas {f['obs_purgadas']}, "
              f"teste {f['obs_teste']} obs ({f['datas_teste']} datas) "
              f"| {(time.time() - inicio) / 60:.1f} min")

    for cfg_label, partes in acumulado.items():
        pd.concat(partes, ignore_index=True).to_csv(
            out_dir / f"predictions_ITA_{cfg_label}.csv", index=False)
    pd.DataFrame(fluxo).to_csv(out_dir.parent / "origens.csv", index=False)
    print(f"\n  [saved] predictions_ITA_<config>.csv + origens.csv")


# =============================================================================
# SECCAO 10 — MAIN
# SECTION 10 — MAIN

def main():
    erros_globais = []

    print("=" * 72)
    print("HORSE RACE -- ITA US (iShares Aerospace & Defense ETF)")
    print(f"Horizonte: {MAIN_HORIZON_DAYS} dias de negociacao (calendario exato)")
    print(f"GPR: {GPR_AVAILABILITY_LAG} dias de atraso de disponibilidade")
    print("Configs: sem_indices | so_GPR | so_VIX | GPR_e_VIX")
    print(f"HP tuning: {'RandomizedSearch' if TUNE_HYPERPARAMS else 'defaults'}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # BLOCK 1 — DATA LOADING
    # =========================================================================

    print("\n[1/6] A carregar dados de opcoes...")
    if not OPTIONS_FILE.exists():
        raise FileNotFoundError(f"Ficheiro nao encontrado: {OPTIONS_FILE}")
    options = load_options_excel(OPTIONS_FILE)
    options = options[options["contract"].str.contains("ITA", case=False)].copy()
    n_ita = options["contract"].nunique()
    if n_ita == 0:
        raise ValueError("Nenhum contrato ITA encontrado.")
    n_obs_brutas = len(options)
    # Observations with bid and ask
    n_obs_midquote = int(options["price"].notna().sum())
    print(f"    ITA US: {n_ita} contratos | {n_obs_brutas} obs | "
          f"{options['date'].min().date()} -> {options['date'].max().date()}")
    print(f"    Com mid-quote valido (bid e ask): {n_obs_midquote} obs")

    # Trading calendar: unique dates with spot available
    master_calendar = sorted(
        options.loc[options["spot_ita_us_equity"].notna(), "date"].unique()
    )
    print(f"    Calendario de trading: {len(master_calendar)} datas")

    print("\n[2/6] A carregar GPR (Caldara & Iacoviello) com lag {0} dias...".format(
        GPR_AVAILABILITY_LAG))
    try:
        gpr = load_gpr(GPR_FILE)
        gpr_feature_cols_raw = [c for c in gpr.columns if c != "date"]
        print(f"    GPR: {len(gpr)} obs | componentes: {gpr_feature_cols_raw}")

        # Merge with availability lag (no backfill)
        options = merge_gpr_with_availability_lag(
            options, gpr, gpr_feature_cols_raw,
            availability_lag_days=GPR_AVAILABILITY_LAG
        )
        # GPR column names after the merge (suffix _pit)
        gpr_feature_cols = [f"{c}_pit" for c in gpr_feature_cols_raw
                            if f"{c}_pit" in options.columns]
        print(f"    GPR features (point-in-time, lag={GPR_AVAILABILITY_LAG}d): {gpr_feature_cols}")
    except Exception as exc:
        msg = f"ERRO ao carregar GPR: {exc}"
        print(f"    {msg}")
        erros_globais.append(msg)
        gpr_feature_cols = []

    start_dt, end_dt = options["date"].min(), options["date"].max()

    # Market series read from file. The three are checked together
    # before any is used, so that problems are reported
    # all at once rather than one per run.
    verificar_dados_mercado(start_dt, end_dt)

    print("\n[3/6] A ler taxa de juro sem risco...")
    rf = get_risk_free_rate(start_dt, end_dt)
    options = options.merge(rf, on="date", how="left")
    options["risk_free_rate"] = options["risk_free_rate"].ffill()
    n_rf = int(options["risk_free_rate"].isna().sum())
    print(f"    media {options['risk_free_rate'].mean():.4f}" +
          (f"  [{n_rf} obs sem valor]" if n_rf else ""))

    print("\n[4/6] A ler VIX...")
    vix = get_vix(start_dt, end_dt)
    options = options.merge(vix, on="date", how="left")
    options["vix"] = options["vix"].ffill()
    n_vix = int(options["vix"].isna().sum())
    print(f"    media {options['vix'].mean():.4f} (decimal)" +
          (f"  [{n_vix} obs sem valor]" if n_vix else ""))

    print("\n[5/6] A ler dividend yield...")
    div_yield = get_dividend_yield(start_dt, end_dt)
    options = pd.merge_asof(options.sort_values("date"), div_yield.sort_values("date"),
                            on="date", direction="backward")
    options["dividend_yield"] = options["dividend_yield"].ffill()
    n_dy = int(options["dividend_yield"].isna().sum())
    print(f"    media {options['dividend_yield'].mean():.4f}" +
          (f"  [{n_dy} obs sem valor]" if n_dy else ""))

    print("\n[6/6] A calcular features e target...")
    options = engineer_features(options)
    df_all  = build_forecast_target(options, master_calendar, horizon=MAIN_HORIZON_DAYS)
    print(f"    {len(df_all)} obs com target | {len(df_all.columns)} colunas")

    # CRR benchmark (needs target_date from build_forecast_target)
    if "dividend_yield" in df_all.columns and "spot" in df_all.columns:
        print("    A calcular CRR roll-down forecast (pode demorar ~1 min)...")
        df_all["crr_forecast"] = df_all.apply(
            lambda row: crr_roll_down_forecast(row) if pd.notna(row.get("spot")) else np.nan,
            axis=1
        )
        crr_ok = df_all["crr_forecast"].notna().sum()
        print(f"    CRR: {crr_ok} previsoes calculadas")

    liquid_mask = liquidity_report(df_all)
    print(f"    Obs liquidas (vol>{MIN_LIQUID_VOL} & OI>{MIN_LIQUID_OI}): "
          f"{int(liquid_mask.sum())}/{len(df_all)}")

    # =========================================================================
    # BLOCK 2 — FEATURE SETS
    # =========================================================================

    base_features = [
        "price", "price_lag1", "price_return_1d",
        "iv", "iv_lag1", "iv_change",
        "spread", "spread_lag1",
        "volume_log", "volume_log_lag1",
        "open_int_log",
        "T", "days_to_maturity",
        "risk_free_rate",
    ]
    vix_features  = ["vix"]
    spot_features = [
        "moneyness", "log_moneyness",
        "spot_return_1d", "spot_ma5", "intrinsic_call", "spot",
        "time_value_t",
        "intrinsic_fwd_approx",
    ]

    def make_feat(extra):
        return [c for c in base_features + extra + spot_features if c in df_all.columns]

    configs = [
        ("sem_indices", make_feat([]),                                "Baseline (sem indices externos)"),
        ("so_GPR",      make_feat(gpr_feature_cols),                  "GPR apenas"),
        ("so_VIX",      make_feat(vix_features),                      "VIX apenas"),
        ("GPR_e_VIX",   make_feat(gpr_feature_cols + vix_features),   "GPR + VIX"),
    ]

    print(f"\n    Feature counts: " + " | ".join(f"{l}={len(f)}" for l, f, _ in configs))

    # GPR vs VIX multicollinearity diagnostic
    gpr_col_for_diag = next((c for c in gpr_feature_cols if c.endswith("gpr_pit")), None)
    if gpr_col_for_diag and "vix" in df_all.columns:
        g_s = df_all[gpr_col_for_diag].dropna()
        v_s = df_all["vix"].dropna()
        idx = g_s.index.intersection(v_s.index)
        if len(idx) >= 30:
            rho = float(np.corrcoef(g_s.loc[idx].values, v_s.loc[idx].values)[0, 1])
            vif = float(1.0 / max(1 - rho**2, 1e-9))
            print(f"\n  -- Diagnostico de Multicolinearidade GPR vs VIX --")
            print(f"    Pearson(GPR, VIX): r = {rho:.4f} | VIF bivariado: {vif:.2f}")

    # Common sample: valid rows in the richest configuration
    richest_feat = make_feat(gpr_feature_cols + vix_features)
    richest_ok   = [c for c in richest_feat if c in df_all.columns]
    all_cols = list(dict.fromkeys(
        richest_ok + ["target_t1", "target_date", "naive_forecast",
                      "date", "contract", "price", "moneyness",
                      "volume_log", "crr_forecast"]
    ))
    df_common = (
        df_all[[c for c in all_cols if c in df_all.columns]]
        .dropna(subset=richest_ok + ["target_t1", "target_date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"\n    Amostra comum ({len(configs)} configs): {len(df_common)} obs | "
          f"{df_common['date'].min().date()} -> {df_common['date'].max().date()}")
    print(f"    Datas unicas: {df_common['date'].nunique()}")

    if ORIGEM_MOVEL:
        out_om = OUTPUT_DIR / "ITA"
        _run_origem_movel(df_common, configs, out_om)
        df_common.to_csv(out_om / "dataset_ITA_common.csv", index=False)
        with open(OUTPUT_DIR / "versions.txt", "w") as f:
            f.write(f"data_execucao: {pd.Timestamp.now()}\n")
            f.write(f"seed: {RANDOM_SEED}\nhorizonte: {MAIN_HORIZON_DAYS}\n")
            f.write(f"origem_movel: janela expansiva, blocos mensais desde {ORIGEM_PRIMEIRO_MES}\n")
            f.write(f"modelos: {ORIGEM_MODELOS + ['ElasticNet (delta)']}\n")
            f.write(f"n_iter_base: {N_ITER_SEARCH}\nn_iter_escalado: {SCALE_N_ITER}\n")
            f.write(f"gpr_lag_dias: {GPR_AVAILABILITY_LAG}\n")
        print(f"\n[OK] Outputs guardados em: {OUTPUT_DIR.resolve()}")
        return

    # =========================================================================
    # BLOCK 3 — HORSE RACES
    # =========================================================================

    print("\n" + "=" * 72)
    print("  HORSE RACE: ITA US")
    print("=" * 72)

    out_ita = OUTPUT_DIR / "ITA"
    out_ita.mkdir(parents=True, exist_ok=True)

    resultados  = {}
    predicoes   = {}   # cfg -> {model: test-set predictions}
    testes      = {}   # cfg -> test_df used (to align DM across configs)
    flow_counts = {}   # cfg -> (n_total, n_train, n_purged, n_test)

    for cfg_label, feat_cols, cfg_descr in configs:
        feat_ok = [c for c in feat_cols if c in df_common.columns]

        keep_cols = list(dict.fromkeys(
            ["date", "contract", "target_t1", "target_date",
             "naive_forecast", "price", "moneyness", "volume_log"]
            + (["crr_forecast"] if "crr_forecast" in df_common.columns else [])
            + feat_ok
        ))
        mdf = (
            df_common[[c for c in keep_cols if c in df_common.columns]]
            .dropna(subset=feat_ok + ["target_t1"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(mdf) < 50:
            msg = f"Config '{cfg_label}': dados insuficientes ({len(mdf)} obs). Ignorado."
            print(f"\n  AVISO: {msg}")
            erros_globais.append(msg)
            continue

        train_df, test_df, first_test = purged_holdout_split(
            mdf, train_ratio=TRAIN_RATIO
        )
        n_purged = len(mdf) - len(train_df) - len(test_df)

        print(f"\n  [{cfg_label}]  {cfg_descr}")
        print(f"  [{cfg_label}]  Treino: {len(train_df)} obs "
              f"({train_df['date'].min().date()} -> {train_df['date'].max().date()}) "
              f"| {train_df['date'].nunique()} datas")
        print(f"  [{cfg_label}]  Purgadas na fronteira (alvo em periodo de teste): {n_purged} obs")
        print(f"  [{cfg_label}]  Teste:  {len(test_df)} obs "
              f"({test_df['date'].min().date()} -> {test_df['date'].max().date()}) "
              f"| {test_df['date'].nunique()} datas")
        print(f"  [{cfg_label}]  {len(feat_ok)} features")

        res, preds_cfg, test_used = _run_horse_race(
            train_df, test_df, feat_ok,
            f"ITA_{cfg_label}", out_ita,
            horizon=MAIN_HORIZON_DAYS
        )
        resultados[cfg_label]  = res
        predicoes[cfg_label]   = preds_cfg
        testes[cfg_label]      = test_used
        flow_counts[cfg_label] = (len(mdf), len(train_df), n_purged, len(test_df))
        _print_results_table(res, f"ITA | {cfg_label}  [{cfg_descr}]  [{MAIN_HORIZON_DAYS}d]")

    # Save the full dataset
    df_common.to_csv(out_ita / "dataset_ITA_common.csv", index=False)
    print(f"\n  [saved] dataset_ITA_common.csv ({len(df_common)} obs)")

    # =========================================================================
    # BLOCK 4 — IMPACT TABLES
    # =========================================================================

    if "GPR_e_VIX" in resultados and "sem_indices" in resultados:
        _print_gpr_impact(resultados["GPR_e_VIX"], resultados["sem_indices"],
                          "GPR+VIX combinado vs baseline")

    if "GPR_e_VIX" in resultados and "so_VIX" in resultados:
        _print_gpr_impact(resultados["GPR_e_VIX"], resultados["so_VIX"],
                          "GPR marginal (GPR_e_VIX vs so_VIX)")

    if "GPR_e_VIX" in resultados and "so_GPR" in resultados:
        _print_gpr_impact(resultados["GPR_e_VIX"], resultados["so_GPR"],
                          "VIX marginal (GPR_e_VIX vs so_GPR)")

    _print_comparison_table(resultados)

    # =========================================================================
    # BLOCK 4b — CENTRAL TEST: DM BETWEEN CONFIGURATIONS
    # =========================================================================
    # Compares the predictions of two configs. Negative DM favours config A.

    def _dm_cross_config(cfg_a, cfg_b, titulo):
        if cfg_a not in predicoes or cfg_b not in predicoes:
            return
        ta, tb = testes[cfg_a], testes[cfg_b]
        same = (
            len(ta) == len(tb)
            and (ta["date"].values == tb["date"].values).all()
            and (ta["contract"].values == tb["contract"].values).all()
        )
        if not same:
            print(f"\n  [AVISO] Amostras de teste nao coincidem ({cfg_a} vs {cfg_b}); DM ignorado.")
            return

        y_t   = ta["target_t1"].values
        dts_t = ta["date"].values
        excluir = {"Naive (Random Walk)", "CRR binomial (roll-down)"}
        common_models = [m for m in predicoes[cfg_a]
                         if m in predicoes[cfg_b] and m not in excluir]

        print(f"\n{'=' * 78}")
        print(f"  TESTE CENTRAL — DM por data: {titulo}")
        print(f"  A = {cfg_a} | B = {cfg_b} | DM_stat < 0 favorece A (com GPR)")
        print(f"{'=' * 78}")
        print(f"  {'Modelo':<28} {'DM_stat':>10} {'N_datas':>8} {'p_bruto':>10} {'p_holm':>10}  Sig")
        print(f"  {'-' * 74}")

        rows = []
        for m in common_models:
            r = diebold_mariano_by_date(
                dts_t, y_t, predicoes[cfg_a][m], predicoes[cfg_b][m],
                horizon=MAIN_HORIZON_DAYS
            )
            if not np.isnan(r["statistic"]):
                rows.append((m, r["statistic"], r["n_dates"], r["p_value"]))
        if not rows:
            print("  [sem resultados validos]")
            return
        p_adj = holm_adjust([r[3] for r in rows])
        for (m, stat, nd, p_raw), p_h in zip(rows, p_adj):
            stars = "***" if p_h < 0.01 else "**" if p_h < 0.05 else "*" if p_h < 0.10 else ""
            print(f"  {m:<28} {stat:>10.4f} {nd:>8} {p_raw:>10.4f} {p_h:>10.4f}  {stars}")
        print(f"{'=' * 78}")

    _dm_cross_config("GPR_e_VIX", "so_VIX",
                     "GPR marginal controlando para VIX (pergunta de investigacao)")
    _dm_cross_config("so_GPR", "sem_indices",
                     "GPR vs baseline sem indices")

    # =========================================================================
    # BLOCK 4c — REPRODUCIBILITY
    # =========================================================================

    import sklearn as _sk
    versions_txt = OUTPUT_DIR / "versions.txt"
    with open(versions_txt, "w") as f:
        f.write(f"data_execucao: {pd.Timestamp.now()}\n")
        f.write(f"python: {__import__('sys').version.split()[0]}\n")
        f.write(f"numpy: {np.__version__}\npandas: {pd.__version__}\n")
        f.write(f"scikit-learn: {_sk.__version__}\nxgboost: {xgb.__version__}\n")
        f.write(f"lightgbm: {lgb.__version__}\n")
        f.write(f"catboost: {__import__('catboost').__version__}\n")
        f.write(f"yfinance: {yf.__version__}\n")
        f.write(f"seed: {RANDOM_SEED}\nhorizonte: {MAIN_HORIZON_DAYS}\n")
        f.write(f"n_iter_base: {N_ITER_SEARCH}\nn_iter_escalado: {SCALE_N_ITER}\n")
        f.write(f"offline_only: {OFFLINE_ONLY}\n")
        for _f in (VIX_FILE, RF_FILE, DIV_YIELD_FILE):
            f.write(f"ficheiro: {_f.name}\n")
        f.write(f"gpr_lag_dias: {GPR_AVAILABILITY_LAG}\ntrain_ratio: {TRAIN_RATIO}\n")

    flow_rows = [
        ("contratos_ita", n_ita),
        ("obs_brutas_ita", n_obs_brutas),
        ("obs_com_midquote_valido", n_obs_midquote),
        ("obs_com_target_exato_5d", len(df_all)),
        ("amostra_comum_4configs", len(df_common)),
        ("datas_amostra_comum", int(df_common["date"].nunique())),
    ]
    for cfg_label, (n_t, n_tr, n_pu, n_te) in flow_counts.items():
        flow_rows.append((f"{cfg_label}_treino", n_tr))
        flow_rows.append((f"{cfg_label}_purgadas", n_pu))
        flow_rows.append((f"{cfg_label}_teste", n_te))
    pd.DataFrame(flow_rows, columns=["etapa", "n_obs"]).to_csv(
        OUTPUT_DIR / "sample_flow.csv", index=False
    )
    print(f"\n  [saved] versions.txt + sample_flow.csv (reproducao)")

    # =========================================================================
    # BLOCK 5 — FINAL SUMMARY
    # =========================================================================

    print(f"\n{'=' * 72}")
    print("  SUMARIO FINAL — MELHOR MODELO POR CONFIGURACAO")
    print(f"{'=' * 72}")
    print(f"  {'Config':<16} {'N feat':>6} {'Melhor modelo':<30} {'RMSE':>10} {'R2':>10}")
    print(f"  {'-' * 74}")
    feat_counts = {l: len(f) for l, f, _ in configs}
    for cfg_label, _, _ in configs:
        if cfg_label not in resultados:
            continue
        best = resultados[cfg_label].iloc[0]
        n    = feat_counts.get(cfg_label, "?")
        print(f"  {cfg_label:<16} {n:>6} {best['model']:<30} "
              f"{best['RMSE']:>10.4f} {best['R2']:>10.4f}")
    print(f"{'=' * 72}")

    # =========================================================================
    # BLOCK 6 — ERRORS AND WARNINGS
    # =========================================================================

    if erros_globais:
        print(f"\n{'=' * 72}")
        print(f"  AVISOS / ERROS ({len(erros_globais)} total)")
        print(f"{'=' * 72}")
        for i, msg in enumerate(erros_globais, 1):
            print(f"  [{i}] {msg}")
    else:
        print("\n  [OK] Execucao sem erros ou avisos.")

    print(f"\n[OK] Outputs guardados em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Horse race ITA US. Sem opcoes corre a especificacao principal."
    )
    parser.add_argument("--lag0", action="store_true",
                        help="GPR contemporaneo, sem atraso de disponibilidade (analise ex post)")
    parser.add_argument("--escalado", action="store_true",
                        help="orcamento de busca = 20 x numero de hiperparametros de cada modelo")
    parser.add_argument("--origem-movel", action="store_true",
                        help="robustez temporal: janela expansiva, blocos mensais de teste, "
                             "conjunto principal (OLS, Ridge, EN, XGBoost)")
    args = parser.parse_args()

    # Each variant writes to its own folder, so the primary one is not overwritten
    sufixo = ""
    if args.origem_movel:
        ORIGEM_MOVEL = True
        sufixo += "_origem_movel"
    if args.lag0:
        GPR_AVAILABILITY_LAG = 0
        sufixo += "_lag0"
    if args.escalado:
        SCALE_N_ITER = True
        sufixo += "_escalado"
    OUTPUT_DIR = PROJECT_DIR / f"output_horse_race{sufixo}"

    main()
