"""
Evaluation of the horse-race predictions (metrics and tests of Section 3.7).

Re-estimates nothing: reads the files written by horse_race_forecasting_fixed.py
in output_horse_race/ITA/ (predictions_*.csv, dataset_ITA_common.csv and
importance_ITA_GPR_e_VIX.csv) and computes:
    RMSE, MAE, out-of-sample R2 relative to the random walk and Theil's U
    (full test sample and liquidity subsample: volume > 0, open interest > 0
    and relative spread <= 50%);
    Diebold-Mariano tests by date with the HLN correction, under squared and
    absolute loss, with Holm adjustment within the families of Section 3.7.2
    (J = 5 against the random walk, J = 4 between configurations, J = 8 in the
    extended set);
    minimum detectable effect, loss differential by date, errors by
    moneyness and importance of the GPR block;
    mean bias and decomposition of the mean squared error into bias and variance;
    comparison between training (after the purge) and test;
    central test excluding 1 to 10 June 2026;
    contemporaneous GPR specification (if output_horse_race_lag0/ exists);
    scaled search budget (if output_horse_race_escalado/ exists).

Run after the main script (and the --lag0 and --escalado variants):
    python analise_resultados.py
"""
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import t as student_t

# Outputs of the main script, in the same folder as this file
BASE = str(Path(__file__).resolve().parent / "output_horse_race" / "ITA") + "/"
CFGS = ["sem_indices", "so_GPR", "so_VIX", "GPR_e_VIX"]
MODELS = {"pred_naive_random_walk": "RW", "pred_crr_binomial_roll-down": "CRR",
          "pred_ols": "OLS", "pred_ridge": "Ridge", "pred_elasticnet_delta": "EN-delta",
          "pred_xgboost": "XGBoost", "pred_randomforest": "RF", "pred_lightgbm": "LightGBM",
          "pred_catboost": "CatBoost", "pred_mlp": "MLP"}
MAIN = ["OLS", "Ridge", "EN-delta", "XGBoost"]
ROB = ["RF", "LightGBM", "CatBoost", "MLP"]
ML = MAIN + ROB

def dm(dates, y, a, b, h=5, loss="sq"):
    d = (y - a) ** 2 - (y - b) ** 2 if loss == "sq" else np.abs(y - a) - np.abs(y - b)
    s = pd.DataFrame({"date": dates, "d": d}).groupby("date")["d"].mean().to_numpy()
    n = len(s); m = s.mean(); c = s - m
    lrv = np.mean(c ** 2) + 2 * sum(np.mean(c[k:] * c[:-k]) for k in range(1, h))
    se = np.sqrt(lrv / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = m / se * corr
    p = 2 * student_t.sf(abs(stat), df=n - 1)
    return stat, p, n, m, se / corr, s  # effective se (HLN) = se/corr

def holm(p):
    p = np.asarray(p); o = np.argsort(p); J = len(p)
    adj = np.minimum(np.maximum.accumulate(p[o] * (J - np.arange(J))), 1)
    out = np.empty(J); out[o] = adj; return out

P = {}
for c in CFGS:
    df = pd.read_csv(BASE + f"predictions_ITA_{c}.csv", parse_dates=["date", "target_date"])
    df = df.rename(columns=MODELS)
    P[c] = df
ds = pd.read_csv(BASE + "dataset_ITA_common.csv", parse_dates=["date", "target_date"])
key = ["date", "contract"]
for c in CFGS:
    assert (P[c][key].values == P[CFGS[0]][key].values).all()
test = P["GPR_e_VIX"][key + ["target_t1"]].merge(ds, on=key, how="left", suffixes=("", "_ds"))
vol_pos = test["volume_log"] > 0
oi_pos = test["open_int_log"] > 0
rs = test["spread"] / test["price"]
liq = (vol_pos & oi_pos & (rs <= 0.5)).to_numpy()
y = P["GPR_e_VIX"]["target_t1"].to_numpy()
dates = P["GPR_e_VIX"]["date"].to_numpy()
rw = P["GPR_e_VIX"]["RW"].to_numpy()
T_DATES = pd.Series(dates).nunique()            # number of test dates (T)
FIRST_TEST = pd.Timestamp(pd.Series(dates).min())
print("teste:", len(y), "obs", pd.Series(dates).nunique(), "datas",
      pd.Series(dates).min(), "->", pd.Series(dates).max())
print("contratos no teste:", P["GPR_e_VIX"]["contract"].nunique(),
      "| contratos/data media:", round(len(y) / pd.Series(dates).nunique(), 1))
print("liquido (regra tripla) no teste:", liq.sum(), "obs",
      pd.Series(dates[liq]).nunique(), "datas | vol>0 apenas:", vol_pos.sum())

def metrics(mask, label):
    rows = []
    for c in CFGS:
        df = P[c]
        yy = y[mask]; r = rw[mask]
        sse_rw = np.sum((yy - r) ** 2); rmse_rw = np.sqrt(np.mean((yy - r) ** 2))
        for m in ["RW", "CRR"] + ML:
            pr = df[m].to_numpy()[mask]
            ok = ~np.isnan(pr)
            e = yy[ok] - pr[ok]
            rmse = np.sqrt(np.mean(e ** 2)); mae = np.mean(np.abs(e))
            r2oos = 1 - np.sum(e ** 2) / np.sum((yy[ok] - r[ok]) ** 2)
            rows.append(dict(cfg=c, model=m, n=ok.sum(), RMSE=rmse, MAE=mae,
                             R2_OOS=r2oos, U=rmse / np.sqrt(np.mean((yy[ok] - r[ok]) ** 2))))
    t = pd.DataFrame(rows)
    print(f"\n==== METRICAS [{label}] ====")
    for met in ["RMSE", "MAE", "R2_OOS", "U"]:
        print(f"-- {met}")
        print(t.pivot(index="model", columns="cfg", values=met)[CFGS]
              .reindex(["RW", "CRR"] + ML).round(4).to_string())
    return t

tall = metrics(np.ones(len(y), bool), "amostra completa de teste")
tliq = metrics(liq, "subamostra liquida regra tripla")

def cross(cA, cB, mask, label, models, loss="sq"):
    rows = []
    for m in models:
        a = P[cA][m].to_numpy()[mask]; b = P[cB][m].to_numpy()[mask]
        st, p, n, md, se, s = dm(dates[mask], y[mask], a, b, loss=loss)
        rmseA = np.sqrt(np.mean((y[mask] - a) ** 2)); rmseB = np.sqrt(np.mean((y[mask] - b) ** 2))
        maeA = np.mean(np.abs(y[mask] - a)); maeB = np.mean(np.abs(y[mask] - b))
        rows.append(dict(model=m, RMSE_A=rmseA, RMSE_B=rmseB, dRMSE_pct=100 * (rmseA / rmseB - 1),
                         MAE_A=maeA, MAE_B=maeB, dMAE_pct=100 * (maeA / maeB - 1),
                         DM=st, p=p, n_dates=n, dates_fav_A=int((s < 0).sum()),
                         mean_d=md, se_d=se))
    t = pd.DataFrame(rows)
    t["p_holm_main"] = np.nan
    im = t.model.isin(MAIN)
    t.loc[im, "p_holm_main"] = holm(t.loc[im, "p"])
    t["p_holm_all8"] = holm(t["p"]) if len(t) == 8 else np.nan
    print(f"\n==== {label} | A={cA} vs B={cB} | perda={loss} ====")
    print(t.round(4).to_string(index=False))
    return t

full = np.ones(len(y), bool)
c1 = cross("GPR_e_VIX", "so_VIX", full, "TESTE CENTRAL", ML)
c1a = cross("GPR_e_VIX", "so_VIX", full, "TESTE CENTRAL (MAE)", ML, loss="abs")
c2 = cross("so_GPR", "sem_indices", full, "GPR vs baseline", ML)
c3 = cross("GPR_e_VIX", "so_GPR", full, "VIX marginal dado GPR", ML)
c4 = cross("so_VIX", "sem_indices", full, "VIX vs baseline", ML)
c1l = cross("GPR_e_VIX", "so_VIX", liq, "TESTE CENTRAL liquido", ML)

# DM vs RW, main configs
for c in ["so_VIX", "GPR_e_VIX"]:
    rows = []
    for m in ["CRR"] + ML:
        a = P[c][m].to_numpy(); ok = ~np.isnan(a)
        st, p, n, md, se, s = dm(dates[ok], y[ok], a[ok], rw[ok])
        rows.append(dict(model=m, DM=st, p=p, dates_fav_model=int((s < 0).sum())))
    t = pd.DataFrame(rows); t["p_holm9"] = holm(t.p)
    im = t.model.isin(["CRR"] + MAIN); t.loc[im, "p_holm_main5"] = holm(t.loc[im, "p"])
    print(f"\n==== DM vs RW [{c}] ===="); print(t.round(4).to_string(index=False))

# Power: minimum detectable effect (80% power, 5% two-sided) in the central test
print("\n==== EFEITO MINIMO DETECTAVEL (teste central, perda quadratica) ====")
tc = student_t.ppf(0.975, T_DATES - 1); tp = student_t.ppf(0.80, T_DATES - 1)
for _, r in c1.iterrows():
    mse_b = r.RMSE_B ** 2
    mde = (tc + tp) * r.se_d
    # RMSE reduction corresponding to an MSE reduction = mde
    rmse_red = 100 * (1 - np.sqrt(max(mse_b - mde, 0)) / r.RMSE_B)
    print(f"{r.model:9s} MSE_B={mse_b:7.3f} mean_d={r.mean_d:7.3f} se={r.se_d:6.3f} "
          f"MDE(MSE)={mde:6.3f} ({100*mde/mse_b:5.1f}% do MSE) -> reducao RMSE min ~{rmse_red:5.1f}% "
          f"| observada {-r.dRMSE_pct:5.1f}%")

# Temporal concentration of the gain: loss differential by date (Ridge, EN, OLS, XGB)
print("\n==== DIFERENCIAL POR DATA (GPR_e_VIX - so_VIX), perda quadratica media ====")
tab = {}
for m in MAIN:
    a = P["GPR_e_VIX"][m].to_numpy(); b = P["so_VIX"][m].to_numpy()
    d = (y - a) ** 2 - (y - b) ** 2
    tab[m] = pd.Series(d).groupby(dates).mean()
tab = pd.DataFrame(tab)
tab.insert(0, "var_realizada", pd.Series(y - rw).groupby(dates).mean())   # realised target - current mid-quote
tab.index = pd.to_datetime(tab.index).date
print(tab.round(2).to_string())
tab = tab.drop(columns="var_realizada")
for m in MAIN:
    s = tab[m]; tot = s.sum()
    top3 = s.sort_values().head(3).sum()
    print(f"{m}: soma={tot:.2f} | 3 datas mais favoraveis={top3:.2f} ({100*top3/tot:.0f}% do total)"
          f" | datas favoraveis={int((s<0).sum())}/{T_DATES} | 1a metade={s.iloc[:T_DATES // 2].sum():.2f} 2a metade={s.iloc[T_DATES // 2:].sum():.2f}")

# Moneyness
mon = test["moneyness"].to_numpy()
grp = np.where(mon > 1.05, "ITM", np.where(mon >= 0.95, "ATM", "OTM"))
print("\n==== MONEYNESS (RMSE) ====")
for g in ["ATM", "OTM", "ITM"]:
    mk = grp == g
    line = f"{g} n={mk.sum()} RW={np.sqrt(np.mean((y[mk]-rw[mk])**2)):.3f}"
    for c in ["so_VIX", "GPR_e_VIX"]:
        for m in MAIN:
            line += f" | {c[:6]}-{m}={np.sqrt(np.mean((y[mk]-P[c][m].to_numpy()[mk])**2)):.3f}"
    print(line)

# Training (after the purge) vs test: Table R1
# The training set excludes the purged observations, whose target falls in the test period
tr_df = ds[(ds["date"] < FIRST_TEST) & (ds["target_date"] < FIRST_TEST)]
te_df = test
print("\n==== TREINO (apos purga) vs TESTE ====")

def _descr(d):
    liq_d = (d["volume_log"] > 0) & (d["open_int_log"] > 0) & (d["spread"] / d["price"] <= 0.5)
    por_data = d.groupby("date")
    return {
        "observacoes": len(d),
        "datas": d["date"].nunique(),
        "contratos": d["contract"].nunique(),
        "contratos por data": len(d) / d["date"].nunique(),
        "mid-quote medio": d["price"].mean(),
        "variacao media a 5 dias": (d["target_t1"] - d["price"]).mean(),
        "variacao absoluta media": (d["target_t1"] - d["price"]).abs().mean(),
        "dias ate a maturidade": d["days_to_maturity"].mean(),
        "volatilidade implicita": d["iv"].mean(),
        "VIX (media por data)": por_data["vix"].mean().mean(),
        "VIX min": d["vix"].min(), "VIX max": d["vix"].max(),
        "GPR (media por data)": por_data["gpr_pit"].mean().mean(),
        "GPR min": d["gpr_pit"].min(), "GPR max": d["gpr_pit"].max(),
        "quota sem volume": (d["volume_log"] == 0).mean(),
        "liquidas (regra tripla)": int(liq_d.sum()),
        "quota liquidas": liq_d.mean(),
    }

r1 = pd.DataFrame({"treino": _descr(tr_df), "teste": _descr(te_df)})
print(r1.to_string(float_format=lambda v: f"{v:.3f}"))
spot_teste = te_df.groupby("date")["spot"].first()
print(f"subjacente no teste: {spot_teste.iloc[0]:.2f} -> {spot_teste.iloc[-1]:.2f} "
      f"({100 * (spot_teste.iloc[-1] / spot_teste.iloc[0] - 1):+.1f}%)")
print("corr(GPR_pit, VIX) amostra comum:", ds[["gpr_pit", "vix"]].corr().iloc[0, 1].round(3))

# Variable importance: weight of the GPR block
imp = pd.read_csv(BASE + "importance_ITA_GPR_e_VIX.csv")
g = imp.feature.str.startswith("gpr")
print("\n==== IMPORTANCIA: quota do bloco GPR (GPR_e_VIX) ====")
for col in imp.columns[1:]:
    v = imp[col].abs(); share = v[g].sum() / v.sum()
    rk = v.rank(ascending=False)
    top = imp.loc[v.idxmax(), "feature"]
    best_gpr = imp.loc[v[g].idxmax(), "feature"]
    print(f"{col:12s} quota GPR={100*share:5.1f}% (14 de 37 var = {100*14/37:.0f}%) | top={top} | "
          f"melhor GPR={best_gpr} (rank {int(rk[v[g].idxmax()])}) | VIX rank {int(rk[imp.feature=='vix'].iloc[0])}")
print(imp.set_index("feature").rank(ascending=False).astype(int).head(40).to_string())

# Bias and decomposition of the mean squared error (third diagnostic of Section 3.8)
# MSE = bias^2 + error variance, with error = forecast - realised
print("\n==== VIES MEDIO (previsao - realizado, USD) E QUOTA DO MSE DEVIDA AO VIES ====")
linhas = {}
for m in ["RW", "CRR"] + ML:
    linha = {}
    for c in CFGS:
        e = P[c][m].to_numpy() - y
        ok = ~np.isnan(e)
        vies = e[ok].mean()
        linha[c] = f"{vies:+.2f} ({100 * vies ** 2 / np.mean(e[ok] ** 2):.0f}%)"
    linhas[m] = linha
print(pd.DataFrame(linhas).T[CFGS].to_string())

# Episode of 1 to 10 June 2026: where the central-test differential is concentrated
EPISODIO = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-10"))
fora = ~((pd.to_datetime(dates) >= EPISODIO[0]) & (pd.to_datetime(dates) <= EPISODIO[1]))
print(f"\n==== TESTE CENTRAL SEM AS DATAS DE {EPISODIO[0].date()} A {EPISODIO[1].date()} "
      f"({pd.Series(dates[fora]).nunique()} datas) ====")
for m in MAIN:
    ra = np.sqrt(np.mean((y[fora] - P["GPR_e_VIX"][m].to_numpy()[fora]) ** 2))
    rb = np.sqrt(np.mean((y[fora] - P["so_VIX"][m].to_numpy()[fora]) ** 2))
    print(f"{m:9s} RMSE com GPR={ra:.3f} sem GPR={rb:.3f} dRMSE={100 * (ra / rb - 1):+.1f}%")


def _ler_variante(pasta):
    base = Path(__file__).resolve().parent / pasta / "ITA"
    if not base.exists():
        print(f"\n[{pasta} nao encontrada: correr o script principal com a opcao correspondente]")
        return None
    return base


# Contemporaneous specification (ex post): python horse_race_forecasting_fixed.py --lag0
BASE_L0 = _ler_variante("output_horse_race_lag0")
if BASE_L0 is not None:
    L0 = {c: pd.read_csv(BASE_L0 / f"predictions_ITA_{c}.csv", parse_dates=["date", "target_date"])
             .rename(columns=MODELS) for c in CFGS}
    for c in CFGS:
        assert (L0[c][key].values == P[c][key].values).all(), "amostras de teste diferentes"
    dif_sem_gpr = max(np.nanmax(np.abs(L0[c][m].to_numpy() - P[c][m].to_numpy()))
                      for c in ["sem_indices", "so_VIX"] for m in ML)
    print(f"\n==== GPR CONTEMPORANEO (LAG 0, EX POST) vs DESFASADO (LAG 7) ====")
    # The exact value is floating-point noise (~1e-14) and varies between runs
    print("configuracoes sem GPR identicas nas duas execucoes: "
          + ("sim (diferenca maxima < 1e-10)" if dif_sem_gpr < 1e-10 else f"NAO (diferenca maxima {dif_sem_gpr:.1e})"))
    rmse_ = lambda p: np.sqrt(np.mean((y - p) ** 2))
    rmse_rw = rmse_(rw)
    rows = []
    for m in MAIN:
        b = P["so_VIX"][m].to_numpy(); a7 = P["GPR_e_VIX"][m].to_numpy(); a0 = L0["GPR_e_VIX"][m].to_numpy()
        st0, p0, _, _, _, s0 = dm(dates, y, a0, b)
        st07, p07, _, _, _, _ = dm(dates, y, a0, a7)
        tot = s0.sum(); top3 = np.sort(s0)[:3].sum()
        r_fora = 100 * (np.sqrt(np.mean((y[fora] - a0[fora]) ** 2)) / np.sqrt(np.mean((y[fora] - b[fora]) ** 2)) - 1)
        rows.append(dict(model=m, RMSE_BVIX=rmse_(b), RMSE_lag7=rmse_(a7), RMSE_lag0=rmse_(a0),
                         dRMSE_lag0_pct=100 * (rmse_(a0) / rmse_(b) - 1), p=p0, U_lag0=rmse_(a0) / rmse_rw,
                         DM_lag0_vs_lag7=st07, p_lag0_vs_lag7=p07, datas_fav_GPR=int((s0 < 0).sum()),
                         top3_pct=100 * top3 / tot, dRMSE_fora_episodio_pct=r_fora))
    t0 = pd.DataFrame(rows); t0["p_holm_main"] = holm(t0["p"])
    print(t0.round(3).to_string(index=False))
    print("-- U de Theil e DM contra o random walk com lag 0")
    for c in ["so_GPR", "GPR_e_VIX"]:
        for m in MAIN:
            a = L0[c][m].to_numpy(); st, p, _, _, _, _ = dm(dates, y, a, rw)
            print(f"{c:10s} {m:9s} U={rmse_(a) / rmse_rw:.3f} DM={st:+.2f} p={p:.3f}")

# Scaled search budget: python horse_race_forecasting_fixed.py --escalado
BASE_ESC = _ler_variante("output_horse_race_escalado")
if BASE_ESC is not None:
    print("\n==== ORCAMENTO DE BUSCA: 20 x n_hiperparametros vs 20 candidatos ====")
    res = lambda base, c: pd.read_csv(Path(base) / f"results_ITA_{c}.csv").set_index("model")
    muda = ["ElasticNet (delta)", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "MLP"]
    var = pd.DataFrame({c: 100 * (res(BASE_ESC, c)["RMSE"] / res(BASE, c)["RMSE"] - 1) for c in CFGS})
    print(var.round(2).to_string())
    v = var.loc[muda].abs().to_numpy().ravel()
    print(f"Ridge e OLS sem alteracao por construcao (variacao maxima "
          f"{var.loc[['Ridge', 'OLS']].abs().to_numpy().max():.2f}%)")
    print(f"seis estimadores cuja busca muda ({len(v)} pares): variacao absoluta mediana {np.median(v):.2f}% "
          f"| media {v.mean():.2f}% | maxima {v.max():.1f}%")
    cent = 100 * (res(BASE_ESC, "GPR_e_VIX")["RMSE"] / res(BASE_ESC, "so_VIX")["RMSE"] - 1)
    print("teste central com o orcamento escalado (dRMSE %):")
    print(cent.drop(["Naive (Random Walk)", "CRR binomial (roll-down)"]).round(1).to_string())
    for c in CFGS:
        r = res(BASE_ESC, c).drop(["Naive (Random Walk)", "CRR binomial (roll-down)"])
        print(f"melhor estimador em {c}: {r['Theil_U'].idxmin()} (U = {r['Theil_U'].min():.3f})")
