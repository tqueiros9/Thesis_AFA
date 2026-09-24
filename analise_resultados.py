""" 
Avaliacao das previsoes da horse race (metricas e testes da seccao 3.7).

Nao re-treina nada: le os ficheiros gravados por horse_race_forecasting_fixed.py
em output_horse_race/ITA/ (predictions_*.csv, dataset_ITA_common.csv e
importance_ITA_GPR_e_VIX.csv) e calcula:
    RMSE, MAE, R2 fora da amostra face ao passeio aleatorio e U de Theil
    (amostra de teste completa e subamostra liquida: volume > 0, open
    interest > 0 e spread relativo <= 50%);
    testes Diebold-Mariano por data com correccao HLN, sob perda quadratica e
    absoluta, com ajuste de Holm nas familias da seccao 3.7.2 (J = 5 face ao
    passeio aleatorio, J = 4 entre configuracoes, J = 8 no conjunto alargado);
    efeito minimo detectavel, diferencial de perda por data, erros por
    moneyness e importancia do bloco GPR.

Correr depois do script principal:  python analise_resultados.py
"""
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import t as student_t

# Outputs do script principal, na mesma pasta deste ficheiro
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
    return stat, p, n, m, se / corr, s  # se efectivo (HLN) = se/corr

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
T_DATES = pd.Series(dates).nunique()            # numero de datas de teste (T)
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

# Poder: efeito minimo detectavel (80% poder, 5% bilateral) no teste central
print("\n==== EFEITO MINIMO DETECTAVEL (teste central, perda quadratica) ====")
tc = student_t.ppf(0.975, T_DATES - 1); tp = student_t.ppf(0.80, T_DATES - 1)
for _, r in c1.iterrows():
    mse_b = r.RMSE_B ** 2
    mde = (tc + tp) * r.se_d
    # reducao de RMSE correspondente a uma reducao de MSE = mde
    rmse_red = 100 * (1 - np.sqrt(max(mse_b - mde, 0)) / r.RMSE_B)
    print(f"{r.model:9s} MSE_B={mse_b:7.3f} mean_d={r.mean_d:7.3f} se={r.se_d:6.3f} "
          f"MDE(MSE)={mde:6.3f} ({100*mde/mse_b:5.1f}% do MSE) -> reducao RMSE min ~{rmse_red:5.1f}% "
          f"| observada {-r.dRMSE_pct:5.1f}%")

# Concentracao temporal do ganho: diferencial por data (Ridge, EN, OLS, XGB)
print("\n==== DIFERENCIAL POR DATA (GPR_e_VIX - so_VIX), perda quadratica media ====")
tab = {}
for m in MAIN:
    a = P["GPR_e_VIX"][m].to_numpy(); b = P["so_VIX"][m].to_numpy()
    d = (y - a) ** 2 - (y - b) ** 2
    tab[m] = pd.Series(d).groupby(dates).mean()
tab = pd.DataFrame(tab); tab.index = pd.to_datetime(tab.index).date
print(tab.round(2).to_string())
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

# Deslocamento de regime treino vs teste
tr_mask = ds["date"] < FIRST_TEST
te_mask = ds["date"] >= FIRST_TEST
print("\n==== TREINO vs TESTE (medias na amostra comum) ====")
for v in ["gpr_pit", "gpr_ma30_pit", "vix", "spot", "price", "iv", "days_to_maturity", "moneyness"]:
    print(f"{v:18s} treino={ds.loc[tr_mask, v].mean():9.3f} (min {ds.loc[tr_mask, v].min():8.3f} max {ds.loc[tr_mask, v].max():8.3f})"
          f" teste={ds.loc[te_mask, v].mean():9.3f} (min {ds.loc[te_mask, v].min():8.3f} max {ds.loc[te_mask, v].max():8.3f})")
print("variacao realizada |y - p| media no teste:", np.mean(np.abs(y - rw)).round(3),
      "| mid-quote medio no teste:", rw.mean().round(3), "| mediana:", np.median(rw).round(3))
print("corr(GPR_pit, VIX) amostra comum:", ds[["gpr_pit", "vix"]].corr().iloc[0, 1].round(3))

# Importancia das variaveis: peso do bloco GPR
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
