"""
Evaluation of the rolling-origin robustness exercise (expanding window,
monthly test blocks).

Re-estimates nothing: reads the predictions written by
    python horse_race_forecasting_fixed.py --origem-movel
in output_horse_race_origem_movel/ITA/ and computes, by block and pooled:
    RMSE and Theil's U relative to the random walk, by model and configuration;
    central test (Baseline+VIX+GPR against Baseline+VIX): change in RMSE,
    Diebold-Mariano by date with the HLN correction, dates favouring GPR,
    Holm in the main set (J = 4);
    auxiliary test (Baseline+GPR against Baseline);
    DM of each model against the random walk (pooled);
    mean bias and price drift (training against test) by block;
    the central test without 1 to 10 June;
    comparison with the primary specification on the common dates.

Run after the estimation:
    python analise_origem_movel.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PASTA = Path(__file__).resolve().parent
BASE = PASTA / "output_horse_race_origem_movel" / "ITA"
PRINCIPAL = PASTA / "output_horse_race" / "ITA"
CFGS = ["sem_indices", "so_GPR", "so_VIX", "GPR_e_VIX"]
NOMES = {"pred_naive_random_walk": "RW", "pred_crr_binomial_roll-down": "CRR",
         "pred_ols": "OLS", "pred_ridge": "Ridge",
         "pred_elasticnet_delta": "EN-delta", "pred_xgboost": "XGBoost"}
MAIN = ["OLS", "Ridge", "EN-delta", "XGBoost"]
H = 5


def dm(dates, y, a, b, loss="sq"):
    d = (y - a) ** 2 - (y - b) ** 2 if loss == "sq" else np.abs(y - a) - np.abs(y - b)
    s = pd.DataFrame({"date": dates, "d": d}).groupby("date")["d"].mean().to_numpy()
    n = len(s)
    if n <= max(10, H + 1):
        return np.nan, np.nan, n, np.nan, s
    m = s.mean(); c = s - m
    lrv = np.mean(c ** 2) + 2 * sum(np.mean(c[k:] * c[:-k]) for k in range(1, H))
    if lrv <= 0:
        return np.nan, np.nan, n, m, s
    corr = np.sqrt((n + 1 - 2 * H + H * (H - 1) / n) / n)
    stat = m / np.sqrt(lrv / n) * corr
    return stat, 2 * student_t.sf(abs(stat), df=n - 1), n, m, s


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); J = len(p)
    adj = np.minimum(np.maximum.accumulate(p[o] * (J - np.arange(J))), 1)
    out = np.empty(J); out[o] = adj; return out


P = {c: pd.read_csv(BASE / f"predictions_ITA_{c}.csv", parse_dates=["date", "target_date"])
          .rename(columns=NOMES) for c in CFGS}
key = ["date", "contract"]
for c in CFGS:
    assert (P[c][key].values == P[CFGS[0]][key].values).all()
ds = pd.read_csv(BASE / "dataset_ITA_common.csv", parse_dates=["date", "target_date"])
orig = pd.read_csv(BASE.parent / "origens.csv")
print("==== ORIGENS ====")
print(orig.to_string(index=False))

ref = P["GPR_e_VIX"]
y = ref["target_t1"].to_numpy(); dates = ref["date"].to_numpy()
rw = ref["RW"].to_numpy(); bloco = ref["origem"].to_numpy()
BLOCOS = list(dict.fromkeys(bloco))
masks = {b: bloco == b for b in BLOCOS}
masks["AGREGADO"] = np.ones(len(y), bool)
print(f"\nteste agregado: {len(y)} obs, {pd.Series(dates).nunique()} datas, "
      f"{pd.Series(dates).min().date()} -> {pd.Series(dates).max().date()}")

# ---------------------------------------------------------------------------
# 1. RMSE and Theil's U by block
# ---------------------------------------------------------------------------
print("\n==== RMSE E U DE THEIL POR BLOCO ====")
for nome, mk in masks.items():
    yy, r = y[mk], rw[mk]
    rmse_rw = np.sqrt(np.mean((yy - r) ** 2))
    rows = []
    for m in ["CRR"] + MAIN:
        row = {"modelo": m}
        for c in CFGS:
            p = P[c][m].to_numpy()[mk]; ok = ~np.isnan(p)
            rmse = np.sqrt(np.mean((yy[ok] - p[ok]) ** 2))
            row[f"{c}_RMSE"] = rmse
            row[f"{c}_U"] = rmse / np.sqrt(np.mean((yy[ok] - r[ok]) ** 2))
        rows.append(row)
    print(f"\n-- {nome}  (n = {mk.sum()}, datas = {pd.Series(dates[mk]).nunique()}, "
          f"RMSE RW = {rmse_rw:.4f})")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Tests between configurations
# ---------------------------------------------------------------------------
def cruzado(cA, cB, mask, titulo, loss="sq"):
    rows = []
    for m in MAIN:
        a = P[cA][m].to_numpy()[mask]; b = P[cB][m].to_numpy()[mask]
        st, p, n, md, s = dm(dates[mask], y[mask], a, b, loss=loss)
        ra = np.sqrt(np.mean((y[mask] - a) ** 2)); rb = np.sqrt(np.mean((y[mask] - b) ** 2))
        rows.append(dict(modelo=m, RMSE_A=ra, RMSE_B=rb, dRMSE_pct=100 * (ra / rb - 1),
                         DM=st, p=p, n_datas=n, datas_fav_A=int((s < 0).sum())))
    t = pd.DataFrame(rows)
    if t["p"].notna().all():
        t["p_holm_J4"] = holm(t["p"])
    print(f"\n-- {titulo}")
    print(t.round(4).to_string(index=False))
    return t


print("\n==== TESTE CENTRAL: GPR_e_VIX (A) contra so_VIX (B); DM<0 favorece o GPR ====")
central = {}
for nome, mk in masks.items():
    central[nome] = cruzado("GPR_e_VIX", "so_VIX", mk, f"{nome} (perda quadratica)")
cruzado("GPR_e_VIX", "so_VIX", masks["AGREGADO"], "AGREGADO (perda absoluta)", loss="abs")

print("\n==== TESTE AUXILIAR: so_GPR (A) contra sem_indices (B) ====")
for nome, mk in masks.items():
    cruzado("so_GPR", "sem_indices", mk, nome)

print("\n==== VIX: so_VIX (A) contra sem_indices (B) ====")
cruzado("so_VIX", "sem_indices", masks["AGREGADO"], "AGREGADO")

# ---------------------------------------------------------------------------
# 3. Each model against the random walk (pooled)
# ---------------------------------------------------------------------------
print("\n==== DM CONTRA O PASSEIO ALEATORIO (agregado; DM<0 favorece o modelo) ====")
for c in CFGS:
    rows = []
    for m in ["CRR"] + MAIN:
        a = P[c][m].to_numpy(); ok = ~np.isnan(a)
        st, p, n, md, s = dm(dates[ok], y[ok], a[ok], rw[ok])
        rows.append(dict(modelo=m, DM=st, p=p, n_datas=n))
    t = pd.DataFrame(rows); t["p_holm_J5"] = holm(t["p"])
    print(f"\n-- {c}"); print(t.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Bias and drift by block
# ---------------------------------------------------------------------------
print("\n==== DERIVA DO PRECO E VIES MEDIO (previsao - realizado) POR BLOCO ====")
ds["delta"] = ds["target_t1"] - ds["price"]
rows = []
for b in BLOCOS:
    mk = masks[b]
    ini = pd.Timestamp(dates[mk].min())
    tr = ds[(ds["date"] < ini) & (ds["target_date"] < ini)]
    row = dict(bloco=b, deriva_treino=tr["delta"].mean(), deriva_teste=(y[mk] - rw[mk]).mean())
    for m in MAIN:
        for c in ["so_VIX", "GPR_e_VIX"]:
            row[f"vies_{m}_{c}"] = (P[c][m].to_numpy()[mk] - y[mk]).mean()   # forecast minus realised
    rows.append(row)
print(pd.DataFrame(rows).round(3).T.to_string(header=False))

# ---------------------------------------------------------------------------
# 4b. Decomposition of the MSE change between configurations: bias^2 and variance
# ---------------------------------------------------------------------------
print("\n==== DECOMPOSICAO: GPR_e_VIX contra so_VIX (MSE = vies^2 + variancia do erro) ====")
rows = []
for nome in BLOCOS + ["AGREGADO"]:
    mk = masks[nome]
    for m in MAIN:
        ea = y[mk] - P["GPR_e_VIX"][m].to_numpy()[mk]
        eb = y[mk] - P["so_VIX"][m].to_numpy()[mk]
        d_mse = np.mean(ea ** 2) - np.mean(eb ** 2)
        d_b2 = ea.mean() ** 2 - eb.mean() ** 2
        d_var = ea.var() - eb.var()
        rows.append(dict(bloco=nome, modelo=m, dMSE=d_mse, d_vies2=d_b2, d_var=d_var,
                         quota_vies_pct=100 * d_b2 / d_mse if d_mse != 0 else np.nan))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Without 1-10 June
# ---------------------------------------------------------------------------
junho = (pd.Series(dates) >= "2026-06-01") & (pd.Series(dates) <= "2026-06-10")
cruzado("GPR_e_VIX", "so_VIX", ~junho.to_numpy(), "AGREGADO sem 1 a 10 de junho")

# ---------------------------------------------------------------------------
# 6. Comparison with the primary specification on the common dates
# ---------------------------------------------------------------------------
if PRINCIPAL.exists():
    print("\n==== ORIGEM MOVEL CONTRA ESPECIFICACAO PRINCIPAL (datas do teste principal) ====")
    rows = []
    for c in ["so_VIX", "GPR_e_VIX"]:
        pm = pd.read_csv(PRINCIPAL / f"predictions_ITA_{c}.csv", parse_dates=["date"])
        pm = pm.rename(columns=NOMES)
        com = ref[key].merge(pm[key], on=key, how="inner")
        mk = ref.set_index(key).index.isin(com.set_index(key).index)
        rm = pm.set_index(key).loc[ref[key][mk].apply(tuple, axis=1)]
        yy = y[mk]
        for m in MAIN:
            r_om = np.sqrt(np.mean((yy - P[c][m].to_numpy()[mk]) ** 2))
            r_pr = np.sqrt(np.mean((yy - rm[m].to_numpy()) ** 2))
            rows.append(dict(cfg=c, modelo=m, n=int(mk.sum()), RMSE_origem_movel=r_om,
                             RMSE_principal=r_pr))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Minimum detectable effect in the pooled central test (as in Section 3.7.3)
# ---------------------------------------------------------------------------
print("\n==== EFEITO MINIMO DETECTAVEL (agregado; 80% poder, 5% bilateral) ====")
for m in MAIN:
    a = P["GPR_e_VIX"][m].to_numpy(); b = P["so_VIX"][m].to_numpy()
    st, p, n, md, s = dm(dates, y, a, b)
    c = s - s.mean()
    lrv = np.mean(c ** 2) + 2 * sum(np.mean(c[k:] * c[:-k]) for k in range(1, H))
    corr = np.sqrt((n + 1 - 2 * H + H * (H - 1) / n) / n)
    se_eff = np.sqrt(lrv / n) / corr
    mde = (student_t.ppf(0.975, n - 1) + student_t.ppf(0.80, n - 1)) * se_eff
    mse_b = np.mean((y - b) ** 2)
    red = 100 * (1 - np.sqrt(max(mse_b - mde, 0)) / np.sqrt(mse_b))
    obs = 100 * (1 - np.sqrt(np.mean((y - a) ** 2)) / np.sqrt(mse_b))
    print(f"{m:9s} T={n} MDE(MSE)={mde:6.3f} -> reducao RMSE minima ~{red:5.1f}% | observada {obs:5.1f}%")
