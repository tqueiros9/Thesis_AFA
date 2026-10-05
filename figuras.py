"""
Figures of Chapter 4, from the predictions saved by the main script.

    fig4_1_teste_central_theil.png   Theil's U without and with GPR (central test)
    fig4_2_diferencial_acumulado.png cumulative loss differential by test date
    fig4_3_variacao_realizada.png    mean realised mid-quote change by test date

Run after the main script:  python figuras.py   (writes to figuras/)
"""
from pathlib import Path

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PASTA = Path(__file__).resolve().parent
B = str(PASTA / "output_horse_race" / "ITA") + "/"
OUT = str(PASTA / "figuras") + "/"
import os; os.makedirs(OUT, exist_ok=True)
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300})

P = {c: pd.read_csv(B + f"predictions_ITA_{c}.csv", parse_dates=["date"])
     for c in ["sem_indices", "so_GPR", "so_VIX", "GPR_e_VIX"]}
y = P["so_VIX"]["target_t1"].to_numpy(); rw = P["so_VIX"]["naive_forecast"].to_numpy()
dates = P["so_VIX"]["date"]
rmse = lambda p: np.sqrt(np.mean((y - p) ** 2))
rmse_rw = rmse(rw)
cols = {"OLS": "pred_ols", "Ridge": "pred_ridge", "Elastic Net (Δ)": "pred_elasticnet_delta",
        "XGBoost": "pred_xgboost"}

# ---------- Figure 4.1: dumbbell of the central test (Theil's U) ----------
fig, ax = plt.subplots(figsize=(6.3, 2.9))
names = list(cols)[::-1]
for i, n in enumerate(names):
    ub = rmse(P["so_VIX"][cols[n]]) / rmse_rw
    ua = rmse(P["GPR_e_VIX"][cols[n]]) / rmse_rw
    ax.plot([ub, ua], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(ub, i, s=55, color=S2, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.scatter(ua, i, s=55, color=S1, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.text(ub + 0.012, i + 0.22, f"{ub:.3f}", color=INK2, fontsize=7.5, ha="left")
    ax.text(ua - 0.012, i + 0.22, f"{ua:.3f}", color=INK2, fontsize=7.5, ha="right")
ucrr = rmse(P["so_VIX"]["pred_crr_binomial_roll-down"]) / rmse_rw
ax.axvline(1.0, color=INK2, lw=1, ls="--", zorder=0)
ax.text(1.0, len(names) - 0.35, " Random walk (U = 1)", color=INK2, fontsize=7.5, va="bottom")
ax.axvline(ucrr, color=GRID, lw=1, ls=":", zorder=0)
ax.text(ucrr, -0.75, f" CRR roll-down ({ucrr:.3f})", color=INK2, fontsize=7.5, va="bottom")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, color=INK)
ax.set_xlabel("Theil's U (RMSE model / RMSE random walk); lower is better")
ax.set_xlim(0.97, 1.42); ax.set_ylim(-0.8, len(names) - 0.1)
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
h1 = ax.scatter([], [], s=55, color=S2, label="Controls + VIX")
h2 = ax.scatter([], [], s=55, color=S1, label="Controls + VIX + GPR")
ax.legend(handles=[h1, h2], loc="lower right", frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig4_1_teste_central_theil.png"); plt.close(fig)

# ---------- Figure 4.2: cumulative loss differential by date ----------
fig, ax = plt.subplots(figsize=(6.3, 3.2))
for (n, c), col in zip(cols.items(), [S1, S2, S3, S4]):
    a = P["GPR_e_VIX"][c].to_numpy(); b = P["so_VIX"][c].to_numpy()
    d = pd.Series((y - a) ** 2 - (y - b) ** 2).groupby(dates.values).mean().cumsum()
    ax.plot(d.index, d.values, color=col, lw=2, marker="o", ms=3)
    ax.text(d.index[-1] + pd.Timedelta(days=0.8), d.values[-1], n, color=INK, fontsize=8, va="center")
ax.axhline(0, color=INK2, lw=1)
ax.axvspan(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-10"), color="#f0efec", zorder=0)
ax.text(pd.Timestamp("2026-06-05"), 25, "1–10 June", color=INK2, fontsize=7.5, ha="center")
ax.set_ylabel("Cumulative mean squared-loss\ndifferential (with GPR − without)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_xlim(dates.min() - pd.Timedelta(days=1), dates.max() + pd.Timedelta(days=7))
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.text(dates.min(), -120, "Below zero: the configuration with GPR has accumulated lower loss",
        color=INK2, fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + "fig4_2_diferencial_acumulado.png"); plt.close(fig)

# ---------- Figure 4.3: mean realised 5-day change by test date ----------
ds = pd.read_csv(B + "dataset_ITA_common.csv", parse_dates=["date"])
inicio_teste = dates.min()   # training = dates and targets before the start of the test (after the purge)
tr = ds[(ds.date < inicio_teste) & (pd.to_datetime(ds.target_date) < inicio_teste)]
drift_tr = (tr.target_t1 - tr.price).mean()
chg = pd.Series(y - rw).groupby(dates.values).mean()
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.bar(chg.index, chg.values, width=0.8, color=S1, edgecolor=SURF, linewidth=0.5)
ax.axhline(0, color=INK2, lw=1)
ax.axhline(drift_tr, color=S2, lw=2, ls="--", label=f"Training-period mean change ({drift_tr:.2f})")
ax.axhline((y - rw).mean(), color=INK2, lw=1.2, ls=":", label=f"Test-period mean change (+{(y - rw).mean():.2f})")
ax.legend(loc="upper left", frameon=False, fontsize=7.5)
ax.set_ylabel("Mean 5-day change in\nmid-quote (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT + "fig4_3_variacao_realizada.png"); plt.close(fig)
print("ok", drift_tr)
