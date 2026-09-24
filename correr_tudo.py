"""
Reproduz toda a analise de uma vez e confirma que os resultados coincidem
com os entregues.

    1. guarda uma copia dos outputs entregues e a respectiva avaliacao;
    2. volta a estimar a especificacao principal, a contemporanea (--lag0) e a
       do orcamento escalado (--escalado);
    3. volta a correr a avaliacao e compara-a com a original, e compara as
       previsoes numericamente;
    4. gera as figuras.

As tres estimacoes correm em sequencia, e nao em paralelo: cada uma ja usa
todos os nucleos do processador, e o CatBoost escreve na pasta temporaria do
sistema, que seria partilhada por execucoes simultaneas.

Correr com:  python correr_tudo.py      (demora cerca de 40 minutos)
Os registos de cada passo ficam em logs/.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
LOGS = PASTA / "logs"
VARIANTES = [
    ("principal", [], "output_horse_race"),
    ("lag0", ["--lag0"], "output_horse_race_lag0"),
    ("escalado", ["--escalado"], "output_horse_race_escalado"),
]
TOLERANCIA = 1e-8   # diferencas abaixo disto sao arredondamento de virgula flutuante

AMBIENTE = dict(os.environ, PYTHONIOENCODING="utf-8")


def correr(nome, args, log):
    """Corre um script da pasta, grava o output em logs/ e para se falhar."""
    inicio = time.time()
    print(f"  {nome:<34}", end="", flush=True)
    with open(LOGS / log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, *args], cwd=PASTA, stdout=f,
                           stderr=subprocess.STDOUT, env=AMBIENTE)
    if r.returncode != 0:
        print("ERRO")
        sys.exit(f"\n{nome} falhou. Ver logs/{log}")
    print(f"ok ({(time.time() - inicio) / 60:.1f} min)")


def comparar_previsoes(antiga, nova):
    """Maior diferenca absoluta entre as previsoes antigas e as novas."""
    maior = 0.0
    for f in sorted((antiga / "ITA").glob("predictions_*.csv")):
        a = pd.read_csv(f)
        b = pd.read_csv(nova / "ITA" / f.name)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            return None
        num = a.select_dtypes("number").columns
        maior = max(maior, float(np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy()))))
    return maior


def main():
    LOGS.mkdir(exist_ok=True)
    existentes = [pasta for _, _, pasta in VARIANTES if (PASTA / pasta / "ITA").exists()]
    comparar = len(existentes) == len(VARIANTES)

    print("[1/4] Resultados entregues")
    copia = Path(tempfile.mkdtemp(prefix="outputs_entregues_"))
    if comparar:
        for pasta in existentes:
            shutil.copytree(PASTA / pasta, copia / pasta)
        correr("avaliacao dos resultados entregues", ["analise_resultados.py"], "avaliacao_antes.txt")
    else:
        print("  outputs entregues incompletos: corre tudo, mas sem comparacao final")

    print("[2/4] Estimacao (as tres especificacoes, em sequencia)")
    for nome, opcoes, _ in VARIANTES:
        correr(f"estimacao {nome}", ["horse_race_forecasting_fixed.py", *opcoes], f"estimacao_{nome}.log")

    print("[3/4] Avaliacao e comparacao")
    correr("avaliacao dos novos resultados", ["analise_resultados.py"], "avaliacao_depois.txt")
    if comparar:
        antes = (LOGS / "avaliacao_antes.txt").read_text(encoding="utf-8")
        depois = (LOGS / "avaliacao_depois.txt").read_text(encoding="utf-8")
        iguais = antes == depois
        print(f"  avaliacao (todas as tabelas da tese): {'IDENTICA' if iguais else 'DIFERENTE'}")
        todas_ok = iguais
        for nome, _, pasta in VARIANTES:
            dif = comparar_previsoes(copia / pasta, PASTA / pasta)
            ok = dif is not None and dif < TOLERANCIA
            todas_ok &= ok
            texto = "estrutura diferente" if dif is None else f"diferenca maxima {dif:.1e}"
            print(f"  previsoes {nome:<9}: {texto} -> {'OK' if ok else 'DIFERENTE'}")
        if not iguais:
            print("  diferencas linha a linha: comparar logs/avaliacao_antes.txt com logs/avaliacao_depois.txt")

    print("[4/4] Figuras")
    correr("figuras", ["figuras.py"], "figuras.log")

    shutil.rmtree(copia, ignore_errors=True)
    if comparar:
        print("\nRESULTADO:", "reproducao exacta dos resultados entregues." if todas_ok
              else "ha diferencas face aos resultados entregues (ver acima).")


if __name__ == "__main__":
    main()
