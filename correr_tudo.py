"""
Runs the analysis from this folder, step by step, asking first which steps to run.

Before anything is executed the script asks a yes/no question for each step
(press Enter to accept the default shown in capitals), prints the selection
and asks for a final confirmation. Nothing needs to be edited in the code.

Steps:
    estimation   primary specification, contemporaneous GPR (--lag0), scaled
                 search budget (--escalado) and rolling origin (--origem-movel);
                 each one writes to its own output_horse_race* folder;
    evaluation   analise_resultados.py (the statistics of Chapter 4) and
                 analise_origem_movel.py (Section 4.7.8), from the saved predictions;
    figures      figuras.py.

The estimations run one after the other, not in parallel: each already uses
every processor core, and CatBoost writes to the system's temporary folder,
which simultaneous runs would share.

Run with:  python correr_tudo.py
The log of each step is saved in logs/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
LOGS = FOLDER / "logs"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

# key, question, script, arguments, log file, output folder it writes (or None)
STEPS = [
    ("primary", "Estimate the primary specification (about 6 min)?",
     "horse_race_forecasting_fixed.py", [], "estimacao_principal.log", "output_horse_race"),
    ("lag0", "Estimate the contemporaneous-GPR robustness, --lag0 (about 6 min)?",
     "horse_race_forecasting_fixed.py", ["--lag0"], "estimacao_lag0.log", "output_horse_race_lag0"),
    ("scaled", "Estimate the scaled-search-budget robustness, --escalado (about 20 min)?",
     "horse_race_forecasting_fixed.py", ["--escalado"], "estimacao_escalado.log", "output_horse_race_escalado"),
    ("rolling", "Estimate the rolling-origin robustness, --origem-movel (about 2 min)?",
     "horse_race_forecasting_fixed.py", ["--origem-movel"], "estimacao_origem_movel.log",
     "output_horse_race_origem_movel"),
    ("eval_main", "Evaluate the primary results and the other variants (analise_resultados.py)?",
     "analise_resultados.py", [], "avaliacao_resultados.txt", None),
    ("eval_rolling", "Evaluate the rolling-origin results (analise_origem_movel.py)?",
     "analise_origem_movel.py", [], "avaliacao_origem_movel.txt", None),
    ("figures", "Generate the figures of Chapter 4 (figuras.py)?",
     "figuras.py", [], "figuras.log", None),
]

# the saved outputs that each evaluation step needs in order to run
NEEDS = {
    "eval_main": "output_horse_race",
    "eval_rolling": "output_horse_race_origem_movel",
    "figures": "output_horse_race",
}
NEEDS_KEY = {"output_horse_race": "primary", "output_horse_race_origem_movel": "rolling"}


def ask(question, default=True):
    """Yes/no question; Enter accepts the default. Asks again on any other answer."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            answer = input(f"{question} {hint} ").strip().lower()
        except EOFError:
            sys.exit("\nNo answer available (input closed). Nothing was run.")
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")


def run(label, script, args, log):
    """Runs a script from this folder, saves its output in logs/ and stops on failure."""
    start = time.time()
    print(f"  {label:<60}", end="", flush=True)
    with open(LOGS / log, "w", encoding="utf-8") as f:
        result = subprocess.run([sys.executable, script, *args], cwd=FOLDER, stdout=f,
                                stderr=subprocess.STDOUT, env=ENV)
    if result.returncode != 0:
        print("FAILED")
        sys.exit(f"\n{label} failed. See logs/{log}")
    print(f"done ({(time.time() - start) / 60:.1f} min)")


def main():
    print("Select the steps to run (Enter = yes).\n")
    selected = {}
    for key, question, *_ in STEPS:
        selected[key] = ask(question)

    # an evaluation step needs its predictions, saved earlier or produced in this run
    for key, folder in NEEDS.items():
        if selected[key] and not (FOLDER / folder / "ITA").exists() and not selected[NEEDS_KEY[folder]]:
            print(f"\n  Skipping '{key}': {folder}/ has no saved predictions and its "
                  f"estimation was not selected.")
            selected[key] = False

    chosen = [step for step in STEPS if selected[step[0]]]
    if not chosen:
        sys.exit("\nNothing selected. Nothing was run.")

    print("\nSelected steps:")
    for _, question, script, args, *_ in chosen:
        print(f"  - {script} {' '.join(args)}".rstrip())
    if not ask("\nStart now?"):
        sys.exit("Cancelled. Nothing was run.")

    LOGS.mkdir(exist_ok=True)
    print()
    for _, _, script, args, log, _ in chosen:
        run(f"{script} {' '.join(args)}".rstrip(), script, args, log)
    print("\nFinished. The log of each step is in logs/.")


if __name__ == "__main__":
    main()
