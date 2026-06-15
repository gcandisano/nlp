"""Consolida resultados de todos los experimentos en all_model_results.csv."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import consolidate_results
from src.paths import RESULTS_METRICS


def main():
    df = consolidate_results(output_path=RESULTS_METRICS / "all_model_results.csv")
    if df.empty:
        print("No hay resultados para consolidar. Ejecutar notebooks 03-06 primero.")
        return
    print(f"Consolidados {len(df)} resultados en {RESULTS_METRICS / 'all_model_results.csv'}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
