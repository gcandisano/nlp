"""Script opcional para generar splits procesados sin ejecutar notebooks."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.paths import DATA_PROCESSED, DATA_RAW
from src.preprocessing import (
    add_clean_text_columns,
    class_distribution_report,
    drop_content_duplicates,
    filter_politics_subset,
    parse_dates,
    temporal_split,
)


def main():
    fake_df = pd.read_csv(DATA_RAW / "Fake.csv")
    true_df = pd.read_csv(DATA_RAW / "True.csv")
    fake_df["label"] = 1
    true_df["label"] = 0
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = parse_dates(df)
    df = df.dropna(subset=["parsed_date"]).reset_index(drop=True)
    df, dedup_stats = drop_content_duplicates(df)
    print(
        f"Deduplicación title+text: {dedup_stats['removed']:,} filas eliminadas "
        f"({dedup_stats['rows_before']:,} -> {dedup_stats['rows_after']:,})"
    )
    if dedup_stats["label_conflicts"]:
        print(f"  Grupos duplicados con etiquetas distintas: {dedup_stats['label_conflicts']}")
    df = add_clean_text_columns(df)

    politics_df = filter_politics_subset(df, include_optional=False)

    for dataset, prefix in [(politics_df, "politics"), (df, "full")]:
        train, val, test = temporal_split(dataset)
        for name, split in [("train", train), ("val", val), ("test", test)]:
            path = DATA_PROCESSED / f"{prefix}_{name}.csv"
            split.to_csv(path, index=False)
            print(f"\n{prefix}_{name}: {len(split):,} rows")
            print(class_distribution_report(split))


if __name__ == "__main__":
    main()
