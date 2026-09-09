"""Generate preprocessing exclusion counts for the reviewer revision.

The counts follow the available preprocessing pipeline that reproduces the
28,093-session modelling dataset used in the manuscript tables.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_CSV = REPO_ROOT / "table_preprocessing_exclusion_counts.csv"
OUTPUT_TEX = REPO_ROOT / "table_preprocessing_exclusion_counts.tex"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "Dataset1_charging_reports.csv", sep=";", quotechar='"')
    for col in ["connection_time", "energy_session"]:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)

    df["plugin_time_dt"] = pd.to_datetime(df["plugin_time"], errors="coerce")
    df["plugout_time_dt"] = pd.to_datetime(df["plugout_time"], errors="coerce")
    df["location"] = df["location"].replace({"BAR_2": "BAR"})

    rows = [{"Step": "Published dataset", "Removed": "-", "Remaining": len(df)}]

    steps = [
        (
            "Missing plug-in or plug-out timestamp",
            lambda x: x["plugin_time_dt"].notna() & x["plugout_time_dt"].notna(),
        ),
        ("Delivered energy <= 0.5 kWh", lambda x: x["energy_session"] > 0.5),
        ("Delivered energy > 150 kWh", lambda x: x["energy_session"] <= 150),
        ("Connection time < 2 h", lambda x: x["connection_time"] >= 2),
        (
            "Excluded low-sample locations",
            lambda x: ~x["location"].isin(["BER", "BOD", "KRO", "OSL_1", "OSL_2"]),
        ),
    ]

    for label, keep_fn in steps:
        keep_mask = keep_fn(df)
        removed = int((~keep_mask).sum())
        df = df[keep_mask].copy()
        rows.append({"Step": label, "Removed": removed, "Remaining": len(df)})

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    latex_rows = []
    for _, row in out.iterrows():
        removed = row["Removed"] if row["Removed"] == "-" else f"{int(row['Removed']):,}"
        remaining = f"{int(row['Remaining']):,}"
        latex_rows.append(f"{row['Step']} & {removed} & {remaining} \\\\")

    OUTPUT_TEX.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nSaved {OUTPUT_CSV.name} and {OUTPUT_TEX.name}")


if __name__ == "__main__":
    main()
