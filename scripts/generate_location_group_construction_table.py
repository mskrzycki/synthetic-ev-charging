"""Generate the location-group construction table used in the revision.

The script reads the merged and preprocessed charging-session files and reports
which original residential location labels were retained in each model location
group. It also reports the locations excluded from the final conditional
modelling dataset.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_CSV = REPO_ROOT / "table2_location_group_construction.csv"
OUTPUT_TEX = REPO_ROOT / "table2_location_group_construction.tex"


GROUP_NAMES = {
    0: "Location group 1",
    1: "Location group 2",
    2: "Location group 3",
    3: "Location group 4",
}


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def join_values(values: pd.Series) -> str:
    return ", ".join(sorted(str(v) for v in values.dropna().unique()))


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "EV_Charging_Data.csv")
    processed = pd.read_csv(DATA_DIR / "EV_Charging_Data_processed.csv")

    group_rows = []
    for group_id, group_df in processed.groupby("location_group", sort=True):
        group_rows.append(
            {
                "Model location group": GROUP_NAMES[int(group_id)],
                "Included original location labels": join_values(group_df["location"]),
                "Area type": join_values(group_df["area_type"]),
                "Sessions": len(group_df),
            }
        )

    retained_locations = set(processed["location"].dropna().unique())
    excluded = merged[~merged["location"].isin(retained_locations)].copy()
    excluded_rows = {
        "Model location group": "Excluded from modelling",
        "Included original location labels": join_values(excluded["location"]),
        "Area type": join_values(excluded["area_type"]),
        "Sessions": len(excluded),
    }

    table = pd.DataFrame([*group_rows, excluded_rows])
    table.to_csv(OUTPUT_CSV, index=False)

    latex_rows = []
    for _, row in table.iterrows():
        label = latex_escape(row["Model location group"])
        locations = latex_escape(row["Included original location labels"])
        area_type = latex_escape(row["Area type"])
        sessions = f"{int(row['Sessions']):,}"
        latex_rows.append(f"{label} & {locations} & {area_type} & {sessions} \\\\")

    OUTPUT_TEX.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nSaved {OUTPUT_CSV.name} and {OUTPUT_TEX.name}")


if __name__ == "__main__":
    main()
