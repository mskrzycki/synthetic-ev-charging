import pandas as pd

df = pd.read_csv("data/EV_Charging_Data_processed.csv")

# Season encoding from preprocess.py: spring=0, summer=1, autumn=2, winter=3
season_map = {3: "Winter", 0: "Spring", 1: "Summer", 2: "Autumn"}
season_order = ["Winter", "Spring", "Summer", "Autumn"]

# Location group encoding: 0->1, 1->2, 2->3, 3->4
df["location_group"] = df["location_group"].astype(int)
df["season_name"] = df["season"].map(season_map)

pivot = (
    df.groupby(["location_group", "season_name"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=season_order, fill_value=0)
)
pivot.index = [f"Location group {i + 1}" for i in pivot.index]
pivot["Total"] = pivot.sum(axis=1)

totals = pivot.sum(axis=0)
totals.name = "Total"
pivot = pd.concat([pivot, totals.to_frame().T])

# --- LaTeX output ---
cols = season_order + ["Total"]
lines = []
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Number of real charging sessions available for each location group and season.}")
lines.append(r"\label{tab:seasonal-counts}")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{5pt}")
lines.append(r"\renewcommand{\arraystretch}{0.9}")
lines.append(r"\begin{tabular}{@{}lccccc@{}}")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Location group} & \textbf{Winter} & \textbf{Spring} & \textbf{Summer} & \textbf{Autumn} & \textbf{Total} \\"
)
lines.append(r"\midrule")

for idx, row in pivot.iterrows():
    if idx == "Total":
        lines.append(r"\midrule")
    vals = " & ".join(str(int(row[c])) for c in cols)
    lines.append(f"{idx} & {vals} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

latex = "\n".join(lines)
print(latex)
