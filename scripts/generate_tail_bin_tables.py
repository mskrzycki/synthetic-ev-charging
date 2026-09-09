"""Generate upper-tail bin coverage and frequency tables.

Outputs:
- tail_bin_frequency_by_location_runs.csv
- tail_bin_frequency_by_location_summary.csv
- tail_bin_coverage_by_location_runs.csv
- table12_tail_bin_coverage_by_location.csv
- tail_bins_connection_time.png/pdf
- tail_bins_energy_session.png/pdf
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib")
)

import matplotlib.pyplot as plt
import pandas as pd

from revision_analysis_common import (
    BASE_DIR,
    LOCATIONS,
    MODEL_SPECS,
    SEEDS,
    load_real_eval,
    load_synthetic,
    mean_sd,
)


VARIABLES = {
    "connection_time": "Connection time",
    "energy_session": "Energy session",
}


def bin_masks(series: pd.Series, q90: float, q95: float, q99: float) -> dict[str, pd.Series]:
    return {
        "P90-P95": (series >= q90) & (series < q95),
        "P95-P99": (series >= q95) & (series < q99),
        ">P99": series >= q99,
    }


def main() -> None:
    real = load_real_eval()
    frequency_rows = []
    coverage_rows = []
    table_rows = []

    for location in LOCATIONS:
        real_location = real[real["location_group"] == location]
        for variable, label in VARIABLES.items():
            q90, q95, q99 = real_location[variable].quantile([0.90, 0.95, 0.99])
            tail_range = {
                "P90-P95": f"{q90:.2f}-{q95:.2f}",
                "P95-P99": f"{q95:.2f}-{q99:.2f}",
                ">P99": f">{q99:.2f}",
            }

            by_model_bin: dict[str, dict[str, list[float]]] = {
                model: {b: [] for b in tail_range} for model in MODEL_SPECS
            }
            by_model_covered: dict[str, dict[str, list[int]]] = {
                model: {b: [] for b in tail_range} for model in MODEL_SPECS
            }
            by_model_seed_coverage: dict[str, list[float]] = {
                model: [] for model in MODEL_SPECS
            }

            for model in MODEL_SPECS:
                for seed in SEEDS:
                    synthetic = load_synthetic(model, location, seed)
                    masks = bin_masks(synthetic[variable], q90, q95, q99)
                    seed_covered = 0
                    for bin_name, mask in masks.items():
                        percent = 100.0 * float(mask.mean())
                        covered = int(mask.any())
                        seed_covered += covered
                        frequency_rows.append(
                            {
                                "location_group": location + 1,
                                "variable": variable,
                                "model": MODEL_SPECS[model]["label"],
                                "seed": seed,
                                "bin": bin_name,
                                "percent": percent,
                                "covered": covered,
                                "q90": q90,
                                "q95": q95,
                                "q99": q99,
                            }
                        )
                        coverage_rows.append(
                            {
                                "location_group": location + 1,
                                "variable": variable,
                                "model": MODEL_SPECS[model]["label"],
                                "seed": seed,
                                "bin": bin_name,
                                "covered": covered,
                            }
                        )
                        by_model_bin[model][bin_name].append(percent)
                        by_model_covered[model][bin_name].append(covered)
                    by_model_seed_coverage[model].append(100.0 * seed_covered / len(tail_range))

            row = {
                "Location group": location + 1,
                "Tail variable": label,
                "P90-P95": f"{q90:.2f}-{q95:.2f} h" if variable == "connection_time" else f"{q90:.2f}-{q95:.2f} kWh",
                "P95-P99": f"{q95:.2f}-{q99:.2f} h" if variable == "connection_time" else f"{q95:.2f}-{q99:.2f} kWh",
                "Above P99": f">{q99:.2f} h" if variable == "connection_time" else f">{q99:.2f} kWh",
                "Real covered bins": "3/3",
            }
            for model in MODEL_SPECS:
                label_model = MODEL_SPECS[model]["label"]
                row[f"{label_model} coverage"] = mean_sd(
                    by_model_seed_coverage[model], decimals=1
                ).replace("±", "$\\pm$")
                row[f"{label_model} bins"] = "; ".join(
                    f"{bin_name}:{sum(by_model_covered[model][bin_name])}/{len(by_model_covered[model][bin_name])}"
                    for bin_name in tail_range
                )
            table_rows.append(row)

    freq = pd.DataFrame(frequency_rows)
    freq.to_csv(BASE_DIR / "tail_bin_frequency_by_location_runs.csv", index=False)
    summary = (
        freq.groupby(["location_group", "variable", "model", "bin"], as_index=False)
        .agg(
            mean_percent=("percent", "mean"),
            sd_percent=("percent", "std"),
            q90=("q90", "first"),
            q95=("q95", "first"),
            q99=("q99", "first"),
        )
        .sort_values(["location_group", "variable", "model", "bin"])
    )
    summary.to_csv(BASE_DIR / "tail_bin_frequency_by_location_summary.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(BASE_DIR / "tail_bin_coverage_by_location_runs.csv", index=False)
    pd.DataFrame(table_rows).to_csv(BASE_DIR / "table12_tail_bin_coverage_by_location.csv", index=False)

    for variable, title in VARIABLES.items():
        plot_data = summary[summary["variable"] == variable]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
        for ax, location in zip(axes.ravel(), sorted(plot_data["location_group"].unique())):
            sub = plot_data[plot_data["location_group"] == location]
            pivot = sub.pivot(index="bin", columns="model", values="mean_percent").reindex(
                ["P90-P95", "P95-P99", ">P99"]
            )
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"Location group {location}")
            ax.set_xlabel("")
            ax.set_ylabel("Sessions (%)")
            ax.tick_params(axis="x", rotation=0)
        fig.suptitle(f"Upper-tail frequency: {title}")
        fig.tight_layout()
        stem = "tail_bins_connection_time" if variable == "connection_time" else "tail_bins_energy_session"
        fig.savefig(BASE_DIR / f"{stem}.png", dpi=300)
        fig.savefig(BASE_DIR / f"{stem}.pdf")
        plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
