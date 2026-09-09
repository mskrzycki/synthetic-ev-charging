"""Generate the revised monthly case-study table.

Outputs:
- monthly_case_study_metrics_runs.csv
- monthly_case_study_metrics_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from revision_analysis_common import (
    BASE_DIR,
    LOCATIONS,
    MODEL_SPECS,
    SEEDS,
    load_real_eval,
    load_synthetic,
)


MONTHS = ["January", "May", "July", "October"]
METRICS = [
    "Average Score",
    "Plug-in Hour TVC",
    "Connection Time KS",
    "Energy Session KS",
]


def categorical_similarity(real: pd.Series, synthetic: pd.Series) -> float:
    real_probs = real.value_counts(normalize=True).sort_index()
    synth_probs = synthetic.value_counts(normalize=True).sort_index()
    cats = sorted(set(real_probs.index) | set(synth_probs.index))
    real_probs = real_probs.reindex(cats, fill_value=0)
    synth_probs = synth_probs.reindex(cats, fill_value=0)
    _ = jensenshannon(real_probs.values, synth_probs.values)
    return float(1.0 - 0.5 * np.abs(real_probs.values - synth_probs.values).sum())


def ks_complement(real: pd.Series, synthetic: pd.Series) -> float:
    return float(1.0 - ks_2samp(real.dropna(), synthetic.dropna()).statistic)


def compute_metrics(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    metrics = {
        "Plug-in Hour TVC": categorical_similarity(real["plugin_hour"], synthetic["plugin_hour"]),
        "Connection Time KS": ks_complement(real["connection_time"], synthetic["connection_time"]),
        "Energy Session KS": ks_complement(real["energy_session"], synthetic["energy_session"]),
    }
    metrics["Average Score"] = float(np.mean(list(metrics.values())))
    return metrics


def main() -> None:
    real = load_real_eval()
    rows = []

    for model in MODEL_SPECS:
        label = MODEL_SPECS[model]["label"]
        for location in LOCATIONS:
            real_location = real[real["location_group"] == location]
            for seed in SEEDS:
                synthetic = load_synthetic(model, location, seed)
                yearly = compute_metrics(real_location, synthetic)
                rows.append(
                    {
                        "model": model,
                        "model_label": label,
                        "location": location,
                        "seed": seed,
                        "period": "Year",
                        **yearly,
                    }
                )

                for month in MONTHS:
                    real_month = real_location[real_location["month_name"] == month]
                    synthetic_month = synthetic[synthetic["month_name"] == month]
                    rows.append(
                        {
                            "model": model,
                            "model_label": label,
                            "location": location,
                            "seed": seed,
                            "period": month,
                            **compute_metrics(real_month, synthetic_month),
                        }
                    )

    runs = pd.DataFrame(rows)
    runs.to_csv(BASE_DIR / "monthly_case_study_metrics_runs.csv", index=False)

    summary_rows = []
    for period in ["Year", *MONTHS]:
        for metric in METRICS:
            row = {"Evaluation period": period, "Metric": metric}
            for model in MODEL_SPECS:
                label = MODEL_SPECS[model]["label"]
                vals = runs[(runs["model"] == model) & (runs["period"] == period)][metric]
                row[label] = f"{vals.mean():.3f} ± {vals.std():.3f}"
                row[f"{label}_n"] = int(vals.notna().sum())
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(BASE_DIR / "monthly_case_study_metrics_summary.csv", index=False)
    print(summary[["Evaluation period", "Metric", "Diffusion", "CTGAN", "TabDDPM"]].to_string(index=False))


if __name__ == "__main__":
    main()

