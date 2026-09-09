"""Generate the nearest-neighbor memorization diagnostic.

Outputs:
- privacy_memorization_diagnostic_runs.csv
- privacy_memorization_diagnostic_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from revision_analysis_common import (
    BASE_DIR,
    LOCATIONS,
    MODEL_SPECS,
    SEEDS,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    evaluation_filter,
    mean_sd,
)


FEATURES = ["plugin_hour", "connection_time", "energy_session"]
GROUPS = ["location_group", "season", "weekday_group"]


def normalized_distance_to_train(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    distances = []
    for _, q_group in query.groupby(GROUPS, dropna=False):
        key = tuple(q_group.iloc[0][col] for col in GROUPS)
        train_group = train
        for col, value in zip(GROUPS, key):
            train_group = train_group[train_group[col] == value]
        if train_group.empty:
            distances.extend([np.nan] * len(q_group))
            continue

        center = train_group[FEATURES].mean()
        scale = train_group[FEATURES].std().replace(0, 1).fillna(1)
        train_x = ((train_group[FEATURES] - center) / scale).to_numpy(float)
        query_x = ((q_group[FEATURES] - center) / scale).to_numpy(float)
        tree = cKDTree(train_x)
        d, _ = tree.query(query_x, k=1)
        distances.extend(d.tolist())
    return np.asarray(distances, dtype=float)


def rounded_duplicate_rate(train: pd.DataFrame, query: pd.DataFrame) -> float:
    train_keys = set(
        zip(
            train["location_group"].astype(int),
            train["season"].astype(int),
            train["weekday_group"].astype(str),
            train["plugin_hour"].round(0).astype(int),
            train["connection_time"].round(3),
            train["energy_session"].round(3),
        )
    )
    query_keys = list(
        zip(
            query["location_group"].astype(int),
            query["season"].astype(int),
            query["weekday_group"].astype(str),
            query["plugin_hour"].round(0).astype(int),
            query["connection_time"].round(3),
            query["energy_session"].round(3),
        )
    )
    return 100.0 * sum(key in train_keys for key in query_keys) / len(query_keys)


def main() -> None:
    train = evaluation_filter(pd.read_csv(TRAIN_PATH))
    val_df = pd.read_csv(VAL_PATH).drop(columns="strata", errors="ignore")
    test_df = pd.read_csv(TEST_PATH).drop(columns="strata", errors="ignore")
    held_out = evaluation_filter(pd.concat([val_df, test_df], ignore_index=True))

    baseline_distances = normalized_distance_to_train(train, held_out)
    baseline_distances = baseline_distances[~np.isnan(baseline_distances)]
    very_close_threshold = float(np.quantile(baseline_distances, 0.01))

    rows = [
        {
            "model": "held_out_real",
            "model_label": "Held-out real",
            "location": "all",
            "seed": "baseline",
            "median_nn_distance": float(np.median(baseline_distances)),
            "very_close_rate_percent": 100.0 * float(np.mean(baseline_distances <= very_close_threshold)),
            "rounded_duplicate_rate_percent": rounded_duplicate_rate(train, held_out),
            "n": int(len(held_out)),
            "very_close_threshold": very_close_threshold,
        }
    ]

    for model in MODEL_SPECS:
        for seed in SEEDS:
            parts = []
            for location in LOCATIONS:
                path = MODEL_SPECS[model]["path"](location, seed)
                parts.append(evaluation_filter(pd.read_csv(path)))
            synthetic = pd.concat(parts, ignore_index=True)
            distances = normalized_distance_to_train(train, synthetic)
            distances = distances[~np.isnan(distances)]
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_SPECS[model]["label"],
                    "location": "all",
                    "seed": seed,
                    "median_nn_distance": float(np.median(distances)),
                    "very_close_rate_percent": 100.0
                    * float(np.mean(distances <= very_close_threshold)),
                    "rounded_duplicate_rate_percent": rounded_duplicate_rate(train, synthetic),
                    "n": int(len(synthetic)),
                    "very_close_threshold": very_close_threshold,
                }
            )

    runs = pd.DataFrame(rows)
    runs.to_csv(BASE_DIR / "privacy_memorization_diagnostic_runs.csv", index=False)

    summary_rows = []
    baseline = runs[runs["model"] == "held_out_real"].iloc[0]
    summary_rows.append(
        {
            "Model": "Held-out real",
            "Median NN distance": f"{baseline['median_nn_distance']:.4f}",
            "Very-close rate (%)": f"{baseline['very_close_rate_percent']:.2f}",
            "Rounded duplicate rate (%)": f"{baseline['rounded_duplicate_rate_percent']:.3f}",
            "n": int(baseline["n"]),
        }
    )
    for model in MODEL_SPECS:
        label = MODEL_SPECS[model]["label"]
        sub = runs[runs["model"] == model]
        summary_rows.append(
            {
                "Model": label,
                "Median NN distance": mean_sd(sub["median_nn_distance"], decimals=4),
                "Very-close rate (%)": mean_sd(sub["very_close_rate_percent"], decimals=2),
                "Rounded duplicate rate (%)": mean_sd(sub["rounded_duplicate_rate_percent"], decimals=3),
                "n": int(round(sub["n"].mean())),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(BASE_DIR / "privacy_memorization_diagnostic_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
