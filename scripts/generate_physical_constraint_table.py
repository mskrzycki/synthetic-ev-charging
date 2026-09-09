"""Generate physical-constraint violation rates for real and synthetic samples.

The input samples are first passed through the same broad post-processing filter
used in the main evaluation: positive connection time, positive delivered
energy, connection time <= 120 h, and delivered energy <= 150 kWh. The reported
100 kWh and average-power screens are then applied to those retained samples.

Output:
- table11_physical_constraint_violations.csv
"""

from __future__ import annotations

import pandas as pd

from revision_analysis_common import (
    BASE_DIR,
    LOCATIONS,
    MODEL_SPECS,
    SEEDS,
    evaluation_filter,
    load_real_eval,
    load_synthetic,
    mean_sd,
)


def constraint_rates(df: pd.DataFrame) -> dict[str, float]:
    power = df["energy_session"] / df["connection_time"]
    checks = {
        "Non-positive connection time": df["connection_time"] <= 0,
        "Connection time > 120 h": df["connection_time"] > 120,
        "Non-positive energy": df["energy_session"] <= 0,
        "Energy > 100 kWh": df["energy_session"] > 100,
        "Average power > 11 kW": power > 11,
        "Average power > 22 kW": power > 22,
    }
    broad = (
        checks["Non-positive connection time"]
        | checks["Connection time > 120 h"]
        | checks["Non-positive energy"]
        | checks["Energy > 100 kWh"]
        | checks["Average power > 22 kW"]
    )
    checks["Any broad violation"] = broad
    return {name: 100.0 * float(mask.mean()) for name, mask in checks.items()}


def main() -> None:
    # Keep the common filter explicit here so the script documents the same
    # post-processing used for the manuscript tables.
    real = evaluation_filter(load_real_eval())
    real_rates = constraint_rates(real)

    synthetic_rates: dict[str, dict[str, list[float]]] = {
        model: {name: [] for name in real_rates} for model in MODEL_SPECS
    }
    for model in MODEL_SPECS:
        for location in LOCATIONS:
            for seed in SEEDS:
                synthetic = load_synthetic(model, location, seed)
                rates = constraint_rates(synthetic)
                for name, value in rates.items():
                    synthetic_rates[model][name].append(value)

    rows = []
    for name, real_value in real_rates.items():
        row = {"Constraint": name, "Real held-out (%)": f"{real_value:.3f}"}
        for model in MODEL_SPECS:
            label = MODEL_SPECS[model]["label"]
            row[f"{label} (%)"] = mean_sd(synthetic_rates[model][name]).replace("±", "$\\pm$")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(BASE_DIR / "table11_physical_constraint_violations.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

