"""Shared helpers for revised manuscript analyses.

The scripts in this folder reproduce the additional reviewer-response tables
added during manuscript revision. They use the same file naming convention as
the generated synthetic samples in this repository.
"""

from __future__ import annotations

from calendar import month_name
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "diffusion" / "sets" / "train.csv"
VAL_PATH = BASE_DIR / "diffusion" / "sets" / "val.csv"
TEST_PATH = BASE_DIR / "diffusion" / "sets" / "test.csv"

SEEDS = [406, 100, 200, 300, 400]
LOCATIONS = [0, 1, 2, 3]

MODEL_SPECS = {
    "diffusion": {
        "label": "Diffusion",
        "path": lambda loc, seed: BASE_DIR
        / "diffusion"
        / "synthetic"
        / f"synthetic_location_{loc}_best_diffusion_model_seed{seed}.csv",
    },
    "ctgan": {
        "label": "CTGAN",
        "path": lambda loc, seed: BASE_DIR
        / "CTGAN"
        / "synthetic"
        / f"synthetic_year_{loc}_d2_8h_seed{seed}.csv",
    },
    "tabddpm": {
        "label": "TabDDPM",
        "path": lambda loc, seed: BASE_DIR
        / "TabDDPM"
        / "synthetic"
        / f"synthetic_year_{loc}_tabddpm_seed{seed}.csv",
    },
}


def evaluation_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the common post-processing filter used before final evaluation."""

    valid = (
        (df["connection_time"] > 0)
        & (df["energy_session"] > 0)
        & (df["connection_time"] <= 120)
        & (df["energy_session"] <= 150)
    )
    return df[valid].copy()


def add_time_features(df: pd.DataFrame, plugin_time: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["plugin_time"] = plugin_time
    df = df.dropna(subset=["plugin_time"])
    df["plugout_time"] = df["plugin_time"] + pd.to_timedelta(
        df["connection_time"], unit="h"
    )
    df["plugin_hour"] = df["plugin_time"].dt.hour
    df["plugout_hour"] = df["plugout_time"].dt.hour
    df["month"] = df["plugin_time"].dt.month
    df["month_name"] = df["month"].map(lambda x: month_name[int(x)])
    return df


def format_real_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["plugin_day", "plugin_month", "plugin_hour"]).copy()
    plugin_time = pd.to_datetime(
        {
            "year": 2025,
            "month": df["plugin_month"],
            "day": df["plugin_day"],
            "hour": df["plugin_hour"],
        },
        errors="coerce",
    )
    return add_time_features(df, plugin_time)


def format_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["__date__", "plugin_hour"]).copy()
    base_time = pd.to_datetime(df["__date__"]) + pd.to_timedelta(
        df["plugin_hour"], unit="h"
    )
    jitter_minutes = (np.arange(len(df)) * 37 % 61) - 30
    perturb = pd.to_timedelta(jitter_minutes, unit="m")
    return add_time_features(df, base_time + perturb)


def load_real_train() -> pd.DataFrame:
    return format_real_data(evaluation_filter(pd.read_csv(TRAIN_PATH)))


def load_real_eval() -> pd.DataFrame:
    val_df = pd.read_csv(VAL_PATH).drop(columns="strata", errors="ignore")
    test_df = pd.read_csv(TEST_PATH).drop(columns="strata", errors="ignore")
    return format_real_data(evaluation_filter(pd.concat([val_df, test_df], ignore_index=True)))


def load_synthetic(model: str, location: int, seed: int) -> pd.DataFrame:
    path = MODEL_SPECS[model]["path"](location, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    return format_synthetic_data(evaluation_filter(pd.read_csv(path)))


def mean_sd(values: pd.Series | list[float], decimals: int = 3) -> str:
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return "n/a"
    return f"{s.mean():.{decimals}f} ± {s.std():.{decimals}f}"

