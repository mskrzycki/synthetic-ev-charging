# imports

from __future__ import annotations
import json
import math
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdmetrics.single_column import KSComplement, TVComplement

from custom_ctgan import CustomCTGAN


# paths
BASE  = Path("...")
DATA  = BASE / "data" / "EV_Charging_Data_processed.csv"
SETS  = BASE / "CTGAN" / "sets"
MODELS = BASE / "CTGAN" / "models"
SYNTH  = BASE / "CTGAN" / "synthetic"
BEST_P = BASE / "best_params.json"

for p in (SETS, MODELS, SYNTH):
    p.mkdir(parents=True, exist_ok=True)

SEED          = 406
YEAR_TO_SYNTH = 2025

np.random.seed(SEED) # set seed

COND_COLS = ["season", "weekday_group", "location_group"]
OUTPUT_FOCUS = ["plugin_hour", "connection_time", "energy_session"]
OTHER_CATS = ["plugin_day", "plugin_month"]
OTHER_CONT = ["electricity_price", "temperature", "humidity","solar_radiation","wind_speed"]
ALL_COLS = COND_COLS + OUTPUT_FOCUS + OTHER_CATS + OTHER_CONT


df_full = pd.read_csv(DATA, usecols=ALL_COLS)


# create metadata 

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_full)
for c in ["plugin_hour", "plugin_day", "plugin_month", "season",
          "location_group", "weekday_group"]:
    metadata.update_column(c, sdtype="categorical")

# grid search 

SEARCH_KEYS = ["epochs", "batch_size", "embedding_dim", "generator_decay","discriminator_decay", "lr", "discriminator_steps", "pac"]

GRID = {
    "epochs":              [100, 200, 300],
    "batch_size":          [500, 1000],
    "embedding_dim":       [128, 256],
    "generator_decay":     [1e6, 1e7],
    "discriminator_decay": [1e6, 1e7],
    "lr":                  [1e-3, 1e-4],
    "discriminator_steps": [1, 5],
    "pac":                 [10, 20]}

# best configuration (initial)
BEST_CFG = dict(
    batch_size=500,
    generator_dim=[128, 128],
    discriminator_dim=[128, 128],
    embedding_dim=128,
    generator_decay=1e6,
    discriminator_decay=1e6,
    generator_lr=1e-4,
    discriminator_lr=1e-4,
    discriminator_steps=1,
    pac=10,
    epochs=300,
)

MUTATOR = {
    "epochs":              lambda v, c: c.update(epochs=v),
    "batch_size":          lambda v, c: c.update(batch_size=v),
    "embedding_dim":       lambda v, c: c.update(embedding_dim=v),
    "generator_decay":     lambda v, c: c.update(generator_decay=v),
    "discriminator_decay": lambda v, c: c.update(discriminator_decay=v),
    "lr":                  lambda v, c: c.update(generator_lr=v, discriminator_lr=v),
    "discriminator_steps": lambda v, c: c.update(discriminator_steps=v),
    "pac":                 lambda v, c: c.update(pac=v),
}

SCORE_COLS = ["connection_time", "energy_session", "plugin_hour"]

# use the evaluation on connection time, energy session and plugin_hour
METRIC = {
    "connection_time": KSComplement,
    "energy_session":  KSComplement,
    "plugin_hour":     TVComplement,
}


def make_splits(frame: pd.DataFrame):
    _df = frame.copy()
    _df["strata"] = (_df["location_group"].astype(str) + "_" + _df["season"].astype(str) + "_" + _df["weekday_group"])
    tr_val, _test = train_test_split(_df, test_size=0.15, random_state=SEED, stratify=_df["strata"])
    train, val = train_test_split(tr_val, test_size=0.1765, random_state=SEED, stratify=tr_val["strata"])
    for part in (train, val):
        part.drop(columns="strata", inplace=True)
    return train, val

# set model params, train, sample and evalutate
def eval_cfg(cfg, train_df, val_df):
    model = CustomCTGAN(
        cross_att_heads=8,
        generator_dim=[128, 128],
        discriminator_dim=[128, 128],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        embedding_dim=cfg["embedding_dim"],
        generator_lr=cfg["generator_lr"],
        discriminator_lr=cfg["discriminator_lr"],
        discriminator_steps=cfg["discriminator_steps"],
        pac=cfg["pac"],
        generator_decay=cfg["generator_decay"],
        discriminator_decay=cfg["discriminator_decay"],
        verbose=False,
        metadata=metadata)

    model.fit(train_df)
    synth = model.sample(len(val_df))

    scores = [METRIC[col].compute(val_df[col], synth[col]) for col in SCORE_COLS]
    return float(np.mean(scores))

# try different param combination and keep the best
def run_hyperparam_search():
    train_df, val_df = make_splits(df_full)
    best_overall_score = -math.inf
    cfg = BEST_CFG.copy()

    for key in SEARCH_KEYS:
        best_val, best_score = None, -math.inf
        for candidate in GRID[key]:
            test_cfg = cfg.copy()
            MUTATOR[key](candidate, test_cfg)
            t0 = time.time()
            score = eval_cfg(test_cfg, train_df, val_df)
            print(f"  {candidate!r:>10}: {score:.4f}  ({time.time()-t0:.1f}s)")
            if score > best_score:
                best_val, best_score = candidate, score
        MUTATOR[key](best_val, cfg)
        if best_score > best_overall_score:
            best_overall_score = best_score

    BEST_P.write_text(json.dumps(cfg, indent=2))
    print("\nSaved best params to", BEST_P.relative_to(BASE))
    return cfg

# helpers
def weekday_group(idx: int) -> str:
    return ("Mon-Th" if idx < 4 else "Friday" if idx == 4 else
            "Saturday" if idx == 5 else "Sunday")

def season_idx(dt: datetime) -> int:
    m, d = dt.month, dt.day
    if (m == 12 and d >= 22) or m in {1, 2} or (m == 3 and d <= 20):
        return 0          # Winter
    if (m == 3 and d >= 21) or m in {4, 5} or (m == 6 and d <= 21):
        return 1          # Spring
    if (m == 6 and d >= 22) or m in {7, 8} or (m == 9 and d <= 22):
        return 2          # Summer
    return 3              # Autumn

def leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def build_calendar(year):
    start = datetime(year, 1, 1)
    total = 366 if leap(year) else 365
    buckets: Dict[Tuple[int, str], List[datetime]] = {}
    for off in range(total):
        dt = start + timedelta(days=off)
        key = (season_idx(dt), weekday_group(dt.weekday()))
        buckets.setdefault(key, []).append(dt)
    return buckets

def sample_bucket(synth: CustomCTGAN,lg, seas, wg, n,cols):
    if n == 0:
        return pd.DataFrame(columns=cols)
    cond = Condition( {"location_group": lg, "season": seas, "weekday_group": wg},num_rows=n)
    block = synth.sample_from_conditions([cond])
    return block

def generate_yearly_data(synth: CustomCTGAN, lg, real_df, year,out_dir,tag):
    calendar = build_calendar(year)
    real_lg  = real_df[real_df["location_group"] == lg]
    counts   = real_lg.groupby(["season", "weekday_group"]).size()

    frames = []
    for (seas, wg), dates in calendar.items():
        total_sessions = counts.get((seas, wg), 0)
        if total_sessions == 0:
            continue

        lam = total_sessions / len(dates)        # mean per-day
        daily_ns = np.random.poisson(lam, size=len(dates))
        n_total  = daily_ns.sum()

        block = sample_bucket(
            synth, lg, seas, wg, n_total,
            cols=real_df.columns
        )

        offset = 0
        for dt, n_day in zip(dates, daily_ns):
            if n_day == 0:
                continue
            chunk = block.iloc[offset:offset+n_day].copy()
            chunk["__date__"]          = dt.strftime("%Y-%m-%d")
            chunk["__weekday_group__"] = wg
            chunk["__season__"]        = seas
            chunk["__count__"]         = int(n_day)
            frames.append(chunk)
            offset += n_day

    if frames:
        out_df   = pd.concat(frames, ignore_index=True)
        out_path = out_dir / f"synthetic_year_{lg}_{tag}.csv"
        out_df.to_csv(out_path, index=False)
        print("Saved", out_path.relative_to(BASE))

def main():
    full_df = pd.read_csv(DATA)[ALL_COLS]
    train_df, val_df, _ = make_splits(full_df)

    run_hyperparam_search()

    final = CustomCTGAN(
        cross_att_heads=8,
        generator_dim=[128, 128],
        discriminator_dim=[128, 128],
        epochs=BEST_CFG["epochs"],
        batch_size=BEST_CFG["batch_size"],
        embedding_dim=BEST_CFG["embedding_dim"],
        generator_lr=BEST_CFG["generator_lr"],
        discriminator_lr=BEST_CFG["discriminator_lr"],
        discriminator_steps=BEST_CFG["discriminator_steps"],
        pac=BEST_CFG["pac"],
        generator_decay=BEST_CFG["generator_decay"],
        discriminator_decay=BEST_CFG["discriminator_decay"],
        verbose=True,
        metadata=metadata,
    )
    final.fit(train_df)

    model_path = MODELS / "model_best.pkl"
    with model_path.open("wb") as f:
        pickle.dump(final, f)

    generate_yearly_data(
        final, lg=3, real_df=full_df, year=2025,
        out_dir=BASE / "CTGAN" / "synthetic",
        tag="best")

if __name__ == "__main__":
    main()