from __future__ import annotations
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sdv.sampling import Condition
from tqdm import tqdm
from custom_ctgan import CustomCTGAN

BASE = Path("...")
RAW_CSV    = BASE / "data" / "EV_Charging_Data_processed.csv"
MODEL_DIR  = BASE / "CTGAN" / "models"
OUT_DIR    = BASE / "CTGAN" / "synthetic"
YEAR       = 2025
SEED       = 406

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

COND_COLS = ["season", "weekday_group", "location_group"]
OUTPUT_FOCUS = ["plugin_hour", "connection_time", "energy_session"]
OTHER_CATS = ["plugin_day", "plugin_month"]
OTHER_CONT = ["electricity_price", "temperature", "humidity","solar_radiation","wind_speed"]
ALL_COLS = COND_COLS + OUTPUT_FOCUS + OTHER_CATS + OTHER_CONT

# helpers
def weekday_group(idx):
    return ("Mon-Th" if idx < 4 else "Friday" if idx == 4 else "Saturday" if idx == 5 else "Sunday")

def season_idx(date):
    m, d = date.month, date.day
    if (m == 12 and d >= 22) or m in {1, 2} or (m == 3 and d <= 20):
        return 0  # Winter
    if (m == 3 and d >= 21) or m in {4, 5} or (m == 6 and d <= 21):
        return 1  # Spring
    if (m == 6 and d >= 22) or m in {7, 8} or (m == 9 and d <= 22):
        return 2  # Summer
    return 3      # Autumn

def leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

# create buckets for each season and weekday group comb.
def build_calendar(year):
    start = datetime(year, 1, 1)
    total = 366 if leap(year) else 365
    buckets = {}
    for off in range(total):
        dt = start + timedelta(days=off)
        key = (season_idx(dt), weekday_group(dt.weekday()))
        buckets.setdefault(key, []).append(dt)
    return buckets

# generate data for the bucket condition
def sample_bucket(synth: CustomCTGAN, lg: int, seas: int, wg: str, n: int) -> pd.DataFrame:
    if n == 0:
        return pd.DataFrame(columns=ALL_COLS)
    cond = Condition({"location_group": lg, "season": seas, "weekday_group": wg}, num_rows=n) # set the condition
    block = synth.sample_from_conditions([cond]) # sample
    if len(block) < n: # if undersampling - replicate
        reps = math.ceil(n / len(block))
        block = pd.concat([block] * reps, ignore_index=True).iloc[:n]
    return block

# generate a full year of data using the bucket sampling
def generate_for_model(model_path: Path, lg: int, df_real: pd.DataFrame, calendar: Dict[Tuple[int, str], List[datetime]]):
    with model_path.open("rb") as f:
        synth: CustomCTGAN = pickle.load(f)

    tag = model_path.stem.replace("model_", "")
    real_lg = df_real[df_real["location_group"] == lg]
    counts = real_lg.groupby(["season", "weekday_group"]).size()

    frames = []

    # process each bucket
    for (seas, wg), dates in calendar.items():
        total_sessions = counts.get((seas, wg), 0)
        if total_sessions == 0:
            continue
        lam = total_sessions / len(dates)  # historical mean per day

        # Poisson‑random daily counts around that mean 
        daily_ns = np.random.poisson(lam, size=len(dates))

        n_total = daily_ns.sum()
        block = sample_bucket(synth, lg, seas, wg, n_total)

        # distribute across different dates of the year
        offset = 0
        for date, n_day in zip(dates, daily_ns):
            if n_day == 0:
                continue
            chunk = block.iloc[offset : offset + n_day].copy()
            chunk["__date__"]          = date.strftime("%Y-%m-%d")
            chunk["__weekday_group__"] = wg
            chunk["__season__"]        = seas
            chunk["__count__"]         = int(n_day)
            frames.append(chunk)
            offset += n_day
            
    # save output if generated
    if frames:
        out_df = pd.concat(frames, ignore_index=True)
        out_file = OUT_DIR / f"synthetic_year_{lg}_{tag}.csv"
        out_df.to_csv(out_file, index=False)
        print("saved", out_file.relative_to(BASE))


def main():
    # load data
    df_real = pd.read_csv(RAW_CSV, usecols=ALL_COLS)
    np.random.seed(SEED)
    lg_rand = int(np.random.choice(df_real["location_group"].unique())) # random location

    calendar = build_calendar(YEAR) # build a calendar for a year
    models = sorted(MODEL_DIR.glob("model_*.pkl")) # load all models

    for mp in tqdm(models, desc="models"):
        # for every model sample for every year
        generate_for_model(mp, lg_rand, df_real, calendar)

if __name__ == "__main__":
    main()



