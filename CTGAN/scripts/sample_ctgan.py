from __future__ import annotations
import io
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sdv.sampling import Condition
from tqdm import tqdm
import torch
from custom_ctgan import CustomCTGAN

class CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA tensors to CPU when CUDA is unavailable."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        if module.startswith('torch.cuda'):
            module = module.replace('torch.cuda', 'torch', 1)
        return super().find_class(module, name)

BASE = Path(__file__).resolve().parents[2]
RAW_CSV    = BASE / "data" / "EV_Charging_Data_processed.csv"
MODEL_DIR  = BASE / "CTGAN" / "models"
OUT_DIR    = BASE / "CTGAN" / "synthetic"
YEAR       = 2025
SEED       = 406
SEEDS      = [406, 100, 200, 300, 400]  # 5 seeds for reproducibility runs
BEST_MODEL_TAG = "d2_8h"  # best model identified by evaluation (model_d2_8h.pkl)

OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    n = int(n)
    if n == 0:
        return pd.DataFrame(columns=ALL_COLS)
    cond = Condition({"location_group": lg, "season": seas, "weekday_group": wg}, num_rows=n) # set the condition
    block = synth.sample_from_conditions([cond]) # sample
    if len(block) < n: # if undersampling - replicate
        reps = math.ceil(n / len(block))
        block = pd.concat([block] * reps, ignore_index=True).iloc[:n]
    return block

# generate a full year of data using the bucket sampling
def generate_for_model(model_path: Path, lg: int, seed: int, df_real: pd.DataFrame, calendar: Dict[Tuple[int, str], List[datetime]]):
    with model_path.open("rb") as f:
        synth: CustomCTGAN = CPUUnpickler(f).load()

    # Fix metadata API mismatch: current SDV expects _original_metadata to have a
    # .tables dict, but the pickled model stores a bare SingleTableMetadata.
    # Wrap it in a lightweight proxy so .tables[table_name] resolves correctly.
    meta = synth._original_metadata
    table_name = getattr(synth, '_table_name', 'table')
    if not hasattr(meta, 'tables'):
        class _MetaProxy:
            def __init__(self, single_meta, tname):
                self.tables = {tname: single_meta}
                # forward attribute access to the inner metadata
                self._inner = single_meta
            def __getattr__(self, item):
                return getattr(self._inner, item)
        synth._original_metadata = _MetaProxy(meta, table_name)

    tag = model_path.stem.replace("model_", "")
    tag = f"{tag}_seed{seed}"
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

    best_model = MODEL_DIR / f"model_{BEST_MODEL_TAG}.pkl"
    if not best_model.exists():
        raise FileNotFoundError(f"Best CTGAN model not found: {best_model}")

    calendar = build_calendar(YEAR) # build a calendar for a year
    all_locations = sorted(df_real["location_group"].dropna().astype(int).unique().tolist())

    for seed in tqdm(SEEDS, desc="Seeds"):
        for lg in tqdm(all_locations, desc=f"Locations (seed={seed})", leave=False):
            np.random.seed(seed)
            generate_for_model(best_model, lg, seed, df_real, calendar)

if __name__ == "__main__":
    main()



