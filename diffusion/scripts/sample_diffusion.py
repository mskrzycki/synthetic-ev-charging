import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdt import HyperTransformer
from torch.utils.data import Dataset
from diffusion import DDPM

BASE = Path("...")
RAW_CSV = BASE / "data" / "EV_Charging_Data_processed.csv"
MODEL_DIR = BASE / "diffusion" / "models"
OUT_DIR = BASE / "diffusion" / "synthetic"
YEAR = 2025
SEED = 406
FIXED_LOCATION = 3  # compare for the same location

STEP_REDUCTION = 1000
MAX_PER_BUCKET = 500

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# columns
COND_COLS = ["season", "weekday_group", "location_group"]
OUTPUT_COLS = ["plugin_hour", "connection_time", "energy_session"]
OTHER_COLS = ["plugin_day", "plugin_month", "electricity_price", "temperature", "humidity", "solar_radiation", "wind_speed"]
ALL_COLS = COND_COLS + OUTPUT_COLS + OTHER_COLS

# data set
class FullDataset(Dataset):
    def __init__(self, input_data, cond_data):
        self.input_data = torch.from_numpy(input_data)
        self.cond_data = torch.from_numpy(cond_data)

    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return { "input": self.input_data[idx].unsqueeze(0), "condition": self.cond_data[idx]}

# helpers for yearly sampling
def weekday_group(idx):
    return ("Mon-Th" if idx < 4 else "Friday" if idx == 4 else "Saturday" if idx == 5 else "Sunday")

def season_idx(dt):
    m, d = dt.month, dt.day
    if (m == 12 and d >= 22) or m in {1, 2} or (m == 3 and d <= 20):
        return 0          # Winter
    if (m == 3 and d >= 21) or m in {4, 5} or (m == 6 and d <= 21):
        return 1          # Spring
    if (m == 6 and d >= 22) or m in {7, 8} or (m == 9 and d <= 22):
        return 2          # Summer
    return 3              # Autumn


def leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# build the calendar and the buckets for season x weekday group
def build_calendar(year):
    start = datetime(year, 1, 1)
    length = 366 if leap(year) else 365
    buckets = {}
    for offset in range(length):
        dt = start + timedelta(days=offset)
        key = (season_idx(dt), weekday_group(dt.weekday()))
        buckets.setdefault(key, []).append(dt)
    return buckets

@torch.no_grad()

# generate batches
def sample_batch(model: DDPM, cond_vec, n):
    # conditional vectors
    c = torch.from_numpy(cond_vec).float().to(model.opt.device)
    c = c.view(1, -1).repeat(n, 1)
    
    # noise
    x = torch.randn(n, model.opt.seq_len, model.opt.input_dim, device=model.opt.device)
    
    if hasattr(model.eps_model, "input_projector") and hasattr(model.eps_model.input_projector, "flatten_parameters"):
        model.eps_model.input_projector.flatten_parameters()
    
    # denoising steps
    for step in range(model.opt.n_steps - 1, -1, -1):
        t = torch.full((n,), step, dtype=torch.long, device=model.opt.device)
        x = model.p_sample(x, c, t)
    
    return x.detach().cpu().numpy().reshape(n, -1)

# generate for the year
def generate_for_model(model_path, df_real, calendar, ht: HyperTransformer, condition_trans_cols):   
    with open(model_path, 'rb') as f:  # load the model
        ddpm_model: DDPM = pickle.load(f)
    
    ddpm_model.opt.device = dev
    ddpm_model.eps_model = ddpm_model.eps_model.to(dev)
    
    schedule_tensors = ['beta', 'alpha', 'alpha_bar', 'sigma2'] # moving tensors to the device
    for tensor_name in schedule_tensors:
        tensor = getattr(ddpm_model, tensor_name)
        if isinstance(tensor, torch.Tensor):
            setattr(ddpm_model, tensor_name, tensor.to(dev))
    
    ddpm_model.opt.n_steps = STEP_REDUCTION # step reduction (to make it faster)
    
    # checks
    if not hasattr(ddpm_model.opt, 'seq_len'):
        ddpm_model.opt.seq_len = 1
    if not hasattr(ddpm_model.opt, 'input_dim'):
        if hasattr(ddpm_model.eps_model, 'input_dim'):
            ddpm_model.opt.input_dim = ddpm_model.eps_model.input_dim
        else:
            ddpm_model.opt.input_dim = 16
    
    if hasattr(ddpm_model.eps_model, "input_projector") and hasattr(ddpm_model.eps_model.input_projector, "flatten_parameters"):
        ddpm_model.eps_model.input_projector.flatten_parameters()
    
    tag = model_path.stem.replace("model_", "") # take the tag from the saved model
    
    real_lg = df_real[df_real["location_group"] == FIXED_LOCATION] # filter for the lcoation
    counts = real_lg.groupby(["season", "weekday_group"]).size()

    all_output_cols = ht._output_columns 
    
    # filter according to preprocessing limits
    MAX_CONNECTION_TIME = 5 * 24   
    MIN_CONNECTION_TIME = 2 / 60      
    MAX_ENERGY_SESSION = 150
    MIN_ENERGY_SESSION = 0.5
    MIN_PLUGIN_HOUR = 0
    MAX_PLUGIN_HOUR = 23       
    
    frames = []
    
    # sample for each combination season, weekday gr
    for (seas, wg), dates in tqdm(calendar.items(), desc=f"Buckets {tag}"):
        total = int(counts.get((seas, wg), 0))
        if total == 0:
            continue
            
        lam = total / len(dates)
        daily = np.random.poisson(lam, size=len(dates))
        n_tot = int(daily.sum())
        if n_tot == 0:
            continue
        n_tot = min(n_tot, MAX_PER_BUCKET)

        # condition vectors
        full_row = pd.DataFrame(0, index=[0], columns=ALL_COLS)
        full_row[['season', 'weekday_group', 'location_group']] = [seas, wg, FIXED_LOCATION]
        
        # transforming the conditioning data
        transformed_full = ht.transform(full_row)
        cond_vec = transformed_full[condition_trans_cols].values[0]

        # generate 
        block = sample_batch(ddpm_model, cond_vec, n_tot)
        
        # prepare output
        non_condition_cols = [col for col in all_output_cols if col not in condition_trans_cols]
        synthetic_transformed = pd.DataFrame(block, columns=non_condition_cols)
        
        # combine
        full_transformed = pd.DataFrame(np.tile(transformed_full.values, (n_tot, 1)),columns=all_output_cols)
        full_transformed[non_condition_cols] = synthetic_transformed.values
        synthetic_data = ht.reverse_transform(full_transformed)
        
        # apply max and min
        valid_mask = (
            (synthetic_data['connection_time'] >= MIN_CONNECTION_TIME) &
            (synthetic_data['connection_time'] <= MAX_CONNECTION_TIME) &
            (synthetic_data['energy_session'] >= MIN_ENERGY_SESSION) &
            (synthetic_data['energy_session'] <= MAX_ENERGY_SESSION) &
            (synthetic_data['plugin_hour'] >= MIN_PLUGIN_HOUR) &
            (synthetic_data['plugin_hour'] <= MAX_PLUGIN_HOUR)
        )
        
        synthetic_data = synthetic_data[valid_mask].copy()
        
        if 'plugin_hour' in synthetic_data: # round
            synthetic_data['plugin_hour'] = (
                synthetic_data['plugin_hour'].round().astype(int).clip(0, 23))

        valid_count = len(synthetic_data)

        synthetic_data = synthetic_data[OUTPUT_COLS].copy()
        synthetic_data["season"] = seas
        synthetic_data["weekday_group"] = wg
        synthetic_data["location_group"] = FIXED_LOCATION
        
        offset = 0 # assigning dates
        for date, cnt in zip(dates, daily):
            if cnt == 0:
                continue
            cnt = min(cnt, MAX_PER_BUCKET)
            if offset + cnt > valid_count:
                break
                
            chunk = synthetic_data.iloc[offset:offset+cnt].copy()
            chunk["__date__"] = date.strftime("%Y-%m-%d")
            chunk["__weekday_group__"] = wg
            chunk["__season__"] = seas
            chunk["__count__"] = int(cnt)
            frames.append(chunk)
            offset += cnt

    # save
    if frames:
        out_df = pd.concat(frames, ignore_index=True)
        out_file = OUT_DIR / f"synthetic_location_{FIXED_LOCATION}_{tag}.csv"
        out_df.to_csv(out_file, index=False)

def main():
    df_real = pd.read_csv(RAW_CSV)[ALL_COLS].copy()
    calendar = build_calendar(YEAR)
    
    df_real = df_real[df_real["location_group"] == FIXED_LOCATION]
    
    with open(MODEL_DIR / 'hyper_transformer.pkl', 'rb') as f:
        ht, condition_trans_cols = pickle.load(f) 
    
    models = sorted(MODEL_DIR.glob("model_*.pkl"))
        
    for mp in tqdm(models, desc="Processing models"):
        generate_for_model(mp, df_real, calendar, ht, condition_trans_cols)

if __name__ == "__main__":
    main()