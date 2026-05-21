from __future__ import annotations
import argparse
import itertools
import json
import math
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sdmetrics.single_column import KSComplement, TVComplement
from tqdm import tqdm
from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder, GaussianNormalizer, FloatFormatter
from diffusion import DDPM  
from network import Attention, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = Path("...")
DATA_CSV = BASE / "data" / "EV_Charging_Data_processed.csv"
SET_DIR = BASE / "diffusion" / "sets"
MODEL_DIR = BASE / "diffusion" / "models"
OUT_DIR = BASE / "diffusion" / "synthetic"
CFG_FILE = MODEL_DIR / "best_diffusion_cfg.json"
MODEL_FILE = MODEL_DIR / "best_diffusion_model.pkl"

SEED = 406
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

for p in (SET_DIR, MODEL_DIR, OUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

COND_COLS = ["season", "weekday_group", "location_group"]
OUTPUT_FOCUS = ["plugin_hour", "connection_time", "energy_session"]
OTHER_CATS = ["plugin_day", "plugin_month"]
OTHER_CONT = ["electricity_price", "temperature", "humidity","solar_radiation","wind_speed"]
ALL_COLS = COND_COLS + OUTPUT_FOCUS + OTHER_CATS + OTHER_CONT

# helpers 
def leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

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

def build_calendar(year):
    start = datetime(year, 1, 1)
    days = 366 if leap(year) else 365
    calendar = {}
    for day in range(days):
        date = start + timedelta(days=day)
        season = season_idx(date)
        wday_group = weekday_group(date.weekday())
        key = (season, wday_group)
        calendar.setdefault(key, []).append(date)
    return calendar

def load_and_preprocess(): # load dfs and hyperparameter tuning
    train_path = SET_DIR / "train.csv"
    val_path = SET_DIR / "val.csv"
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    df_train = df_train[ALL_COLS]
    df_val = df_val[ALL_COLS]
    
    ht = HyperTransformer()
    
    sdtypes = {
        'season': 'categorical',
        'weekday_group': 'categorical',
        'location_group': 'categorical',
        'plugin_hour': 'numerical', 
        'plugin_day': 'categorical',
        'plugin_month': 'categorical',
        'connection_time': 'numerical',
        'energy_session': 'numerical',
        'electricity_price': 'numerical',
        'temperature': 'numerical',
        'humidity': 'numerical',
        'solar_radiation': 'numerical',
        'wind_speed': 'numerical'}
    
    transformers = {
        'season': OneHotEncoder(),
        'weekday_group': OneHotEncoder(),
        'location_group': OneHotEncoder(),
        'plugin_hour': GaussianNormalizer(),  # continuous transformation
        'plugin_day': OneHotEncoder(),
        'plugin_month': OneHotEncoder(),
        'connection_time': GaussianNormalizer(),
        'energy_session': GaussianNormalizer(),
        'electricity_price': FloatFormatter(missing_value_replacement='mean'),
        'temperature': FloatFormatter(missing_value_replacement='mean'),
        'humidity': FloatFormatter(missing_value_replacement='mean'),
        'solar_radiation': FloatFormatter(missing_value_replacement='mean'),
        'wind_speed': FloatFormatter(missing_value_replacement='mean')}
    
    ht.set_config({'sdtypes': sdtypes, 'transformers': transformers})
    ht.fit(df_train)
    
    condition_trans_cols = []
    output_columns = (ht.get_output_columns() if hasattr(ht, 'get_output_columns') else ht._output_columns)

    for col in COND_COLS:
        matched = [c for c in output_columns if c.startswith(f"{col}.")]
        condition_trans_cols.extend(matched if matched else [col])
        
        return df_train, df_val, ht, condition_trans_cols

class ConditionalDataset(Dataset):
    def __init__(self, x: np.ndarray, c: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"input": self.x[idx].unsqueeze(0), "condition": self.c[idx]}

# building the model based on the configuration
def build_model(cfg):
    opt_dict = {
        "network": cfg["network"],
        "depth": cfg["depth"],
        "nhead": cfg.get("nhead", 2),
        "seq_len": 1,
        "input_dim": cfg["input_dim"],
        "cond_dim": cfg["cond_dim"],
        "hidden_dim": cfg["hidden_dim"],
        "n_steps": cfg["n_steps"],
        "schedule": cfg["schedule"],
        "beta_start": 1e-4,
        "beta_end": cfg["beta_end"],
        "init_lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "n_epochs": cfg["n_epochs"],
        "device": device}
    
    opt = SimpleNamespace(**opt_dict)
    
    return DDPM(opt, DataLoader([]))

@torch.no_grad()
def sample_batch(model: DDPM, cond_vecs):
    n_samples = len(cond_vecs)
    c = torch.tensor(cond_vecs, dtype=torch.float32).to(device)
    x = torch.randn(n_samples, model.opt.seq_len, model.opt.input_dim, device=device)
    
    for step in range(model.opt.n_steps-1, -1, -1):
        t = torch.full((n_samples,), step, dtype=torch.long, device=device)
        x = model.p_sample(x, c, t)
    
    return x.cpu().numpy().reshape(n_samples, model.opt.input_dim)

def evaluate_synthetic(df_real, df_synth):
    metrics = {}
    metrics["KS_plugin"] = KSComplement.compute(real_data=df_real["plugin_hour"], synthetic_data=df_synth["plugin_hour"])
    metrics["KS_connection"] = KSComplement.compute(real_data=df_real["connection_time"], synthetic_data=df_synth["connection_time"])
    metrics["KS_energy"] = KSComplement.compute(real_data=df_real["energy_session"], synthetic_data=df_synth["energy_session"])
    metrics["score"] = (metrics["KS_plugin"] + metrics["KS_connection"] + metrics["KS_energy"]) / 3.0
    return metrics

GRID = {
    "network": ["attention"],
    "depth": [3],
    "nhead": [2],
    "n_steps": [250,500, 1000],
    "n_epochs": [100, 200],
    "lr": [1e-3, 5e-4],
    "batch_size": [128, 256],
    "hidden_dim": [128, 256],
    "schedule": ["linear", "cosine"],
    "beta_end": [0.01, 0.02]}

def generate_key_combinations(grid):
    combinations = []
    for n_steps in grid["n_steps"]:
        for n_epochs in grid["n_epochs"]:
            for lr in grid["lr"]:
                combinations.append({
                    "network": "attention",
                    "depth": 3,
                    "nhead": 2,
                    "n_steps": n_steps,
                    "n_epochs": n_epochs,
                    "lr": lr,
                    "batch_size": 128,
                    "hidden_dim": 256,
                    "schedule": "linear",
                    "beta_end": 0.02})
    
    variations = [
        {"batch_size": 256},
        {"hidden_dim": 128},
        {"schedule": "cosine"},
        {"beta_end": 0.01},
        {"batch_size": 256, "hidden_dim": 128},
        {"schedule": "cosine", "beta_end": 0.01}]
    
    base_config = combinations[0].copy()
    
    for var in variations:
        if len(combinations) >= 20: # max 20 combinations
            break
        new_config = base_config.copy()
        new_config.update(var)
        combinations.append(new_config)
    
    return combinations[:20]

def transform_data(ht: HyperTransformer, condition_cols, df):
    transformed = ht.transform(df)
    valid_condition_cols = [col for col in condition_cols if col in transformed.columns]
    cond_data = transformed[valid_condition_cols].values.astype(np.float32)
    input_data = transformed.drop(columns=valid_condition_cols).values.astype(np.float32)
    return input_data, cond_data

def hypertune(df_train, df_val, ht: HyperTransformer, condition_cols):
    X_train, C_train = transform_data(ht, condition_cols, df_train)
    X_val, C_val = transform_data(ht, condition_cols, df_val)
    
    configs = generate_key_combinations(GRID)    
    best_cfg = None
    best_score = -np.inf
    
    for i, cfg in enumerate(configs):
        print(f"\n Evaluating configuration {i+1}/{len(configs)} ")
        
        try:
            start_time = time.time()
            cfg["input_dim"] = X_train.shape[1]
            cfg["cond_dim"] = C_train.shape[1]
            
            train_ds = ConditionalDataset(X_train, C_train)
            loader = DataLoader(
                train_ds,
                batch_size=cfg["batch_size"],
                shuffle=True,
                pin_memory=True
            )
            
            model = build_model(cfg)
            model.data_loader = loader
            model.train()
            
            synth_np = sample_batch(model, C_val)
            
            synth_transformed = pd.DataFrame(synth_np, 
                columns=[col for col in ht._output_columns if col not in condition_cols])
            
            full_transformed = pd.DataFrame(np.tile(C_val[0], (len(synth_np), 1)),columns=condition_cols)
            full_transformed = pd.concat([full_transformed, synth_transformed], axis=1)
            
            df_synth = ht.reverse_transform(full_transformed)
            
            df_synth = df_synth[(df_synth['plugin_hour'] >= 0) & (df_synth['plugin_hour'] <= 23)]
            
            metrics = evaluate_synthetic(df_val, df_synth)
            score = metrics["score"]

            if score > best_score:
                best_score = score
                best_cfg = cfg.copy()
                
        except Exception as e:
            print(f"Error evaluating configuration: {e}")
            continue
    
    print(f"\nBest configuration: {best_cfg}, score: {best_score:.4f}")
    return best_cfg

def generate_yearly_sample( model: DDPM, ht: HyperTransformer,condition_cols,year = 2025,max_per_bucket = 500):
    calendar = build_calendar(year)
    location_group = 3
    print(f"Generating for fixed location: {location_group}")
    all_sessions = []
    config = ht.get_config()
    sdtypes = config["sdtypes"]
    transformers = config["transformers"]
    default_values = {}
    for col, transformer_name in transformers.items():
        if col in COND_COLS:
            continue
        transformer = transformers[col]
        
        if transformer_name == "OneHotEncoder":
            default_values[col] = transformer["categories"][0]

        elif transformer_name in ["GaussianNormalizer", "FloatFormatter"]:
            default_values[col] = 0.0
        else:
            default_values[col] = 0.0
    
    for (season, wday_group), dates in calendar.items():
        row_data = default_values.copy()
        row_data.update({"season": season, "weekday_group": wday_group, "location_group": location_group})
        
        cond_df = pd.DataFrame([row_data])
        
        transformed_full = ht.transform(cond_df)
        cond_vec = transformed_full[condition_cols].values[0]
        
        avg_sessions = total_sessions / len(dates)
        daily_counts = np.random.poisson(avg_sessions, size=len(dates))
        total_sessions = min(daily_counts.sum(), max_per_bucket)
        
        if total_sessions == 0:
            continue
            
        synth_np = sample_batch(model, np.tile(cond_vec, (total_sessions, 1)))
        
        synth_transformed = pd.DataFrame(synth_np, columns=[col for col in ht._output_columns if col not in condition_cols])
        
        full_transformed = pd.DataFrame(np.tile(cond_vec, (total_sessions, 1)), columns=condition_cols)
        full_transformed = pd.concat([full_transformed, synth_transformed], axis=1)
        
        sessions = ht.reverse_transform(full_transformed)
        
        sessions = sessions[(sessions['plugin_hour'] >= 0) & (sessions['plugin_hour'] <= 23)]
        
        start_idx = 0
        for date, count in zip(dates, daily_counts):
            if count == 0:
                continue
                
            end_idx = start_idx + count
            if end_idx > len(sessions):
                break
                
            day_sessions = sessions.iloc[start_idx:end_idx].copy()
            day_sessions["date"] = date.strftime("%Y-%m-%d")
            day_sessions["season"] = season
            day_sessions["weekday_group"] = wday_group
            day_sessions["location_group"] = location_group
            
            all_sessions.append(day_sessions)
            start_idx = end_idx
    
    return pd.concat(all_sessions, ignore_index=True)

def main(skip_tune=False, year=2025):
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        
        _, _, ht, condition_cols = load_and_preprocess()
        
        synthetic_df = generate_yearly_sample(model, ht, condition_cols, year=year)
        
        out_file = OUT_DIR / f"synthetic_{year}.csv"
        synthetic_df.to_csv(out_file, index=False)
        print(f"Saved synthetic data: {out_file}")
        return

    df_train, df_val, ht, condition_cols = load_and_preprocess()
    
    if not skip_tune:
        best_cfg = hypertune(df_train, df_val, ht, condition_cols)
        with open(CFG_FILE, "w") as f:
            json.dump(best_cfg, f, indent=2)
    elif CFG_FILE.exists():
        with open(CFG_FILE, "r") as f:
            best_cfg = json.load(f)
    else:
        raise FileNotFoundError(f"No saved configuration")

    # final training
    X_train, C_train = transform_data(ht, condition_cols, df_train)
    
    best_cfg["input_dim"] = X_train.shape[1]
    best_cfg["cond_dim"] = C_train.shape[1]
    
    train_ds = ConditionalDataset(X_train, C_train)
    loader = DataLoader(train_ds, batch_size=best_cfg["batch_size"], shuffle=True)
    
    model = build_model(best_cfg)
    model.data_loader = loader
    model.train()
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    synthetic_df = generate_yearly_sample(model, ht, condition_cols, year=year)
    
    out_file = OUT_DIR / f"synthetic_{year}.csv"
    synthetic_df.to_csv(out_file, index=False)
    
if __name__ == "__main__":
    main()