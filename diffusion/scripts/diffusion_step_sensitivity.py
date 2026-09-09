from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdt import HyperTransformer
from rdt.transformers import FloatFormatter, GaussianNormalizer, OneHotEncoder
from torch.utils.data import DataLoader, Dataset

from diffusion import DDPM


BASE = Path(__file__).resolve().parents[2]
SET_DIR = BASE / "diffusion" / "sets"
RESULT_DIR = BASE / "diffusion" / "results"

COND_COLS = ["season", "weekday_group", "location_group"]
OUTPUT_FOCUS = ["plugin_hour", "connection_time", "energy_session"]
OTHER_CATS = ["plugin_day", "plugin_month"]
OTHER_CONT = ["electricity_price", "temperature", "humidity", "solar_radiation", "wind_speed"]
ALL_COLS = COND_COLS + OUTPUT_FOCUS + OTHER_CATS + OTHER_CONT

MIN_CONNECTION_TIME = 2 / 60
MAX_CONNECTION_TIME = 5 * 24
MIN_ENERGY_SESSION = 0.5
MAX_ENERGY_SESSION = 150
MIN_PLUGIN_HOUR = 0
MAX_PLUGIN_HOUR = 23


class ConditionalDataset(Dataset):
    def __init__(self, x: np.ndarray, c: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"input": self.x[idx].unsqueeze(0), "condition": self.c[idx]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_transformer(df_train: pd.DataFrame) -> tuple[HyperTransformer, list[str]]:
    ht = HyperTransformer()
    sdtypes = {
        "season": "categorical",
        "weekday_group": "categorical",
        "location_group": "categorical",
        "plugin_hour": "numerical",
        "plugin_day": "categorical",
        "plugin_month": "categorical",
        "connection_time": "numerical",
        "energy_session": "numerical",
        "electricity_price": "numerical",
        "temperature": "numerical",
        "humidity": "numerical",
        "solar_radiation": "numerical",
        "wind_speed": "numerical",
    }
    transformers = {
        "season": OneHotEncoder(),
        "weekday_group": OneHotEncoder(),
        "location_group": OneHotEncoder(),
        "plugin_hour": GaussianNormalizer(),
        "plugin_day": OneHotEncoder(),
        "plugin_month": OneHotEncoder(),
        "connection_time": GaussianNormalizer(),
        "energy_session": GaussianNormalizer(),
        "electricity_price": FloatFormatter(missing_value_replacement="mean"),
        "temperature": FloatFormatter(missing_value_replacement="mean"),
        "humidity": FloatFormatter(missing_value_replacement="mean"),
        "solar_radiation": FloatFormatter(missing_value_replacement="mean"),
        "wind_speed": FloatFormatter(missing_value_replacement="mean"),
    }

    ht.set_config({"sdtypes": sdtypes, "transformers": transformers})
    ht.fit(df_train[ALL_COLS])

    output_columns = ht.get_output_columns() if hasattr(ht, "get_output_columns") else ht._output_columns
    condition_trans_cols: list[str] = []
    for col in COND_COLS:
        matched = [c for c in output_columns if c.startswith(f"{col}.")]
        condition_trans_cols.extend(matched if matched else [col])

    return ht, condition_trans_cols


def transform_data(ht: HyperTransformer, condition_cols: list[str], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformed = ht.transform(df[ALL_COLS])
    valid_condition_cols = [col for col in condition_cols if col in transformed.columns]
    cond_data = transformed[valid_condition_cols].values.astype(np.float32)
    input_data = transformed.drop(columns=valid_condition_cols).values.astype(np.float32)
    return input_data, cond_data


def build_model(cfg: dict, device: torch.device) -> DDPM:
    opt = SimpleNamespace(
        network="attention",
        depth=cfg["depth"],
        nhead=cfg["nhead"],
        seq_len=1,
        input_dim=cfg["input_dim"],
        cond_dim=cfg["cond_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_steps=cfg["n_steps"],
        schedule=cfg["schedule"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        init_lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        device=device,
    )
    return DDPM(opt, DataLoader([]))


@torch.no_grad()
def sample_batch(model: DDPM, cond_vecs: np.ndarray) -> np.ndarray:
    model.eps_model.eval()
    n_samples = len(cond_vecs)
    c = torch.tensor(cond_vecs, dtype=torch.float32, device=model.opt.device)
    x = torch.randn(n_samples, model.opt.seq_len, model.opt.input_dim, device=model.opt.device)

    if hasattr(model.eps_model, "input_projector") and hasattr(model.eps_model.input_projector, "flatten_parameters"):
        model.eps_model.input_projector.flatten_parameters()

    for step in range(model.opt.n_steps - 1, -1, -1):
        t = torch.full((n_samples,), step, dtype=torch.long, device=model.opt.device)
        x = model.p_sample(x, c, t)

    return x.cpu().numpy().reshape(n_samples, model.opt.input_dim)


def train_ddpm(model: DDPM, log_every: int = 10) -> list[float]:
    model.eps_model.train()
    epoch_losses = []
    for epoch in range(model.opt.n_epochs):
        batch_losses = []
        for data in model.data_loader:
            x0 = data["input"].to(model.opt.device)
            c = data["condition"].to(model.opt.device)

            model.optimizer.zero_grad()
            loss = model.cal_loss(x0, c)
            loss.backward()
            model.optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        model.lr_scheduler.step()

        if (epoch + 1) == 1 or (epoch + 1) % log_every == 0 or (epoch + 1) == model.opt.n_epochs:
            print(f"  epoch {epoch + 1:>3}/{model.opt.n_epochs} | loss={avg_loss:.4f}", flush=True)

    return epoch_losses


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[
        (df["connection_time"] >= MIN_CONNECTION_TIME)
        & (df["connection_time"] <= MAX_CONNECTION_TIME)
        & (df["energy_session"] >= MIN_ENERGY_SESSION)
        & (df["energy_session"] <= MAX_ENERGY_SESSION)
        & (df["plugin_hour"] >= MIN_PLUGIN_HOUR)
        & (df["plugin_hour"] <= MAX_PLUGIN_HOUR)
    ]
    if len(df) == 0:
        return df
    df["plugin_hour"] = df["plugin_hour"].round().astype(int).clip(MIN_PLUGIN_HOUR, MAX_PLUGIN_HOUR)
    df["plugin_month"] = df["plugin_month"].round().astype(int).clip(1, 12)
    return df


def generate_validation_sample(
    model: DDPM,
    ht: HyperTransformer,
    condition_cols: list[str],
    df_val_lg: pd.DataFrame,
) -> pd.DataFrame:
    _, cond_val = transform_data(ht, condition_cols, df_val_lg)
    synth_np = sample_batch(model, cond_val)
    output_columns = ht.get_output_columns() if hasattr(ht, "get_output_columns") else ht._output_columns
    non_condition_cols = [col for col in output_columns if col not in condition_cols]

    transformed_val = ht.transform(df_val_lg[ALL_COLS]).reset_index(drop=True)
    synth_transformed = pd.DataFrame(synth_np, columns=non_condition_cols)
    full_transformed = transformed_val.copy()
    full_transformed[non_condition_cols] = synth_transformed.values
    df_synth = ht.reverse_transform(full_transformed)
    return postprocess(df_synth)


def tv_complement(real: pd.Series, synthetic: pd.Series) -> float:
    real = real.dropna().round().astype(int)
    synthetic = synthetic.dropna().round().astype(int)
    if real.empty or synthetic.empty:
        return np.nan
    categories = sorted(set(real.unique()).union(set(synthetic.unique())))
    real_freq = real.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    synth_freq = synthetic.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    tv_distance = 0.5 * np.abs(real_freq - synth_freq).sum()
    return float(1.0 - tv_distance)


def ks_complement(real: pd.Series, synthetic: pd.Series) -> float:
    real_values = np.sort(real.dropna().astype(float).values)
    synth_values = np.sort(synthetic.dropna().astype(float).values)
    if len(real_values) == 0 or len(synth_values) == 0:
        return np.nan
    values = np.sort(np.concatenate([real_values, synth_values]))
    real_cdf = np.searchsorted(real_values, values, side="right") / len(real_values)
    synth_cdf = np.searchsorted(synth_values, values, side="right") / len(synth_values)
    return float(1.0 - np.max(np.abs(real_cdf - synth_cdf)))


def period_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> dict[str, float]:
    plugin = tv_complement(df_real["plugin_hour"], df_synth["plugin_hour"])
    connection = ks_complement(df_real["connection_time"], df_synth["connection_time"])
    energy = ks_complement(df_real["energy_session"], df_synth["energy_session"])
    score = float(np.nanmean([plugin, connection, energy]))
    return {
        "plugin_hour_TVComplement": plugin,
        "connection_time_KSComplement": connection,
        "energy_session_KSComplement": energy,
        "score": score,
    }


def monthly_average_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    month_scores = []
    for month in range(1, 13):
        real_m = df_real[df_real["plugin_month"].round().astype(int) == month]
        synth_m = df_synth[df_synth["plugin_month"].round().astype(int) == month]
        if len(real_m) == 0 or len(synth_m) == 0:
            continue
        month_scores.append(period_score(real_m, synth_m)["score"])
    return float(np.nanmean(month_scores)) if month_scores else np.nan


def run_sensitivity(args: argparse.Namespace) -> pd.DataFrame:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    df_train = pd.read_csv(SET_DIR / "train.csv")[ALL_COLS]
    df_val = pd.read_csv(SET_DIR / "val.csv")[ALL_COLS]
    df_val_lg = df_val[df_val["location_group"].astype(int) == args.location_group].reset_index(drop=True)

    ht, condition_cols = fit_transformer(df_train)
    x_train, c_train = transform_data(ht, condition_cols, df_train)

    rows = []
    for n_steps in args.n_steps:
        set_seed(args.seed)
        cfg = {
            "depth": 3,
            "nhead": 2,
            "hidden_dim": 256,
            "n_epochs": args.epochs,
            "lr": 5e-4,
            "batch_size": 128,
            "n_steps": int(n_steps),
            "schedule": "linear",
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "input_dim": x_train.shape[1],
            "cond_dim": c_train.shape[1],
        }

        train_ds = ConditionalDataset(x_train, c_train)
        loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        model = build_model(cfg, device)
        model.data_loader = loader

        print(f"Training sensitivity model: n_steps={n_steps}, epochs={args.epochs}, device={device}", flush=True)
        start = time.time()
        losses = train_ddpm(model, log_every=args.log_every)
        runtime = time.time() - start

        df_synth = generate_validation_sample(model, ht, condition_cols, df_val_lg)
        yearly = period_score(df_val_lg, df_synth)
        monthly_avg = monthly_average_score(df_val_lg, df_synth)
        overall = float(np.nanmean([yearly["score"], monthly_avg]))

        row = {
            "n_steps": int(n_steps),
            "schedule": cfg["schedule"],
            "beta_start": cfg["beta_start"],
            "beta_end": cfg["beta_end"],
            "depth": cfg["depth"],
            "nhead": cfg["nhead"],
            "hidden_dim": cfg["hidden_dim"],
            "n_epochs": cfg["n_epochs"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "location_group": args.location_group,
            "valid_synthetic_rows": len(df_synth),
            "validation_rows": len(df_val_lg),
            "final_training_loss": losses[-1] if losses else np.nan,
            "runtime_seconds": runtime,
            "yearly_plugin_hour_TVComplement": yearly["plugin_hour_TVComplement"],
            "yearly_connection_time_KSComplement": yearly["connection_time_KSComplement"],
            "yearly_energy_session_KSComplement": yearly["energy_session_KSComplement"],
            "yearly_score": yearly["score"],
            "monthly_avg_score": monthly_avg,
            "overall_score": overall,
        }
        rows.append(row)

        partial = pd.DataFrame(rows).sort_values("n_steps")
        partial.to_csv(RESULT_DIR / "diffusion_step_sensitivity_lg3.csv", index=False)
        print(f"  saved partial results after n_steps={n_steps}", flush=True)

        cfg_out = RESULT_DIR / f"diffusion_step_sensitivity_config_{n_steps}.json"
        with open(cfg_out, "w") as f:
            json.dump({**cfg, "device": str(device)}, f, indent=2)

    result = pd.DataFrame(rows).sort_values("n_steps")
    result.to_csv(RESULT_DIR / "diffusion_step_sensitivity_lg3.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limited diffusion-step sensitivity analysis for reviewer response R1 #9.")
    parser.add_argument("--n-steps", nargs="+", type=int, default=[250, 500, 1000])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--location-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=406)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    results = run_sensitivity(parse_args())
    print("\nSaved diffusion step sensitivity results:")
    print(results.to_string(index=False))
