from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[2]
RAW_CSV = BASE / "data" / "EV_Charging_Data_processed.csv"
OUT_DIR = BASE / "TabDDPM" / "synthetic"
WORK_DIR = BASE / "TabDDPM"
SETS_DIR = WORK_DIR / "sets"
DATASET_DIR = WORK_DIR / "tabddpm_data" / "ev_charging"
MODEL_DIR = WORK_DIR / "models" / "ev_charging"

COND_COLS = ["season", "weekday_group", "location_group"]
NUM_COLS = [
    "connection_time",
    "energy_session",
    "electricity_price",
    "temperature",
    "humidity",
    "solar_radiation",
    "wind_speed",
]
CAT_COLS = ["plugin_hour", "plugin_day", "plugin_month"]
ALL_COLS = COND_COLS + CAT_COLS + NUM_COLS
OUTPUT_COLS = ["plugin_hour", "connection_time", "energy_session"]

SEEDS = [406, 100, 200, 300, 400]
YEAR = 2025


def install_tabddpm_compat_shims() -> None:
    """Provide tiny fallbacks for legacy TabDDPM dependencies during smoke tests.

    The original yandex-research/tab-ddpm code imports a few old utility packages
    at module import time. Full experiments should use the TabDDPM environment, but
    these shims are enough for the train/sample code paths used by this adapter.
    """
    import random
    import types

    if "zero" not in sys.modules:
        try:
            __import__("zero")
        except ModuleNotFoundError:
            zero = types.ModuleType("zero")

            def improve_reproducibility(seed: int = 0) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            class Timer:
                def __init__(self) -> None:
                    self._start = None

                def run(self) -> None:
                    import time
                    self._start = time.time()

                def __str__(self) -> str:
                    if self._start is None:
                        return "0:00:00"
                    import datetime
                    import time
                    return str(datetime.timedelta(seconds=int(time.time() - self._start)))

            def iter_batches(batch, chunk_size: int):
                for start in range(0, len(batch), chunk_size):
                    yield batch[start:start + chunk_size]

            zero.improve_reproducibility = improve_reproducibility
            zero.Timer = Timer
            zero.iter_batches = iter_batches
            zero.random = types.SimpleNamespace(get_state=lambda: None, set_state=lambda state: None)
            zero.hardware = types.SimpleNamespace(get_gpus_info=lambda: [])
            sys.modules["zero"] = zero

    if "icecream" not in sys.modules:
        try:
            __import__("icecream")
        except ModuleNotFoundError:
            icecream = types.ModuleType("icecream")
            icecream.install = lambda *args, **kwargs: None
            sys.modules["icecream"] = icecream

    if "tomli" not in sys.modules:
        try:
            __import__("tomli")
        except ModuleNotFoundError:
            import tomllib
            tomli = types.ModuleType("tomli")
            tomli.load = tomllib.load
            tomli.loads = tomllib.loads
            sys.modules["tomli"] = tomli

    if "tomli_w" not in sys.modules:
        try:
            __import__("tomli_w")
        except ModuleNotFoundError:
            tomli_w = types.ModuleType("tomli_w")
            tomli_w.dump = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("tomli_w is unavailable in this environment")
            )
            sys.modules["tomli_w"] = tomli_w

    if "category_encoders" not in sys.modules:
        try:
            __import__("category_encoders")
        except ModuleNotFoundError:
            category_encoders = types.ModuleType("category_encoders")

            class LeaveOneOutEncoder:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("category_encoders is required for counter categorical encoding")

            category_encoders.LeaveOneOutEncoder = LeaveOneOutEncoder
            sys.modules["category_encoders"] = category_encoders

    if "rtdl" not in sys.modules:
        try:
            __import__("rtdl")
        except ModuleNotFoundError:
            rtdl = types.ModuleType("rtdl")

            class _UnavailableRTDL:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("rtdl is required for this TabDDPM utility path")

            rtdl.CLSToken = _UnavailableRTDL
            rtdl.NumericalFeatureTokenizer = _UnavailableRTDL
            rtdl.CategoricalFeatureTokenizer = _UnavailableRTDL
            sys.modules["rtdl"] = rtdl


def tabddpm_repo() -> Path:
    configured = os.environ.get("TABDDPM_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return BASE.parent / "tab-ddpm"


def add_tabddpm_to_path() -> Path:
    repo = tabddpm_repo()
    if not (repo / "tab_ddpm").exists():
        raise FileNotFoundError(
            f"Cannot find TabDDPM repository at {repo}. Set TABDDPM_REPO=/path/to/tab-ddpm."
        )
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    install_tabddpm_compat_shims()
    return repo


def make_strata(df: pd.DataFrame) -> pd.Series:
    return (
        df["location_group"].astype(str)
        + "_"
        + df["season"].astype(str)
        + "_"
        + df["weekday_group"].astype(str)
    )


def split_data(df: pd.DataFrame, seed: int = 306) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()[ALL_COLS]
    df["strata"] = make_strata(df)
    tr_val, test_set = train_test_split(df, test_size=0.15, random_state=seed, stratify=df["strata"])
    train_set, val_set = train_test_split(
        tr_val, test_size=0.1765, random_state=seed, stratify=tr_val["strata"]
    )
    return (
        train_set.reset_index(drop=True),
        val_set.reset_index(drop=True),
        test_set.reset_index(drop=True),
    )


def condition_key(row: pd.Series) -> str:
    return f"{int(row['season'])}|{row['weekday_group']}|{int(row['location_group'])}"


def write_split_arrays(split_frames: Dict[str, pd.DataFrame], dataset_dir: Path) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    all_keys = sorted({condition_key(row) for df in split_frames.values() for _, row in df.iterrows()})
    cond_to_label = {key: i for i, key in enumerate(all_keys)}
    label_to_cond = {str(v): k for k, v in cond_to_label.items()}

    for split, df in split_frames.items():
        clean = df.drop(columns=["strata"], errors="ignore").copy()
        y = clean.apply(condition_key, axis=1).map(cond_to_label).to_numpy(dtype=np.int64)
        x_num = clean[NUM_COLS].to_numpy(dtype=np.float32)
        x_cat = clean[CAT_COLS].astype(str).to_numpy()
        np.save(dataset_dir / f"X_num_{split}.npy", x_num)
        np.save(dataset_dir / f"X_cat_{split}.npy", x_cat)
        np.save(dataset_dir / f"y_{split}.npy", y)

    info = {
        "task_type": "multiclass",
        "n_classes": len(cond_to_label),
        "conditioning": {
            "columns": COND_COLS,
            "encoding": "joint_label",
            "label_to_condition": label_to_cond,
            "note": (
                "TabDDPM supports one conditioning label. The EV conditioning tuple "
                "(season, weekday_group, location_group) is encoded as one multiclass y."
            ),
        },
        "columns": {"numerical": NUM_COLS, "categorical": CAT_COLS},
        "time_variable_note": (
            "plugin_hour, plugin_day, and plugin_month are categorical features to avoid imposing "
            "linear distance across cyclic calendar boundaries."
        ),
    }
    (dataset_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def prepare(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data, usecols=ALL_COLS)
    train_df, val_df, test_df = split_data(df, seed=args.split_seed)
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SETS_DIR / "train.csv", index=False)
    val_df.to_csv(SETS_DIR / "val.csv", index=False)
    test_df.to_csv(SETS_DIR / "test.csv", index=False)
    info = write_split_arrays({"train": train_df, "val": val_df, "test": test_df}, DATASET_DIR)
    print(f"Prepared TabDDPM dataset at {DATASET_DIR}")
    print(f"Joint conditioning labels: {info['n_classes']}")


def train(args: argparse.Namespace) -> None:
    add_tabddpm_to_path()
    from scripts.train import train as tabddpm_train

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_params = {
        "num_classes": int(json.loads((DATASET_DIR / "info.json").read_text())["n_classes"]),
        "is_y_cond": True,
        "rtdl_params": {
            "d_layers": args.layers,
            "dropout": args.dropout,
        },
    }
    tabddpm_train(
        parent_dir=str(MODEL_DIR),
        real_data_path=str(DATASET_DIR),
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        model_type="mlp",
        model_params=model_params,
        num_timesteps=args.timesteps,
        gaussian_loss_type="mse",
        scheduler=args.scheduler,
        T_dict={
            "seed": args.seed,
            "normalization": "minmax",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default",
        },
        device=torch.device(args.device),
        seed=args.seed,
    )
    config = vars(args).copy()
    config["model_params"] = model_params
    (MODEL_DIR / "benchmark_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"Saved TabDDPM model to {MODEL_DIR}")


def weekday_group(idx: int) -> str:
    return "Mon-Th" if idx < 4 else "Friday" if idx == 4 else "Saturday" if idx == 5 else "Sunday"


def season_idx(dt: datetime) -> int:
    m, d = dt.month, dt.day
    if (m == 12 and d >= 22) or m in {1, 2} or (m == 3 and d <= 20):
        return 0
    if (m == 3 and d >= 21) or m in {4, 5} or (m == 6 and d <= 21):
        return 1
    if (m == 6 and d >= 22) or m in {7, 8} or (m == 9 and d <= 22):
        return 2
    return 3


def leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def build_calendar(year: int) -> Dict[Tuple[int, str], List[datetime]]:
    start = datetime(year, 1, 1)
    buckets: Dict[Tuple[int, str], List[datetime]] = {}
    for offset in range(366 if leap(year) else 365):
        dt = start + timedelta(days=offset)
        buckets.setdefault((season_idx(dt), weekday_group(dt.weekday())), []).append(dt)
    return buckets


@dataclass
class LoadedTabDDPM:
    diffusion: object
    dataset: object
    num_features: int
    cat_features: int
    device: torch.device
    batch_size: int

    def sample_condition(self, label: int, n: int) -> pd.DataFrame:
        y_dist = torch.zeros(self.dataset.n_classes, dtype=torch.float32, device=self.device)
        y_dist[label] = 1.0
        x_gen, y_gen = self.diffusion.sample_all(n, self.batch_size, y_dist, ddim=False)
        x = x_gen.cpu().numpy()

        cols = {}
        offset = 0
        if self.num_features:
            x_num = self.dataset.num_transform.inverse_transform(x[:, : self.num_features])
            cols.update({col: x_num[:, i] for i, col in enumerate(NUM_COLS)})
            offset = self.num_features
        if self.cat_features:
            x_cat = self.dataset.cat_transform.inverse_transform(x[:, offset:])
            cols.update({col: x_cat[:, i] for i, col in enumerate(CAT_COLS)})
        out = pd.DataFrame(cols)
        out["__label__"] = y_gen.cpu().numpy()
        return out


def load_sampler(args: argparse.Namespace) -> LoadedTabDDPM:
    add_tabddpm_to_path()
    from scripts.utils_train import get_model, make_dataset
    from tab_ddpm import GaussianMultinomialDiffusion

    info = json.loads((DATASET_DIR / "info.json").read_text())
    model_params = {
        "num_classes": int(info["n_classes"]),
        "is_y_cond": True,
        "rtdl_params": {"d_layers": args.layers, "dropout": args.dropout},
    }
    T = {
        "seed": args.seed,
        "normalization": "minmax",
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }
    import lib

    D = make_dataset(str(DATASET_DIR), lib.Transformations(**T), info["n_classes"], True, False)
    K = np.array(D.get_category_sizes("train"))
    if len(K) == 0:
        K = np.array([0])
    num_features = D.X_num["train"].shape[1] if D.X_num is not None else 0
    d_in = int(np.sum(K) + num_features)
    model_params["d_in"] = d_in
    model = get_model("mlp", model_params, num_features, category_sizes=D.get_category_sizes("train"))
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
    device = torch.device(args.device)
    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_features,
        denoise_fn=model,
        num_timesteps=args.timesteps,
        gaussian_loss_type="mse",
        scheduler=args.scheduler,
        device=device,
    )
    diffusion.to(device)
    diffusion.eval()
    return LoadedTabDDPM(diffusion, D, num_features, len(CAT_COLS), device, args.batch_size)


def valid_sessions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["plugin_hour"] = pd.to_numeric(out["plugin_hour"], errors="coerce").round().astype("Int64")
    out["connection_time"] = pd.to_numeric(out["connection_time"], errors="coerce")
    out["energy_session"] = pd.to_numeric(out["energy_session"], errors="coerce")
    valid = (
        out["plugin_hour"].between(0, 23)
        & out["connection_time"].between(2 / 60, 5 * 24)
        & out["energy_session"].between(0.5, 150)
    )
    out = out[valid].copy()
    out["plugin_hour"] = out["plugin_hour"].astype(int)
    return out


def sample_year(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_real = pd.read_csv(args.data, usecols=ALL_COLS)
    calendar = build_calendar(args.year)
    info = json.loads((DATASET_DIR / "info.json").read_text())
    cond_to_label = {v: int(k) for k, v in info["conditioning"]["label_to_condition"].items()}
    locations = sorted(df_real["location_group"].dropna().astype(int).unique().tolist())
    sampler = load_sampler(args)

    for seed in tqdm(args.seeds, desc="Seeds"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        for location_id in tqdm(locations, desc=f"Locations (seed={seed})", leave=False):
            real_lg = df_real[df_real["location_group"] == location_id]
            counts = real_lg.groupby(["season", "weekday_group"]).size()
            frames = []
            for (seas, wg), dates in calendar.items():
                total = int(counts.get((seas, wg), 0))
                if total == 0:
                    continue
                daily = np.random.poisson(total / len(dates), size=len(dates))
                n_total = int(daily.sum())
                if args.max_per_bucket is not None:
                    n_total = min(n_total, args.max_per_bucket)
                if n_total <= 0:
                    continue
                key = f"{int(seas)}|{wg}|{int(location_id)}"
                if key not in cond_to_label:
                    continue
                block = valid_sessions(sampler.sample_condition(cond_to_label[key], n_total))
                block = block[OUTPUT_COLS].copy()
                block["season"] = seas
                block["weekday_group"] = wg
                block["location_group"] = location_id
                offset = 0
                for date, cnt in zip(dates, daily):
                    if args.max_per_bucket is not None:
                        cnt = min(int(cnt), args.max_per_bucket)
                    if cnt == 0 or offset + cnt > len(block):
                        continue
                    chunk = block.iloc[offset : offset + cnt].copy()
                    chunk["__date__"] = date.strftime("%Y-%m-%d")
                    chunk["__weekday_group__"] = wg
                    chunk["__season__"] = seas
                    chunk["__count__"] = int(cnt)
                    frames.append(chunk)
                    offset += cnt
            if frames:
                out = pd.concat(frames, ignore_index=True)
                out.to_csv(OUT_DIR / f"synthetic_year_{location_id}_tabddpm_seed{seed}.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TabDDPM benchmark adapter for EV charging experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, default=RAW_CSV)
    p.add_argument("--split-seed", type=int, default=306)
    p.set_defaults(func=prepare)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--timesteps", type=int, default=1000)
    t.add_argument("--batch-size", type=int, default=4096)
    t.add_argument("--lr", type=float, default=0.002)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--scheduler", default="cosine")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--seed", type=int, default=306)
    t.add_argument("--layers", type=int, nargs="+", default=[256, 512, 512, 256])
    t.add_argument("--dropout", type=float, default=0.0)
    t.set_defaults(func=train)
    s = sub.add_parser("sample-year")
    s.add_argument("--data", type=Path, default=RAW_CSV)
    s.add_argument("--year", type=int, default=YEAR)
    s.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    s.add_argument("--timesteps", type=int, default=1000)
    s.add_argument("--batch-size", type=int, default=2000)
    s.add_argument("--scheduler", default="cosine")
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--seed", type=int, default=306)
    s.add_argument("--layers", type=int, nargs="+", default=[256, 512, 512, 256])
    s.add_argument("--dropout", type=float, default=0.0)
    s.add_argument("--max-per-bucket", type=int, default=None)
    s.set_defaults(func=sample_year)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
