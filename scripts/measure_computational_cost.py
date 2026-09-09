#!/usr/bin/env python3
"""Measure computational cost for the EV charging generative models.

The script records hardware information, wall-clock time, and throughput for
generation/inference. It can also time the existing training commands, which is
intended for a server run rather than an interactive local check.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "EV_Charging_Data_processed.csv"
OUT_DIR = BASE / "computational_cost"

MODELS = ("diffusion", "ctgan", "tabddpm")
DEFAULT_SEEDS = [406]
DEFAULT_LOCATIONS = [0]

JOBLIB_TEMP = BASE / "computational_cost" / "joblib_tmp"


@dataclass
class TimingResult:
    model: str
    stage: str
    scope: str
    status: str
    elapsed_seconds: float
    n_rows: int | None = None
    rows_per_second: float | None = None
    command: str | None = None
    note: str | None = None


def parse_ints(values: Iterable[str] | None, default: list[int]) -> list[int]:
    if values is None:
        return default
    return [int(v) for v in values]


def import_from(script_dir: Path, module_name: str):
    sys.path.insert(0, str(script_dir))
    try:
        return importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(script_dir))
        except ValueError:
            pass


def install_numpy_pickle_compat() -> None:
    """Allow old NumPy RandomState pickles to load in newer NumPy releases."""
    try:
        import numpy.random._pickle as numpy_pickle
    except Exception:
        return

    original = getattr(numpy_pickle, "__bit_generator_ctor", None)
    if original is None or getattr(original, "_ev_compat", False):
        return

    def compat_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original(bit_generator_name)

    compat_bit_generator_ctor._ev_compat = True
    numpy_pickle.__bit_generator_ctor = compat_bit_generator_ctor


def configure_single_process_joblib() -> None:
    """Avoid fragile loky semaphore files on shared cluster filesystems."""
    JOBLIB_TEMP.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(JOBLIB_TEMP))
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def run_with_threaded_joblib(func: Callable[[], int]) -> int:
    configure_single_process_joblib()
    try:
        from joblib import parallel_backend
    except Exception:
        return func()
    with parallel_backend("threading", n_jobs=1):
        return func()


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }
    try:
        import psutil

        info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        info["memory_gb"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        else:
            info["gpu_count"] = 0
            info["gpu_names"] = []
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def count_written_rows(directory: Path, started_at: float) -> int:
    total = 0
    for path in directory.glob("*.csv"):
        try:
            if path.stat().st_mtime < started_at:
                continue
        except OSError:
            continue
        try:
            total += len(pd.read_csv(path))
        except Exception:
            pass
    return total


def first_loadable_pickle(paths: list[Path], loader: Callable[[Path], object]) -> Path:
    errors: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            errors.append(f"{path.name}: missing")
            continue
        try:
            loader(path)
            return path
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    raise RuntimeError("No loadable model pickle found. " + " | ".join(errors))


def load_diffusion_transformer(module, df_real: pd.DataFrame):
    transformer_path = module.MODEL_DIR / "hyper_transformer.pkl"
    try:
        with transformer_path.open("rb") as fh:
            return module.pickle.load(fh)
    except Exception as exc:
        message = str(exc)
        if "BitGenerator" not in message and "MT19937" not in message:
            raise

    training_module = import_from(BASE / "diffusion" / "scripts", "training_diffusion")
    train_df = training_module.stratified_split(df_real)
    training_module.fit_preprocessors(train_df)
    with transformer_path.open("rb") as fh:
        return module.pickle.load(fh)


def time_call(
    model: str,
    stage: str,
    scope: str,
    func: Callable[[], int | None],
    note: str | None = None,
) -> TimingResult:
    synchronize_cuda()
    start = time.perf_counter()
    try:
        n_rows = func()
        synchronize_cuda()
        elapsed = time.perf_counter() - start
        throughput = n_rows / elapsed if n_rows and elapsed > 0 else None
        return TimingResult(model, stage, scope, "ok", elapsed, n_rows, throughput, note=note)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        note = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return TimingResult(model, stage, scope, "failed", elapsed, note=note)


def time_command(model: str, stage: str, command: list[str], cwd: Path) -> TimingResult:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, text=True)
    elapsed = time.perf_counter() - start
    return TimingResult(
        model=model,
        stage=stage,
        scope="training_command",
        status="ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
        elapsed_seconds=elapsed,
        command=" ".join(command),
    )


def diffusion_generation(locations: list[int], seeds: list[int], full_scope: bool) -> int:
    def _run() -> int:
        return _diffusion_generation(locations, seeds, full_scope)

    return run_with_threaded_joblib(_run)


def _diffusion_generation(locations: list[int], seeds: list[int], full_scope: bool) -> int:
    module = import_from(BASE / "diffusion" / "scripts", "sample_diffusion")
    df_real = pd.read_csv(module.RAW_CSV)[module.ALL_COLS].copy()
    calendar = module.build_calendar(module.YEAR)
    target_locations = (
        sorted(df_real["location_group"].dropna().astype(int).unique().tolist())
        if full_scope
        else locations
    )
    ht, condition_trans_cols = load_diffusion_transformer(module, df_real)
    model_candidates = [
        module.MODEL_DIR / "best_diffusion_model.pkl",
        module.MODEL_DIR / "model_d3_h2.pkl",
        module.MODEL_DIR / "model_d2_h8.pkl",
    ] + sorted(module.MODEL_DIR.glob("model_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)

    def load_diffusion_pickle(path: Path) -> object:
        with path.open("rb") as fh:
            return module.CPUUnpickler(fh).load()

    model_path = first_loadable_pickle(model_candidates, load_diffusion_pickle)
    started_at = time.time()
    for seed in seeds:
        np.random.seed(seed)
        module.torch.manual_seed(seed)
        for location_id in target_locations:
            module.generate_for_model(
                model_path,
                df_real,
                calendar,
                location_id,
                seed,
                ht,
                condition_trans_cols,
            )
    return count_written_rows(module.OUT_DIR, started_at)


def ctgan_generation(locations: list[int], seeds: list[int], full_scope: bool) -> int:
    def _run() -> int:
        return _ctgan_generation(locations, seeds, full_scope)

    return run_with_threaded_joblib(_run)


def _ctgan_generation(locations: list[int], seeds: list[int], full_scope: bool) -> int:
    module = import_from(BASE / "CTGAN" / "scripts", "sample_ctgan")
    df_real = pd.read_csv(module.RAW_CSV, usecols=module.ALL_COLS)
    calendar = module.build_calendar(module.YEAR)
    target_locations = (
        sorted(df_real["location_group"].dropna().astype(int).unique().tolist())
        if full_scope
        else locations
    )
    started_at = time.time()
    model_candidates = [
        module.MODEL_DIR / f"model_{module.BEST_MODEL_TAG}.pkl",
    ] + sorted(module.MODEL_DIR.glob("model_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)

    def load_ctgan_pickle(path: Path) -> object:
        with path.open("rb") as fh:
            return module.CPUUnpickler(fh).load()

    model_path = first_loadable_pickle(model_candidates, load_ctgan_pickle)
    for seed in seeds:
        np.random.seed(seed)
        for location_id in target_locations:
            module.generate_for_model(model_path, location_id, seed, df_real, calendar)
    return count_written_rows(module.OUT_DIR, started_at)


def tabddpm_generation(locations: list[int], seeds: list[int], full_scope: bool) -> int:
    def _run() -> int:
        return _tabddpm_generation(locations, seeds, full_scope)

    return run_with_threaded_joblib(_run)


def _tabddpm_generation(locations: list[int], seeds: list[int], full_scope: bool) -> int:
    module = import_from(BASE / "TabDDPM" / "scripts", "tabddpm_benchmark")
    args = SimpleNamespace(
        data=DATA_PATH,
        year=module.YEAR,
        seeds=seeds,
        timesteps=1000,
        batch_size=2000,
        scheduler="cosine",
        device="cuda" if module.torch.cuda.is_available() else "cpu",
        seed=306,
        layers=[256, 512, 512, 256],
        dropout=0.0,
        max_per_bucket=None if full_scope else 500,
    )
    started_at = time.time()
    if full_scope:
        module.sample_year(args)
        return count_written_rows(module.OUT_DIR, started_at)

    # The adapter's sample_year function runs all locations. For a quick timing
    # pass, reproduce the same logic for selected locations only.
    module.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_real = pd.read_csv(args.data, usecols=module.ALL_COLS)
    calendar = module.build_calendar(args.year)
    info = json.loads((module.DATASET_DIR / "info.json").read_text())
    cond_to_label = {v: int(k) for k, v in info["conditioning"]["label_to_condition"].items()}
    sampler = module.load_sampler(args)
    for seed in seeds:
        np.random.seed(seed)
        module.torch.manual_seed(seed)
        for location_id in locations:
            real_lg = df_real[df_real["location_group"] == location_id]
            counts = real_lg.groupby(["season", "weekday_group"]).size()
            frames = []
            for (seas, wg), dates in calendar.items():
                total = int(counts.get((seas, wg), 0))
                if total == 0:
                    continue
                daily = np.random.poisson(total / len(dates), size=len(dates))
                n_total = min(int(daily.sum()), args.max_per_bucket)
                if n_total <= 0:
                    continue
                key = f"{int(seas)}|{wg}|{int(location_id)}"
                if key not in cond_to_label:
                    continue
                block = module.valid_sessions(sampler.sample_condition(cond_to_label[key], n_total))
                block = block[module.OUTPUT_COLS].copy()
                block["season"] = seas
                block["weekday_group"] = wg
                block["location_group"] = location_id
                offset = 0
                for date, cnt in zip(dates, daily):
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
                out.to_csv(module.OUT_DIR / f"synthetic_year_{location_id}_tabddpm_seed{seed}.csv", index=False)
    return count_written_rows(module.OUT_DIR, started_at)


def default_training_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    py = sys.executable
    commands = {
        "diffusion": [py, "diffusion/scripts/training_diffusion.py"],
        "ctgan": [py, "CTGAN/scripts/ctgan_train_ttv.py"],
        "tabddpm_prepare": [py, "TabDDPM/scripts/tabddpm_benchmark.py", "prepare"],
        "tabddpm": [
            py,
            "TabDDPM/scripts/tabddpm_benchmark.py",
            "train",
            "--steps",
            str(args.tabddpm_steps),
            "--timesteps",
            str(args.tabddpm_timesteps),
        ],
    }
    return commands


def write_outputs(results: list[TimingResult], hw: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hardware.json").write_text(json.dumps(hw, indent=2), encoding="utf-8")
    csv_path = output_dir / "computational_cost_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TimingResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)
    md = [
        "# Computational Cost Benchmark",
        "",
        "## Hardware",
        "",
        f"- Platform: {hw.get('platform')}",
        f"- CPU count: {hw.get('cpu_count')}",
        f"- Memory: {hw.get('memory_gb')} GB",
        f"- CUDA available: {hw.get('cuda_available')}",
        f"- GPU(s): {', '.join(hw.get('gpu_names', [])) if hw.get('gpu_names') else 'None reported'}",
        "",
        "## Timing Results",
        "",
        "| Model | Stage | Scope | Status | Time (s) | Rows | Rows/s |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in results:
        md.append(
            "| {model} | {stage} | {scope} | {status} | {elapsed:.2f} | {rows} | {rps} |".format(
                model=row.model,
                stage=row.stage,
                scope=row.scope,
                status=row.status,
                elapsed=row.elapsed_seconds,
                rows="" if row.n_rows is None else row.n_rows,
                rps="" if row.rows_per_second is None else f"{row.rows_per_second:.2f}",
            )
        )
    md.append("")
    md.append("Use the CSV file for manuscript table formatting.")
    (output_dir / "computational_cost_results.md").write_text("\n".join(md), encoding="utf-8")
    failures = [row for row in results if row.status != "ok" or row.status.startswith("failed")]
    if failures:
        text = []
        for row in failures:
            text.append(f"## {row.model} {row.stage} {row.scope} [{row.status}]")
            text.append(row.note or "")
            text.append("")
        (output_dir / "failures.log").write_text("\n".join(text), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure hardware, training time, and generation time.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--include-training", action="store_true", help="Also time model training commands.")
    parser.add_argument(
        "--training-models",
        nargs="+",
        choices=MODELS,
        default=None,
        help="Models to train when --include-training is set. Defaults to --models.",
    )
    parser.add_argument(
        "--generation-models",
        nargs="+",
        choices=MODELS,
        default=None,
        help="Models to use for generation timing. Defaults to --models.",
    )
    parser.add_argument(
        "--generation-scope",
        choices=["quick", "full"],
        default="quick",
        help="quick uses selected locations/seeds; full uses all available location groups and seeds.",
    )
    parser.add_argument("--locations", nargs="+", default=None, help="Location groups for quick generation.")
    parser.add_argument("--seeds", nargs="+", default=None, help="Seeds for generation timing.")
    parser.add_argument("--tabddpm-steps", type=int, default=30000)
    parser.add_argument("--tabddpm-timesteps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    locations = parse_ints(args.locations, DEFAULT_LOCATIONS)
    seeds = parse_ints(args.seeds, DEFAULT_SEEDS)
    full_scope = args.generation_scope == "full"
    training_models = args.training_models or args.models
    generation_models = args.generation_models or args.models
    results: list[TimingResult] = []
    hw = hardware_info()

    if args.include_training:
        commands = default_training_commands(args)
        for model in training_models:
            if model == "tabddpm":
                results.append(time_command("tabddpm", "prepare", commands["tabddpm_prepare"], BASE))
            results.append(time_command(model, "training", commands[model], BASE))

    generation_map: dict[str, Callable[[list[int], list[int], bool], int]] = {
        "diffusion": diffusion_generation,
        "ctgan": ctgan_generation,
        "tabddpm": tabddpm_generation,
    }
    for model in generation_models:
        results.append(
            time_call(
                model=model,
                stage="generation",
                scope=args.generation_scope,
                func=lambda m=model: generation_map[m](locations, seeds, full_scope),
            )
        )

    write_outputs(results, hw, args.output_dir)
    print(f"Saved computational-cost results to {args.output_dir}")


if __name__ == "__main__":
    main()
