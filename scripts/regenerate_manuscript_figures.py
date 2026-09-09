"""Regenerate manuscript Figures 5-11 at higher resolution.

The script reproduces the association and load-profile diagnostic figures used in
the revised manuscript. It writes PNG files to a chosen output directory; copy
the outputs into the manuscript `figs/` folder after inspection.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


LOCATION_GROUP = 3
BEST_CTGAN = "d2_h8"
BEST_DIFFUSION = "d3_h2"

C_REAL = "#2166ac"
C_CTGAN = "#d6604d"
C_DIFF = "#4dac26"
C_OTHER = "#bdbdbd"

ASSOC_COLS = ["plugin_hour", "connection_time", "energy_session"]


def safe_read_csv(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    return pd.read_csv(io.StringIO(content))


def filter_sessions(df: pd.DataFrame) -> pd.DataFrame:
    ok = (
        (df["connection_time"] > 0)
        & (df["connection_time"] <= 120)
        & (df["energy_session"] > 0)
        & (df["energy_session"] <= 150)
    )
    return df.loc[ok].copy()


def ensure_plugout_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "plugout_hour" not in df.columns:
        df = df.copy()
        df["plugout_hour"] = (
            np.floor(df["plugin_hour"] + df["connection_time"]).astype(int) % 24
        )
    return df


def parse_ctgan_id(stem: str) -> str | None:
    match = re.search(r"_(d\d)_(\d+)h$", stem)
    if match:
        return f"{match.group(1)}_h{match.group(2)}"
    match = re.search(r"_(d\d)_off$", stem)
    if match:
        return f"{match.group(1)}_h0"
    return None


def parse_diff_id(stem: str) -> str | None:
    match = re.search(r"_(d\d)_h(\d+)$", stem)
    if match:
        return f"{match.group(1)}_h{match.group(2)}"
    match = re.search(r"_(d\d)_hnone$", stem)
    if match:
        return f"{match.group(1)}_h0"
    return None


def load_inputs(base_dir: Path):
    real = safe_read_csv(base_dir / "data" / "EV_Charging_Data_processed.csv")
    real = real.loc[real["location_group"] == LOCATION_GROUP].copy()
    real = ensure_plugout_hour(filter_sessions(real))
    real["plugin_time"] = pd.to_datetime(real["plugin_time"])
    real["date"] = real["plugin_time"].dt.date

    ctgan_models: dict[str, pd.DataFrame] = {}
    for path in sorted((base_dir / "CTGAN" / "synthetic").glob("*.csv")):
        mid = parse_ctgan_id(path.stem)
        if mid is None:
            continue
        df = ensure_plugout_hour(filter_sessions(safe_read_csv(path)))
        df["date"] = pd.to_datetime(df["__date__"]).dt.date
        ctgan_models[mid] = df

    diff_models: dict[str, pd.DataFrame] = {}
    for path in sorted((base_dir / "diffusion" / "synthetic").glob("*.csv")):
        mid = parse_diff_id(path.stem)
        if mid is None:
            continue
        df = ensure_plugout_hour(filter_sessions(safe_read_csv(path)))
        df["date"] = pd.to_datetime(df["__date__"]).dt.date
        diff_models[mid] = df

    return real, ctgan_models, diff_models


def assoc_matrix(df: pd.DataFrame) -> np.ndarray:
    sub = df[ASSOC_COLS].dropna().astype(float)
    result = stats.spearmanr(sub)
    corr = getattr(result, "statistic", None)
    if corr is None:
        corr = result.correlation
    return np.atleast_2d(np.array(corr, dtype=float))


def frob_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro"))


def frob_metrics_df(real: pd.DataFrame, models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mat_real = assoc_matrix(real)
    rows = []
    for mid, df in models.items():
        match = re.match(r"d(\d+)_h(\d+)", mid)
        if not match:
            continue
        rows.append(
            {
                "model_id": mid,
                "depth": int(match.group(1)),
                "heads": int(match.group(2)),
                "frobenius": frob_norm(mat_real, assoc_matrix(df)),
            }
        )
    return pd.DataFrame(rows).set_index("model_id")


def sessions_to_hourly_load(df: pd.DataFrame) -> pd.DataFrame:
    starts = df["plugin_hour"].values.astype(float)
    durs = np.clip(df["connection_time"].values.astype(float), 0.01, 24.0)
    energies = np.clip(df["energy_session"].values.astype(float), 0.0, np.inf)
    dates = df["date"].values
    powers = energies / durs

    records = []
    for i in range(len(df)):
        start, duration, power, date_value = starts[i], durs[i], powers[i], dates[i]
        end = start + duration
        hour = int(start)
        while hour < end:
            overlap = min(hour + 1, end) - max(hour, start)
            if overlap > 0:
                records.append((date_value, hour % 24, power * overlap))
            hour += 1

    load = pd.DataFrame(records, columns=["date", "hour", "load_kw"])
    return load.groupby(["date", "hour"])["load_kw"].sum().reset_index()


def make_grid(load_df: pd.DataFrame) -> pd.DataFrame:
    return (
        load_df.pivot_table(
            index="date", columns="hour", values="load_kw", aggfunc="sum", fill_value=0.0
        )
        .reindex(columns=range(24), fill_value=0.0)
    )


def norm_grid(grid: pd.DataFrame, n_sessions: int) -> pd.DataFrame:
    return grid / (n_sessions / len(grid))


def hourly_cv(norm: pd.DataFrame) -> np.ndarray:
    mu = norm.mean(axis=0).values
    sd = norm.std(axis=0).values
    return sd / (mu + 1e-6)


def build_load_inputs(real, ctgan_models, diff_models):
    grid_real = make_grid(sessions_to_hourly_load(real))
    norm_real = norm_grid(grid_real, len(real))
    ctgan_norms = {
        mid: norm_grid(make_grid(sessions_to_hourly_load(df)), len(df))
        for mid, df in ctgan_models.items()
    }
    diff_norms = {
        mid: norm_grid(make_grid(sessions_to_hourly_load(df)), len(df))
        for mid, df in diff_models.items()
    }
    return norm_real, ctgan_norms, diff_norms


def model_metrics(norm: pd.DataFrame, norm_real: pd.DataFrame) -> dict[str, float]:
    mu_real = norm_real.mean(axis=0).values
    peak_real = norm_real.max(axis=1).values
    mu_syn = norm.mean(axis=0).values
    rmse = np.sqrt(np.mean((mu_real - mu_syn) ** 2))
    ks, _ = stats.ks_2samp(peak_real, norm.max(axis=1).values)
    cv_err = np.mean(np.abs(hourly_cv(norm) - hourly_cv(norm_real)))
    peak_err = abs(norm.max(axis=1).mean() - peak_real.mean()) / (
        peak_real.mean() + 1e-9
    )
    return {
        "rmse": rmse,
        "ks_peak": ks,
        "cv_err": cv_err,
        "peak_rel_err": peak_err,
    }


def build_metrics_df(norms: dict[str, pd.DataFrame], norm_real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, norm in norms.items():
        match = re.match(r"d(\d+)_h(\d+)", mid)
        if not match:
            continue
        rows.append(
            {
                "model_id": mid,
                "depth": int(match.group(1)),
                "heads": int(match.group(2)),
                **model_metrics(norm, norm_real),
            }
        )
    return pd.DataFrame(rows).set_index("model_id")


def save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_assoc_matrices(real, ctgan_models, diff_models, output_dir: Path) -> None:
    labels = [
        ("Real", real),
        (f"CTGAN {BEST_CTGAN}", ctgan_models[BEST_CTGAN]),
        (f"Diffusion {BEST_DIFFUSION}", diff_models[BEST_DIFFUSION]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, (label, df) in zip(axes, labels):
        sns.heatmap(
            pd.DataFrame(assoc_matrix(df), index=ASSOC_COLS, columns=ASSOC_COLS),
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.6,
            cbar=True,
            annot_kws={"size": 11},
            ax=ax,
        )
        ax.set_title(label, fontsize=12)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.suptitle(
        "Spearman Association Matrices - Real vs Best Synthetic Models",
        fontsize=14,
        fontweight="bold",
    )
    save(fig, output_dir, "assoc_matrices_best.png")


def plot_pairwise(real, ctgan_models, diff_models, output_dir: Path) -> None:
    pairs = [
        ("plugin_hour", "connection_time", "Plug-in hour", "Connection time (h)"),
        ("plugin_hour", "energy_session", "Plug-in hour", "Energy (kWh)"),
        ("connection_time", "energy_session", "Connection time (h)", "Energy (kWh)"),
    ]
    datasets = [
        ("Real", real, C_REAL),
        (f"CTGAN ({BEST_CTGAN})", ctgan_models[BEST_CTGAN], C_CTGAN),
        (f"Diffusion ({BEST_DIFFUSION})", diff_models[BEST_DIFFUSION], C_DIFF),
    ]
    fig, axes = plt.subplots(len(pairs), len(datasets), figsize=(16, 13.5), constrained_layout=True)
    for row, (xcol, ycol, xlabel, ylabel) in enumerate(pairs):
        for col, (label, df, color) in enumerate(datasets):
            ax = axes[row, col]
            sub = df[[xcol, ycol]].dropna().astype(float)
            sub = sub.sample(min(5000, len(sub)), random_state=42)
            ax.hexbin(sub[xcol], sub[ycol], gridsize=35, cmap="Blues", mincnt=1, linewidths=0.1)
            rho, _ = stats.spearmanr(sub[xcol], sub[ycol])
            ax.text(
                0.97,
                0.97,
                f"rho = {rho:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=9)
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold", color=color)
    fig.suptitle(
        "Pairwise 2-D Distributions - Key Charging Variables\n"
        f"Real vs Best CTGAN ({BEST_CTGAN}) vs Best Diffusion ({BEST_DIFFUSION}) - "
        f"Location Group {LOCATION_GROUP}",
        fontsize=14,
        fontweight="bold",
    )
    save(fig, output_dir, "assoc_pairwise_2d.png")


def plot_frobenius(real, ctgan_models, diff_models, output_dir: Path) -> None:
    ctgan_frob = frob_metrics_df(real, ctgan_models)
    diff_frob = frob_metrics_df(real, diff_models)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
    for ax, (label, mdf, best_mid) in zip(
        axes,
        [("CTGAN", ctgan_frob, BEST_CTGAN), ("Diffusion", diff_frob, BEST_DIFFUSION)],
    ):
        piv = (
            mdf.reset_index()[["depth", "heads", "frobenius"]]
            .pivot_table(index="depth", columns="heads", values="frobenius")
            .sort_index(ascending=False)
        )
        sns.heatmap(
            piv,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar=True,
            square=True,
            linewidths=0.6,
            linecolor="white",
            annot_kws={"size": 10},
            ax=ax,
        )
        match = re.match(r"d(\d+)_h(\d+)", best_mid)
        if match:
            bd, bh = int(match.group(1)), int(match.group(2))
            if bd in piv.index and bh in piv.columns:
                ri = list(piv.index).index(bd)
                ci = list(piv.columns).index(bh)
                ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=False, edgecolor="black", lw=2.8))
        ax.set_title(f"{label} (lower = closer to real dependency structure)", fontsize=11)
        ax.set_xlabel("Attention heads")
        ax.set_ylabel("Depth")
    fig.suptitle(
        "Frobenius Norm of Spearman Association-Matrix Difference: Synthetic vs Real\n"
        "(black box = best model from distributional evaluation)",
        fontsize=14,
        fontweight="bold",
    )
    save(fig, output_dir, "assoc_frobenius_heatmap.png")


def plot_load_heatmaps(ctgan_metrics, diff_metrics, output_dir: Path) -> None:
    cmap_rg = LinearSegmentedColormap.from_list("rg", ["#006d2c", "#ffffcc", "#bd0026"], N=256)
    metric_cfg = [
        ("rmse", "Load-curve RMSE"),
        ("ks_peak", "Peak KS stat"),
        ("cv_err", "CV error"),
        ("peak_rel_err", "Peak rel. error"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    for col, (metric, title) in enumerate(metric_cfg):
        for row, (label, mdf, best_mid) in enumerate(
            [("CTGAN", ctgan_metrics, BEST_CTGAN), ("Diffusion", diff_metrics, BEST_DIFFUSION)]
        ):
            ax = axes[row, col]
            piv = (
                mdf.reset_index()[["depth", "heads", metric]]
                .pivot_table(index="depth", columns="heads", values=metric)
                .sort_index(ascending=False)
            )
            sns.heatmap(
                piv,
                annot=True,
                fmt=".3f",
                cmap=cmap_rg,
                cbar=False,
                square=True,
                linewidths=0.6,
                linecolor="white",
                annot_kws={"size": 10},
                ax=ax,
                vmin=piv.values.min(),
                vmax=piv.values.max(),
            )
            match = re.match(r"d(\d+)_h(\d+)", best_mid)
            if match:
                bd, bh = int(match.group(1)), int(match.group(2))
                if bd in piv.index and bh in piv.columns:
                    r = list(piv.index).index(bd)
                    c = list(piv.columns).index(bh)
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", lw=2.8))
            ax.set_title(f"{label} | {title} (lower is better)", fontsize=11)
            ax.set_xlabel("Attention heads", fontsize=9)
            ax.set_ylabel("Depth", fontsize=9)
            ax.tick_params(length=0, labelsize=9)
    fig.suptitle(
        "Load-Profile Quality Metrics - Depth x Heads Grid\n"
        "(green = best value per metric; black box = best model from distributional evaluation)",
        fontsize=14,
        fontweight="bold",
    )
    save(fig, output_dir, "lp_metric_heatmaps.png")


def plot_load_curves(norm_real, ctgan_norms, diff_norms, ctgan_metrics, diff_metrics, output_dir: Path) -> None:
    hours = np.arange(24)
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharey=False, constrained_layout=True)
    for ax, (method, norms, best_mid, best_color, metrics) in zip(
        axes,
        [
            ("CTGAN", ctgan_norms, BEST_CTGAN, C_CTGAN, ctgan_metrics),
            ("Diffusion", diff_norms, BEST_DIFFUSION, C_DIFF, diff_metrics),
        ],
    ):
        mu_r = norm_real.mean(axis=0).values
        sd_r = norm_real.std(axis=0).values
        ax.fill_between(hours, mu_r - sd_r, mu_r + sd_r, color=C_REAL, alpha=0.12)
        ax.plot(hours, mu_r, color=C_REAL, lw=2.5, ls="--", label="Real (+/- 1 SD shaded)", zorder=5)
        for mid, norm in norms.items():
            if mid != best_mid:
                ax.plot(hours, norm.mean(axis=0).values, color=C_OTHER, lw=0.9, alpha=0.55, zorder=2)
        norm_b = norms[best_mid]
        mu_b = norm_b.mean(axis=0).values
        sd_b = norm_b.std(axis=0).values
        rmse_b = metrics.loc[best_mid, "rmse"]
        ax.fill_between(hours, mu_b - sd_b, mu_b + sd_b, color=best_color, alpha=0.2)
        ax.plot(hours, mu_b, color=best_color, lw=2.5, label=f"Best: {best_mid} (RMSE={rmse_b:.4f})")
        ax.set_title(f"{method} - {len(norms)} configs vs Real (Location Group {LOCATION_GROUP})", fontsize=12)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Load (kW / avg. session)")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=10)
    fig.suptitle("Mean Daily Load Curves - All Model Configurations", fontsize=14, fontweight="bold")
    save(fig, output_dir, "lp_all_models_curves.png")


def plot_peak(norm_real, ctgan_norms, diff_norms, output_dir: Path) -> None:
    peak_real = norm_real.max(axis=1).values
    entries = [
        ("Real", peak_real, C_REAL),
        (f"CTGAN\n{BEST_CTGAN}", ctgan_norms[BEST_CTGAN].max(axis=1).values, C_CTGAN),
        (f"Diffusion\n{BEST_DIFFUSION}", diff_norms[BEST_DIFFUSION].max(axis=1).values, C_DIFF),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    bp = axes[0].boxplot([e[1] for e in entries], patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch, entry in zip(bp["boxes"], entries):
        patch.set_facecolor(entry[2])
        patch.set_alpha(0.7)
    axes[0].set_xticks(range(1, len(entries) + 1))
    axes[0].set_xticklabels([e[0] for e in entries])
    axes[0].set_ylabel("Daily peak load (kW / session)")
    axes[0].set_title("Peak demand - boxplot")

    xs = np.linspace(0, max(e[1].max() for e in entries) * 1.05, 300)
    for label, arr, color in entries:
        kde = stats.gaussian_kde(arr)
        axes[1].plot(xs, kde(xs), color=color, lw=2, label=label.replace("\n", " "))
        axes[1].axvline(arr.mean(), color=color, lw=1.2, ls="--", alpha=0.7)
    axes[1].set_xlabel("Daily peak load (kW / session)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Peak demand - KDE (dashed = mean)")
    axes[1].legend(fontsize=10)
    fig.suptitle("Peak Demand: Real vs Best Synthetic Models", fontsize=14, fontweight="bold")
    save(fig, output_dir, "lp_peak_demand.png")


def plot_variability(norm_real, ctgan_norms, diff_norms, output_dir: Path) -> None:
    hours = np.arange(24)
    cv_real = hourly_cv(norm_real)
    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    for col, (method, norms, best_mid, best_color) in enumerate(
        [("CTGAN", ctgan_norms, BEST_CTGAN, C_CTGAN), ("Diffusion", diff_norms, BEST_DIFFUSION, C_DIFF)]
    ):
        cv_best = hourly_cv(norms[best_mid])
        ax = fig.add_subplot(gs[0, col])
        for mid, norm in norms.items():
            if mid != best_mid:
                ax.plot(hours, hourly_cv(norm), color=C_OTHER, lw=0.9, alpha=0.5)
        ax.plot(hours, cv_real, color=C_REAL, lw=2.5, ls="--", label="Real", zorder=4)
        ax.plot(hours, cv_best, color=best_color, lw=2.5, label=f"Best: {best_mid}", zorder=5)
        ax.set_title(f"{method} - CV profiles ({len(norms)} configs)")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("CV (std/mean)")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=9)

        ax2 = fig.add_subplot(gs[1, col])
        delta = cv_best - cv_real
        bar_colors = [best_color if d >= 0 else "#888888" for d in delta]
        ax2.bar(hours, delta, color=bar_colors, alpha=0.8, width=0.8)
        ax2.axhline(0, color="black", lw=1)
        worst_h = int(np.argmax(np.abs(delta)))
        offset_x = worst_h + 1.5 if worst_h < 20 else worst_h - 5
        offset_y = delta[worst_h] * 1.3 + (0.05 if delta[worst_h] >= 0 else -0.05)
        ax2.annotate(
            f"h={worst_h}\n{delta[worst_h]:+.2f}",
            xy=(worst_h, delta[worst_h]),
            xytext=(offset_x, offset_y),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1),
        )
        ax2.set_title(f"Delta CV: {method} best ({best_mid}) - Real")
        ax2.set_xlabel("Hour of day")
        ax2.set_ylabel("Delta CV")
        ax2.set_xticks(range(0, 24, 2))
    fig.suptitle("Hourly Variability (CV): Real vs All Configurations", fontsize=14, fontweight="bold")
    save(fig, output_dir, "lp_variability.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("regenerated_figs"))
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )

    real, ctgan_models, diff_models = load_inputs(base_dir)
    missing = [mid for mid in [BEST_CTGAN] if mid not in ctgan_models] + [
        mid for mid in [BEST_DIFFUSION] if mid not in diff_models
    ]
    if missing:
        raise RuntimeError(f"Missing expected model outputs: {missing}")

    norm_real, ctgan_norms, diff_norms = build_load_inputs(real, ctgan_models, diff_models)
    ctgan_metrics = build_metrics_df(ctgan_norms, norm_real)
    diff_metrics = build_metrics_df(diff_norms, norm_real)

    plot_assoc_matrices(real, ctgan_models, diff_models, output_dir)
    plot_pairwise(real, ctgan_models, diff_models, output_dir)
    plot_peak(norm_real, ctgan_norms, diff_norms, output_dir)
    plot_frobenius(real, ctgan_models, diff_models, output_dir)
    plot_load_heatmaps(ctgan_metrics, diff_metrics, output_dir)
    plot_load_curves(norm_real, ctgan_norms, diff_norms, ctgan_metrics, diff_metrics, output_dir)
    plot_variability(norm_real, ctgan_norms, diff_norms, output_dir)

    print(f"Regenerated figures written to: {output_dir}")


if __name__ == "__main__":
    main()
