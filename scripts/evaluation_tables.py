"""
Compute Table 3 (yearly performance summary), Table 4 (seasonal performance),
Table 5 (Frobenius / dependency structure), and Table 6 (load-profile metrics)
for Diffusion and CTGAN models across all locations and 5 seeds.

Metrics reported as mean ± std across seeds and locations.
"""

from calendar import month_name
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp, spearmanr, wilcoxon, linregress, t

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
TEST_PATH = BASE_DIR / "diffusion" / "sets" / "test.csv"
VAL_PATH  = BASE_DIR / "diffusion" / "sets" / "val.csv"

DIFF_SYN_DIR  = BASE_DIR / "diffusion" / "synthetic"
CTGAN_SYN_DIR = BASE_DIR / "CTGAN"     / "synthetic"

SEEDS     = [406, 100, 200, 300, 400]
LOCATIONS = [0, 1, 2, 3]

CATEGORICAL_COLS = ["plugin_hour", "plugout_hour"]
NUMERICAL_COLS   = ["connection_time", "energy_session"]
ALL_COLS         = CATEGORICAL_COLS + NUMERICAL_COLS

# Months identified as strongest / weakest in prior single-location analysis
STRONG_MONTH = "January"
WEAK_MONTH   = "May"

# ---------------------------------------------------------------------------
# Data helpers (copied from evaluation notebooks)
# ---------------------------------------------------------------------------

def filter_invalid_sessions(df: pd.DataFrame) -> pd.DataFrame:
    valid = (
        (df["connection_time"] > 0) & (df["energy_session"] > 0)
        & (df["connection_time"] <= 120) & (df["energy_session"] <= 150)
    )
    return df[valid].copy()


def add_time_features(df: pd.DataFrame, plugin_time: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["plugin_time"]  = plugin_time
    df = df.dropna(subset=["plugin_time"])
    df["plugout_time"] = df["plugin_time"] + pd.to_timedelta(df["connection_time"], unit="h")
    df["plugin_hour"]  = df["plugin_time"].dt.hour
    df["plugout_hour"] = df["plugout_time"].dt.hour
    df["month"]        = df["plugin_time"].dt.month
    df["month_name"]   = df["month"].map(lambda x: month_name[x])
    return df


def format_real_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["plugin_day", "plugin_month", "plugin_hour"]).copy()
    plugin_time = pd.to_datetime(
        {"year": 2025, "month": df["plugin_month"], "day": df["plugin_day"], "hour": df["plugin_hour"]},
        errors="coerce",
    )
    return add_time_features(df, plugin_time)


def format_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["__date__", "plugin_hour"]).copy()
    base_time = pd.to_datetime(df["__date__"]) + pd.to_timedelta(df["plugin_hour"], unit="h")
    jitter_minutes = (np.arange(len(df)) * 37 % 61) - 30
    perturb   = pd.to_timedelta(jitter_minutes, unit="m")
    return add_time_features(df, base_time + perturb)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_distribution_metrics(real: pd.Series, synthetic: pd.Series, var_type: str) -> dict:
    real      = real.dropna()
    synthetic = synthetic.dropna()

    if len(real) == 0 or len(synthetic) == 0:
        return {m: np.nan for m in ["TVComplement", "JensenShannon", "KSComplement", "Wasserstein"]}

    if var_type == "categorical":
        real_probs  = real.value_counts(normalize=True).sort_index()
        synth_probs = synthetic.value_counts(normalize=True).sort_index()
        all_cats    = sorted(set(real_probs.index) | set(synth_probs.index))
        real_probs  = real_probs.reindex(all_cats, fill_value=0)
        synth_probs = synth_probs.reindex(all_cats, fill_value=0)
        return {
            "TVComplement":   1.0 - 0.5 * np.abs(real_probs.values - synth_probs.values).sum(),
            "JensenShannon":  jensenshannon(real_probs.values, synth_probs.values),
        }

    ks_stat, _ = ks_2samp(real, synthetic)
    return {
        "KSComplement": 1.0 - float(ks_stat),
        "Wasserstein":  wasserstein_distance(real, synthetic),
    }


def compute_all_metrics(real_df: pd.DataFrame, synth_fmt: pd.DataFrame) -> dict:
    """Return flat metric dict for one (real, synthetic) pair."""
    metrics   = {}
    for col in ALL_COLS:
        col_type = "categorical" if col in CATEGORICAL_COLS else "numerical"
        dist     = compute_distribution_metrics(real_df[col], synth_fmt[col], col_type)
        metrics.update({f"{col}_{k}": v for k, v in dist.items()})
    return metrics


def monthly_metrics(real_df: pd.DataFrame, synth_fmt: pd.DataFrame, month_label: str) -> dict:
    real_month = real_df[real_df["month_name"] == month_label]
    syn_month  = synth_fmt[synth_fmt["month_name"] == month_label]
    if real_month.empty or syn_month.empty:
        return {}
    metrics = {}
    for col in ALL_COLS:
        col_type = "categorical" if col in CATEGORICAL_COLS else "numerical"
        dist     = compute_distribution_metrics(real_month[col], syn_month[col], col_type)
        metrics.update({f"{col}_{k}": v for k, v in dist.items()})
    return metrics


def composite_score(m: dict) -> float:
    keys = ["plugin_hour_TVComplement", "connection_time_KSComplement", "energy_session_KSComplement"]
    vals = [m.get(k, np.nan) for k in keys]
    valid = [v for v in vals if not np.isnan(v)]
    return float(np.mean(valid)) if valid else np.nan

# ---------------------------------------------------------------------------
# Load real test data (all locations, combined val + test)
# ---------------------------------------------------------------------------

print("Loading real data …")
val_df  = pd.read_csv(VAL_PATH).drop(columns="strata", errors="ignore")
test_df = pd.read_csv(TEST_PATH).drop(columns="strata", errors="ignore")
real_all = pd.concat([val_df, test_df], ignore_index=True)
real_all = filter_invalid_sessions(real_all)
real_all = format_real_data(real_all)

# ---------------------------------------------------------------------------
# Collect per-seed, per-location metric rows
# ---------------------------------------------------------------------------

def collect_rows(model: str) -> list[dict]:
    rows = []
    for loc in LOCATIONS:
        real_loc = real_all[real_all["location_group"] == loc]
        if real_loc.empty:
            continue

        for seed in SEEDS:
            if model == "diffusion":
                path = DIFF_SYN_DIR / f"synthetic_location_{loc}_best_diffusion_model_seed{seed}.csv"
            else:
                path = CTGAN_SYN_DIR / f"synthetic_year_{loc}_d2_8h_seed{seed}.csv"

            if not path.exists():
                print(f"  [skip] {path.name}")
                continue

            synth = filter_invalid_sessions(pd.read_csv(path))
            synth_fmt = format_synthetic_data(synth)

            yearly = compute_all_metrics(real_loc, synth_fmt)
            strong = monthly_metrics(real_loc, synth_fmt, STRONG_MONTH)
            weak   = monthly_metrics(real_loc, synth_fmt, WEAK_MONTH)

            rows.append({
                "model":    model,
                "location": loc,
                "seed":     seed,
                "yearly_score":  composite_score(yearly),
                "strong_score":  composite_score(strong),
                "weak_score":    composite_score(weak),
                **{f"yearly_{k}": v for k, v in yearly.items()},
                **{f"{STRONG_MONTH.lower()}_{k}": v for k, v in strong.items()},
                **{f"{WEAK_MONTH.lower()}_{k}": v for k, v in weak.items()},
            })

    return rows


print("Evaluating Diffusion …")
diff_rows  = collect_rows("diffusion")
print("Evaluating CTGAN …")
ctgan_rows = collect_rows("ctgan")

all_rows = diff_rows + ctgan_rows
df = pd.DataFrame(all_rows)

# ---------------------------------------------------------------------------
# Helper: mean ± std string
# ---------------------------------------------------------------------------

def ms(series: pd.Series, decimals: int = 3) -> str:
    m  = series.mean()
    s  = series.std()
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def fmt_pvalue(p_value: float) -> str:
    if np.isnan(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def paired_wilcoxon_pvalue(
    metric_df: pd.DataFrame,
    metric_col: str,
    pair_cols: tuple[str, str] = ("location", "seed"),
) -> float:
    paired = (
        metric_df
        .pivot_table(index=list(pair_cols), columns="model", values=metric_col, aggfunc="mean")
        .dropna(subset=["diffusion", "ctgan"])
    )
    if len(paired) < 2:
        return np.nan

    diff = paired["diffusion"] - paired["ctgan"]
    if np.allclose(diff, 0, equal_nan=False):
        return 1.0

    try:
        return float(wilcoxon(paired["diffusion"], paired["ctgan"]).pvalue)
    except ValueError:
        return np.nan


def ols_interaction_pvalue(reg_df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    """Return Diffusion-minus-CTGAN slope difference and its two-sided p-value."""
    clean = reg_df.dropna(subset=[score_col, "log10_session_count", "is_diffusion"]).copy()
    if len(clean) < 5:
        return np.nan, np.nan

    x = clean["log10_session_count"].to_numpy(dtype=float)
    x = x - x.mean()
    model = clean["is_diffusion"].to_numpy(dtype=float)
    y = clean[score_col].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(clean)), x, model, x * model])

    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ beta
        dof = len(clean) - design.shape[1]
        if dof <= 0:
            return float(beta[3]), np.nan
        sigma2 = float((residuals @ residuals) / dof)
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se = float(np.sqrt(cov[3, 3]))
        if se == 0:
            return float(beta[3]), np.nan
        stat = float(beta[3] / se)
        p_value = float(2 * t.sf(abs(stat), dof))
        return float(beta[3]), p_value
    except np.linalg.LinAlgError:
        return np.nan, np.nan

# ---------------------------------------------------------------------------
# TABLE 3 — Yearly performance summary
# ---------------------------------------------------------------------------

TABLE3_METRICS = {
    "Plug-in Hour":      [("TV Complement",   "yearly_plugin_hour_TVComplement",      "higher"),
                          ("JS Divergence",   "yearly_plugin_hour_JensenShannon",     "lower")],
    "Plug-out Hour":     [("TV Complement",   "yearly_plugout_hour_TVComplement",     "higher"),
                          ("JS Divergence",   "yearly_plugout_hour_JensenShannon",    "lower")],
    "Connection Time":   [("KS Complement",   "yearly_connection_time_KSComplement",  "higher"),
                          ("Wasserstein Distance", "yearly_connection_time_Wasserstein", "lower")],
    "Energy Session":    [("KS Complement",   "yearly_energy_session_KSComplement",   "higher"),
                          ("Wasserstein Distance", "yearly_energy_session_Wasserstein","lower")],
}

print("\n" + "="*80)
print("TABLE 3 — Yearly performance summary (mean ± std across locations × seeds)")
print("="*80)

header = f"{'Category':<18}{'Metric':<22}{'Diffusion':>22}{'CTGAN':>22}{'p-value':>12}"
print(header)
print("-" * len(header))

table3_rows = []
for category, metric_list in TABLE3_METRICS.items():
    first = True
    for metric_label, col, direction in metric_list:
        diff_vals  = df[df["model"] == "diffusion"][col].dropna()
        ctgan_vals = df[df["model"] == "ctgan"][col].dropna()
        d_str = ms(diff_vals)
        c_str = ms(ctgan_vals)
        p_value = paired_wilcoxon_pvalue(df, col)
        p_str = fmt_pvalue(p_value)
        cat_str = category if first else ""
        print(f"{cat_str:<18}{metric_label:<22}{d_str:>22}{c_str:>22}{p_str:>12}")
        table3_rows.append({
            "Category": category, "Metric": metric_label,
            "Diffusion": d_str, "CTGAN": c_str, "p_value": p_value, "better": direction,
        })
        first = False

table3_df = pd.DataFrame(table3_rows)
out3 = BASE_DIR / "table3_yearly_performance.csv"
table3_df.to_csv(out3, index=False)
print(f"\nSaved → {out3}")

# ---------------------------------------------------------------------------
# TABLE 4 — Seasonal performance (year / strong month / weak month)
# ---------------------------------------------------------------------------

TABLE4_METRICS = {
    "Average Score":        ("yearly_score",                         "strong_score",                          "weak_score"),
    "Plug-in Hour TVC":     ("yearly_plugin_hour_TVComplement",      f"{STRONG_MONTH.lower()}_plugin_hour_TVComplement",   f"{WEAK_MONTH.lower()}_plugin_hour_TVComplement"),
    "Connection Time KS":   ("yearly_connection_time_KSComplement",  f"{STRONG_MONTH.lower()}_connection_time_KSComplement", f"{WEAK_MONTH.lower()}_connection_time_KSComplement"),
    "Energy Session KS":    ("yearly_energy_session_KSComplement",   f"{STRONG_MONTH.lower()}_energy_session_KSComplement",  f"{WEAK_MONTH.lower()}_energy_session_KSComplement"),
}

print("\n" + "="*100)
print(f"TABLE 4 — Seasonal performance (Year / {STRONG_MONTH} / {WEAK_MONTH})  —  mean ± std across locations × seeds")
print("="*100)

col_w = 24
header4 = (
    f"{'Metric':<22}"
    f"{'Diffusion Year':>{col_w}}{'Diffusion '+STRONG_MONTH:>{col_w}}{'Diffusion '+WEAK_MONTH:>{col_w}}"
    f"{'CTGAN Year':>{col_w}}{'CTGAN '+STRONG_MONTH:>{col_w}}{'CTGAN '+WEAK_MONTH:>{col_w}}"
)
print(header4)
print("-" * len(header4))

table4_rows = []
for metric_label, (ycol, scol, wcol) in TABLE4_METRICS.items():
    row = {"Metric": metric_label}
    parts = []
    for model_tag in ("diffusion", "ctgan"):
        sub = df[df["model"] == model_tag]
        for col in (ycol, scol, wcol):
            vals = sub[col].dropna()
            s = ms(vals) if len(vals) > 0 else "n/a"
            parts.append(s)
            key = f"{model_tag.upper()} {'Year' if col==ycol else (STRONG_MONTH if col==scol else WEAK_MONTH)}"
            row[key] = s
    print(f"{metric_label:<22}" + "".join(f"{p:>{col_w}}" for p in parts))
    table4_rows.append(row)

table4_df = pd.DataFrame(table4_rows)
out4 = BASE_DIR / "table4_seasonal_performance.csv"
table4_df.to_csv(out4, index=False)
print(f"\nSaved → {out4}")

# ---------------------------------------------------------------------------
# ROBUSTNESS — Performance versus available sessions by location
# ---------------------------------------------------------------------------

print("\n" + "="*90)
print("ROBUSTNESS — Yearly score versus real session count by location")
print("="*90)

location_counts = (
    real_all.groupby("location_group")
    .size()
    .rename("session_count")
    .reset_index()
    .rename(columns={"location_group": "location"})
)

location_rows = []
for loc in LOCATIONS:
    count_row = location_counts[location_counts["location"] == loc]
    session_count = int(count_row["session_count"].iloc[0]) if not count_row.empty else 0
    row = {"location": loc, "session_count": session_count}
    for model_tag in ("diffusion", "ctgan"):
        vals = df[(df["model"] == model_tag) & (df["location"] == loc)]["yearly_score"].dropna()
        prefix = model_tag
        row[f"{prefix}_yearly_score"] = ms(vals) if len(vals) else "n/a"
        row[f"{prefix}_yearly_score_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{prefix}_yearly_score_std"] = float(vals.std()) if len(vals) else np.nan
        row[f"{prefix}_n_runs"] = int(len(vals))
    location_rows.append(row)

location_perf_df = pd.DataFrame(location_rows)
out_loc = BASE_DIR / "table7_location_robustness.csv"
location_perf_df.to_csv(out_loc, index=False)

print(f"{'Location':<12}{'Sessions':>12}{'Diffusion':>22}{'CTGAN':>22}")
print("-" * 68)
for _, row in location_perf_df.iterrows():
    print(
        f"{int(row['location']):<12}"
        f"{int(row['session_count']):>12}"
        f"{row['diffusion_yearly_score']:>22}"
        f"{row['ctgan_yearly_score']:>22}"
    )
print(f"\nSaved → {out_loc}")

regression_df = df.merge(location_counts, on="location", how="left")
regression_df["log10_session_count"] = np.log10(regression_df["session_count"])
regression_df["is_diffusion"] = (regression_df["model"] == "diffusion").astype(int)

regression_rows = []
for model_tag in ("diffusion", "ctgan"):
    sub = regression_df[regression_df["model"] == model_tag].dropna(
        subset=["yearly_score", "log10_session_count"]
    )
    if len(sub) >= 2:
        res = linregress(sub["log10_session_count"], sub["yearly_score"])
        slope = float(res.slope)
        p_value = float(res.pvalue)
        r2 = float(res.rvalue ** 2)
    else:
        slope, p_value, r2 = np.nan, np.nan, np.nan
    regression_rows.append({
        "model": model_tag,
        "score": "yearly_score",
        "slope_per_log10_session_count": slope,
        "slope_p_value": p_value,
        "r_squared": r2,
        "n": int(len(sub)),
    })

slope_diff, slope_diff_p = ols_interaction_pvalue(regression_df, "yearly_score")
regression_rows.append({
    "model": "diffusion_minus_ctgan",
    "score": "yearly_score",
    "slope_per_log10_session_count": slope_diff,
    "slope_p_value": slope_diff_p,
    "r_squared": np.nan,
    "n": int(regression_df.dropna(subset=["yearly_score", "log10_session_count"]).shape[0]),
})

robustness_df = pd.DataFrame(regression_rows)
out_robust = BASE_DIR / "table8_count_score_regression.csv"
robustness_df.to_csv(out_robust, index=False)

print("\nCount-score regression: yearly_score ~ log10(session_count)")
print(f"{'Model':<24}{'Slope':>12}{'p-value':>12}{'R^2':>10}{'n':>6}")
print("-" * 64)
for _, row in robustness_df.iterrows():
    r2_str = "n/a" if np.isnan(row["r_squared"]) else f"{row['r_squared']:.3f}"
    print(
        f"{row['model']:<24}"
        f"{row['slope_per_log10_session_count']:>12.4f}"
        f"{fmt_pvalue(row['slope_p_value']):>12}"
        f"{r2_str:>10}"
        f"{int(row['n']):>6}"
    )
print(f"\nSaved → {out_robust}")

# ---------------------------------------------------------------------------
# Association / Frobenius helpers
# ---------------------------------------------------------------------------

ASSOC_COLS = ["plugin_hour", "plugout_hour", "connection_time", "energy_session"]


def ensure_plugout_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "plugout_hour" not in df.columns:
        df = df.copy()
        df["plugout_hour"] = (
            np.floor(df["plugin_hour"] + df["connection_time"]).astype(int) % 24
        )
    return df


def assoc_matrix(df: pd.DataFrame) -> np.ndarray:
    sub = df[ASSOC_COLS].dropna().astype(float)
    res = spearmanr(sub)
    # scipy >= 1.9 stores result in .statistic; older versions use .correlation
    if hasattr(res, "statistic") and res.statistic is not None:
        corr = np.asarray(res.statistic, dtype=float)
    else:
        corr = np.asarray(res.correlation, dtype=float)
    return np.atleast_2d(corr)


def frob_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro"))


# ---------------------------------------------------------------------------
# Load-profile helpers
# ---------------------------------------------------------------------------

def sessions_to_hourly_load(df: pd.DataFrame) -> pd.DataFrame:
    starts   = df["plugin_hour"].values.astype(float)
    durs     = np.clip(df["connection_time"].values.astype(float), 0.01, 24.0)
    energies = np.clip(df["energy_session"].values.astype(float), 0.0, np.inf)
    dates    = df["date"].values
    powers   = energies / durs

    records = []
    for i in range(len(df)):
        s, d, p, dt = starts[i], durs[i], powers[i], dates[i]
        end = s + d
        h   = int(s)
        while h < end:
            ov = min(h + 1, end) - max(h, s)
            if ov > 0:
                records.append((dt, h % 24, p * ov))
            h += 1

    load = pd.DataFrame(records, columns=["date", "hour", "load_kw"])
    return load.groupby(["date", "hour"])["load_kw"].sum().reset_index()


def make_norm_grid(df: pd.DataFrame) -> pd.DataFrame:
    grid = (
        sessions_to_hourly_load(df)
        .pivot_table(index="date", columns="hour", values="load_kw",
                     aggfunc="sum", fill_value=0.0)
        .reindex(columns=range(24), fill_value=0.0)
    )
    return grid / (len(df) / len(grid))


def hourly_cv(grid: pd.DataFrame) -> np.ndarray:
    mu = grid.mean(axis=0).values
    sd = grid.std(axis=0).values
    return sd / (mu + 1e-6)


def load_profile_metrics(grid_syn: pd.DataFrame, grid_real: pd.DataFrame) -> dict:
    mu_real  = grid_real.mean(axis=0).values
    mu_syn   = grid_syn.mean(axis=0).values
    rmse     = float(np.sqrt(np.mean((mu_real - mu_syn) ** 2)))
    peak_real = grid_real.max(axis=1).values
    peak_syn  = grid_syn.max(axis=1).values
    ks, _    = ks_2samp(peak_real, peak_syn)
    cv_err   = float(np.mean(np.abs(hourly_cv(grid_syn) - hourly_cv(grid_real))))
    peak_err = float(
        abs(peak_syn.mean() - peak_real.mean()) / (peak_real.mean() + 1e-9)
    )
    return dict(rmse=rmse, ks_peak=float(ks), cv_err=cv_err, peak_rel_err=peak_err)


# ---------------------------------------------------------------------------
# Collect Frobenius + load-profile rows per seed × location
# ---------------------------------------------------------------------------

def collect_struct_rows(model: str) -> list[dict]:
    rows = []
    for loc in LOCATIONS:
        real_loc = real_all[real_all["location_group"] == loc].copy()
        if real_loc.empty:
            continue
        real_loc = ensure_plugout_hour(real_loc)
        real_loc["date"] = real_loc["plugin_time"].dt.date

        mat_real   = assoc_matrix(real_loc)
        grid_real  = make_norm_grid(real_loc)

        for seed in SEEDS:
            if model == "diffusion":
                path = DIFF_SYN_DIR / f"synthetic_location_{loc}_best_diffusion_model_seed{seed}.csv"
            else:
                path = CTGAN_SYN_DIR / f"synthetic_year_{loc}_d2_8h_seed{seed}.csv"

            if not path.exists():
                continue

            synth = filter_invalid_sessions(pd.read_csv(path))
            synth = ensure_plugout_hour(synth)
            synth["date"] = pd.to_datetime(synth["__date__"]).dt.date

            frob  = frob_norm(mat_real, assoc_matrix(synth))
            lp    = load_profile_metrics(make_norm_grid(synth), grid_real)

            rows.append({
                "model":    model,
                "location": loc,
                "seed":     seed,
                "frobenius": frob,
                **lp,
            })
    return rows


print("\nComputing Frobenius + load-profile metrics for Diffusion …")
struct_diff  = collect_struct_rows("diffusion")
print("Computing Frobenius + load-profile metrics for CTGAN …")
struct_ctgan = collect_struct_rows("ctgan")

sdf = pd.DataFrame(struct_diff + struct_ctgan)

# ---------------------------------------------------------------------------
# TABLE 5 — Dependency structure (Frobenius norm)
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("TABLE 5 — Dependency structure: Frobenius norm (mean ± std)")
print("="*70)

frob_d = sdf[sdf["model"] == "diffusion"]["frobenius"]
frob_c = sdf[sdf["model"] == "ctgan"]["frobenius"]
print(f"{'Metric':<30}{'Diffusion':>22}{'CTGAN':>22}")
print("-" * 74)
row5 = {"Metric": "Frobenius Norm",
        "Diffusion": ms(frob_d), "CTGAN": ms(frob_c), "better": "lower"}
print(f"{'Frobenius Norm':<30}{ms(frob_d):>22}{ms(frob_c):>22}")

table5_df = pd.DataFrame([row5])
out5 = BASE_DIR / "table5_frobenius.csv"
table5_df.to_csv(out5, index=False)
print(f"\nSaved → {out5}")

# ---------------------------------------------------------------------------
# TABLE 6 — Load-profile metrics
# ---------------------------------------------------------------------------

LP_METRICS = [
    ("rmse",         "Load-curve RMSE",   "lower"),
    ("ks_peak",      "Peak KS stat",      "lower"),
    ("cv_err",       "CV error",          "lower"),
    ("peak_rel_err", "Peak rel. error",   "lower"),
]

print("\n" + "="*70)
print("TABLE 6 — Load-profile metrics (mean ± std)")
print("="*70)
print(f"{'Metric':<26}{'Diffusion':>22}{'CTGAN':>22}{'p-value':>12}")
print("-" * 70)

table6_rows = []
for col, label, better in LP_METRICS:
    d_vals = sdf[sdf["model"] == "diffusion"][col]
    c_vals = sdf[sdf["model"] == "ctgan"][col]
    p_value = paired_wilcoxon_pvalue(sdf, col)
    p_str = fmt_pvalue(p_value)
    print(f"{label:<26}{ms(d_vals):>22}{ms(c_vals):>22}{p_str:>12}")
    table6_rows.append({"Metric": label,
                         "Diffusion": ms(d_vals), "CTGAN": ms(c_vals),
                         "p_value": p_value, "better": better})

table6_df = pd.DataFrame(table6_rows)
out6 = BASE_DIR / "table6_load_profile_metrics.csv"
table6_df.to_csv(out6, index=False)
print(f"\nSaved → {out6}")

print("\nDone.")
