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

<<<<<<< HEAD
DIFF_SYN_DIR  = BASE_DIR / "diffusion" / "synthetic"
CTGAN_SYN_DIR = BASE_DIR / "CTGAN"     / "synthetic"
=======
DIFF_SYN_DIR    = BASE_DIR / "diffusion" / "synthetic"
CTGAN_SYN_DIR   = BASE_DIR / "CTGAN" / "synthetic"
TABDDPM_SYN_DIR = BASE_DIR / "TabDDPM" / "synthetic"

MODEL_SPECS = {
    "diffusion": {
        "label": "Diffusion",
        "path": lambda loc, seed: DIFF_SYN_DIR / f"synthetic_location_{loc}_best_diffusion_model_seed{seed}.csv",
    },
    "ctgan": {
        "label": "CTGAN",
        "path": lambda loc, seed: CTGAN_SYN_DIR / f"synthetic_year_{loc}_d2_8h_seed{seed}.csv",
    },
    "tabddpm": {
        "label": "TabDDPM",
        "path": lambda loc, seed: TABDDPM_SYN_DIR / f"synthetic_year_{loc}_tabddpm_seed{seed}.csv",
    },
}
MODEL_NAMES = tuple(MODEL_SPECS)
MODEL_LABELS = {name: spec["label"] for name, spec in MODEL_SPECS.items()}
MODEL_COLS = {name: spec["label"].replace("-", "_").replace(" ", "_") for name, spec in MODEL_SPECS.items()}
>>>>>>> ee9996e (added new codes/data used in adding new analysis)

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
<<<<<<< HEAD
            if model == "diffusion":
                path = DIFF_SYN_DIR / f"synthetic_location_{loc}_best_diffusion_model_seed{seed}.csv"
            else:
                path = CTGAN_SYN_DIR / f"synthetic_year_{loc}_d2_8h_seed{seed}.csv"
=======
            path = MODEL_SPECS[model]["path"](loc, seed)
>>>>>>> ee9996e (added new codes/data used in adding new analysis)

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


<<<<<<< HEAD
print("Evaluating Diffusion …")
diff_rows  = collect_rows("diffusion")
print("Evaluating CTGAN …")
ctgan_rows = collect_rows("ctgan")

all_rows = diff_rows + ctgan_rows
=======
all_rows = []
for model_name, spec in MODEL_SPECS.items():
    print(f"Evaluating {spec['label']} …")
    all_rows.extend(collect_rows(model_name))
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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


<<<<<<< HEAD
=======
def holm_adjust_pvalues(p_values: list[float]) -> np.ndarray:
    """Return Holm-adjusted p-values while preserving NaN positions."""
    p_array = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_array.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(~np.isnan(p_array))
    if len(valid_positions) == 0:
        return adjusted

    valid_p = p_array[valid_positions]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    multipliers = np.arange(len(sorted_p), 0, -1)
    sorted_adjusted = np.maximum.accumulate(sorted_p * multipliers)
    sorted_adjusted = np.minimum(sorted_adjusted, 1.0)

    adjusted_valid = np.empty_like(sorted_adjusted)
    adjusted_valid[order] = sorted_adjusted
    adjusted[valid_positions] = adjusted_valid
    return adjusted


>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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

<<<<<<< HEAD
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
=======
table3_rows = []
for category, metric_list in TABLE3_METRICS.items():
    for metric_label, col, direction in metric_list:
        p_value = paired_wilcoxon_pvalue(df, col)
        row = {
            "Category": category,
            "Metric": metric_label,
            "p_value": p_value,
            "better": direction,
        }
        for model_tag in MODEL_NAMES:
            vals = df[df["model"] == model_tag][col].dropna()
            row[MODEL_LABELS[model_tag]] = ms(vals) if len(vals) else "n/a"
        table3_rows.append(row)

table3_df = pd.DataFrame(table3_rows)
table3_df["p_value_holm"] = holm_adjust_pvalues(table3_df["p_value"].tolist())
model_table_cols = [MODEL_LABELS[model_tag] for model_tag in MODEL_NAMES]
table3_df = table3_df[
    ["Category", "Metric"] + model_table_cols + ["p_value", "p_value_holm", "better"]
]

header = (
    f"{'Category':<18}{'Metric':<22}"
    + "".join(f"{label:>22}" for label in model_table_cols)
    + f"{'raw p':>12}{'Holm p':>12}"
)
print(header)
print("-" * len(header))
previous_category = None
for _, row in table3_df.iterrows():
    cat_str = row["Category"] if row["Category"] != previous_category else ""
    model_parts = "".join(f"{row[label]:>22}" for label in model_table_cols)
    print(
        f"{cat_str:<18}{row['Metric']:<22}"
        + model_parts
        + f"{fmt_pvalue(row['p_value']):>12}{fmt_pvalue(row['p_value_holm']):>12}"
    )
    previous_category = row["Category"]

>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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
<<<<<<< HEAD
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
=======
table4_rows = []
table4_pvalue_cells = []
periods = (("Year", "year"), (STRONG_MONTH, STRONG_MONTH.lower()), (WEAK_MONTH, WEAK_MONTH.lower()))
for metric_label, (ycol, scol, wcol) in TABLE4_METRICS.items():
    row = {"Metric": metric_label}
    for model_tag in MODEL_NAMES:
        sub = df[df["model"] == model_tag]
        for col, (period_label, _) in zip((ycol, scol, wcol), periods):
            vals = sub[col].dropna()
            s = ms(vals) if len(vals) > 0 else "n/a"
            key = f"{MODEL_COLS[model_tag]} {period_label}"
            row[key] = s

    for (_, period_key), col in zip(periods, (ycol, scol, wcol)):
        p_value = paired_wilcoxon_pvalue(df, col)
        row[f"p_value_{period_key}"] = p_value
        table4_pvalue_cells.append((len(table4_rows), period_key, p_value))
    table4_rows.append(row)

table4_df = pd.DataFrame(table4_rows)
table4_adjusted = holm_adjust_pvalues([cell[2] for cell in table4_pvalue_cells])
for (row_idx, period_key, _), adjusted_p in zip(table4_pvalue_cells, table4_adjusted):
    table4_df.loc[row_idx, f"p_value_holm_{period_key}"] = adjusted_p
table4_df = table4_df[
    ["Metric"]
    + [
        f"{MODEL_COLS[model_tag]} {period_label}"
        for model_tag in MODEL_NAMES
        for period_label, _ in periods
    ]
    + [
        p_col
        for _, period_key in periods
        for p_col in (f"p_value_{period_key}", f"p_value_holm_{period_key}")
    ]
]

header4 = (
    f"{'Metric':<22}"
    + "".join(
        f"{(MODEL_LABELS[model_tag] + ' ' + period_label):>{col_w}}"
        for model_tag in MODEL_NAMES
        for period_label, _ in periods
    )
    + f"{'Holm p Year':>14}{'Holm p '+STRONG_MONTH:>16}{'Holm p '+WEAK_MONTH:>14}"
)
print(header4)
print("-" * len(header4))
for _, row in table4_df.iterrows():
    summary_parts = [
        row[f"{MODEL_COLS[model_tag]} {period_label}"]
        for model_tag in MODEL_NAMES
        for period_label, _ in periods
    ]
    p_parts = [fmt_pvalue(row[f"p_value_holm_{period_key}"]) for _, period_key in periods]
    print(
        f"{row['Metric']:<22}"
        + "".join(f"{part:>{col_w}}" for part in summary_parts)
        + f"{p_parts[0]:>14}{p_parts[1]:>16}{p_parts[2]:>14}"
    )

>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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
<<<<<<< HEAD
    for model_tag in ("diffusion", "ctgan"):
=======
    for model_tag in MODEL_NAMES:
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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

<<<<<<< HEAD
print(f"{'Location':<12}{'Sessions':>12}{'Diffusion':>22}{'CTGAN':>22}")
print("-" * 68)
=======
loc_header = f"{'Location':<12}{'Sessions':>12}" + "".join(
    f"{MODEL_LABELS[model_tag]:>22}" for model_tag in MODEL_NAMES
)
print(loc_header)
print("-" * len(loc_header))
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
for _, row in location_perf_df.iterrows():
    print(
        f"{int(row['location']):<12}"
        f"{int(row['session_count']):>12}"
<<<<<<< HEAD
        f"{row['diffusion_yearly_score']:>22}"
        f"{row['ctgan_yearly_score']:>22}"
=======
        + "".join(f"{row[f'{model_tag}_yearly_score']:>22}" for model_tag in MODEL_NAMES)
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
    )
print(f"\nSaved → {out_loc}")

regression_df = df.merge(location_counts, on="location", how="left")
regression_df["log10_session_count"] = np.log10(regression_df["session_count"])
regression_df["is_diffusion"] = (regression_df["model"] == "diffusion").astype(int)

regression_rows = []
<<<<<<< HEAD
for model_tag in ("diffusion", "ctgan"):
=======
for model_tag in MODEL_NAMES:
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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
<<<<<<< HEAD
=======
robustness_df["slope_p_value_holm"] = holm_adjust_pvalues(
    robustness_df["slope_p_value"].tolist()
)
robustness_df = robustness_df[
    [
        "model", "score", "slope_per_log10_session_count",
        "slope_p_value", "slope_p_value_holm", "r_squared", "n",
    ]
]
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
out_robust = BASE_DIR / "table8_count_score_regression.csv"
robustness_df.to_csv(out_robust, index=False)

print("\nCount-score regression: yearly_score ~ log10(session_count)")
<<<<<<< HEAD
print(f"{'Model':<24}{'Slope':>12}{'p-value':>12}{'R^2':>10}{'n':>6}")
print("-" * 64)
=======
print(f"{'Model':<24}{'Slope':>12}{'raw p':>12}{'Holm p':>12}{'R^2':>10}{'n':>6}")
print("-" * 76)
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
for _, row in robustness_df.iterrows():
    r2_str = "n/a" if np.isnan(row["r_squared"]) else f"{row['r_squared']:.3f}"
    print(
        f"{row['model']:<24}"
        f"{row['slope_per_log10_session_count']:>12.4f}"
        f"{fmt_pvalue(row['slope_p_value']):>12}"
<<<<<<< HEAD
=======
        f"{fmt_pvalue(row['slope_p_value_holm']):>12}"
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
        f"{r2_str:>10}"
        f"{int(row['n']):>6}"
    )
print(f"\nSaved → {out_robust}")

# ---------------------------------------------------------------------------
# Association / Frobenius helpers
# ---------------------------------------------------------------------------

ASSOC_COLS = ["plugin_hour", "plugout_hour", "connection_time", "energy_session"]
<<<<<<< HEAD
=======
PAIRWISE_COLS = [
    ("plugin_hour", "connection_time"),
    ("plugin_hour", "energy_session"),
    ("connection_time", "energy_session"),
]
PAIRWISE_BINS = {
    "plugin_hour": np.arange(-0.5, 24.5, 1.0),
    "connection_time": np.linspace(0.0, 120.0, 25),
    "energy_session": np.linspace(0.0, 150.0, 26),
}
>>>>>>> ee9996e (added new codes/data used in adding new analysis)


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


<<<<<<< HEAD
=======
def pairwise_2d_tv_complement(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> float:
    """2D histogram TV-complement on fixed bins; 1.0 means identical binned joint distributions."""
    real_xy = real_df[[x_col, y_col]].dropna().astype(float)
    synth_xy = synth_df[[x_col, y_col]].dropna().astype(float)
    if real_xy.empty or synth_xy.empty:
        return np.nan

    real_hist, _, _ = np.histogram2d(
        real_xy[x_col], real_xy[y_col], bins=[PAIRWISE_BINS[x_col], PAIRWISE_BINS[y_col]]
    )
    synth_hist, _, _ = np.histogram2d(
        synth_xy[x_col], synth_xy[y_col], bins=[PAIRWISE_BINS[x_col], PAIRWISE_BINS[y_col]]
    )
    real_total = real_hist.sum()
    synth_total = synth_hist.sum()
    if real_total == 0 or synth_total == 0:
        return np.nan

    real_prob = real_hist / real_total
    synth_prob = synth_hist / synth_total
    return float(1.0 - 0.5 * np.abs(real_prob - synth_prob).sum())


>>>>>>> ee9996e (added new codes/data used in adding new analysis)
# ---------------------------------------------------------------------------
# Load-profile helpers
# ---------------------------------------------------------------------------

<<<<<<< HEAD
def sessions_to_hourly_load(df: pd.DataFrame) -> pd.DataFrame:
=======
def sessions_to_hourly_load(
    df: pd.DataFrame,
    reconstruction: str = "uniform",
    charger_power_kw: float | None = None,
) -> pd.DataFrame:
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
    starts   = df["plugin_hour"].values.astype(float)
    durs     = np.clip(df["connection_time"].values.astype(float), 0.01, 24.0)
    energies = np.clip(df["energy_session"].values.astype(float), 0.0, np.inf)
    dates    = df["date"].values
<<<<<<< HEAD
    powers   = energies / durs

    records = []
    for i in range(len(df)):
        s, d, p, dt = starts[i], durs[i], powers[i], dates[i]
        end = s + d
=======

    records = []
    for i in range(len(df)):
        s, d, energy, dt = starts[i], durs[i], energies[i], dates[i]
        if reconstruction == "uniform":
            charge_duration = d
            power = energy / d
        elif reconstruction == "front_loaded_fixed_power":
            if charger_power_kw is None or charger_power_kw <= 0:
                raise ValueError("charger_power_kw must be positive for fixed-power reconstruction")
            charge_duration = min(d, energy / charger_power_kw)
            power = charger_power_kw
        else:
            raise ValueError(f"Unknown reconstruction: {reconstruction}")

        if charge_duration <= 0 or power <= 0:
            continue

        end = s + charge_duration
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
        h   = int(s)
        while h < end:
            ov = min(h + 1, end) - max(h, s)
            if ov > 0:
<<<<<<< HEAD
                records.append((dt, h % 24, p * ov))
            h += 1

    load = pd.DataFrame(records, columns=["date", "hour", "load_kw"])
    return load.groupby(["date", "hour"])["load_kw"].sum().reset_index()


def make_norm_grid(df: pd.DataFrame) -> pd.DataFrame:
    grid = (
        sessions_to_hourly_load(df)
=======
                records.append((dt, h % 24, power * ov))
            h += 1

    load = pd.DataFrame(records, columns=["date", "hour", "load_kw"])
    if load.empty:
        return load
    return load.groupby(["date", "hour"])["load_kw"].sum().reset_index()


def make_norm_grid(
    df: pd.DataFrame,
    reconstruction: str = "uniform",
    charger_power_kw: float | None = None,
) -> pd.DataFrame:
    grid = (
        sessions_to_hourly_load(df, reconstruction, charger_power_kw)
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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


<<<<<<< HEAD
=======
LOAD_RECONSTRUCTIONS = [
    {
        "key": "uniform",
        "label": "Uniform energy",
        "method": "uniform",
        "charger_power_kw": np.nan,
    },
    {
        "key": "front_loaded_3p7kw",
        "label": "Front-loaded 3.7 kW",
        "method": "front_loaded_fixed_power",
        "charger_power_kw": 3.7,
    },
    {
        "key": "front_loaded_7p4kw",
        "label": "Front-loaded 7.4 kW",
        "method": "front_loaded_fixed_power",
        "charger_power_kw": 7.4,
    },
    {
        "key": "front_loaded_11kw",
        "label": "Front-loaded 11 kW",
        "method": "front_loaded_fixed_power",
        "charger_power_kw": 11.0,
    },
]


>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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
<<<<<<< HEAD
        grid_real  = make_norm_grid(real_loc)

        for seed in SEEDS:
            if model == "diffusion":
                path = DIFF_SYN_DIR / f"synthetic_location_{loc}_best_diffusion_model_seed{seed}.csv"
            else:
                path = CTGAN_SYN_DIR / f"synthetic_year_{loc}_d2_8h_seed{seed}.csv"
=======
        real_grids = {
            spec["key"]: make_norm_grid(
                real_loc,
                reconstruction=spec["method"],
                charger_power_kw=None
                if np.isnan(spec["charger_power_kw"])
                else float(spec["charger_power_kw"]),
            )
            for spec in LOAD_RECONSTRUCTIONS
        }

        for seed in SEEDS:
            path = MODEL_SPECS[model]["path"](loc, seed)
>>>>>>> ee9996e (added new codes/data used in adding new analysis)

            if not path.exists():
                continue

            synth = filter_invalid_sessions(pd.read_csv(path))
            synth = ensure_plugout_hour(synth)
            synth["date"] = pd.to_datetime(synth["__date__"]).dt.date

            frob  = frob_norm(mat_real, assoc_matrix(synth))
<<<<<<< HEAD
            lp    = load_profile_metrics(make_norm_grid(synth), grid_real)
=======
            lp = {}
            for spec in LOAD_RECONSTRUCTIONS:
                charger_power = (
                    None
                    if np.isnan(spec["charger_power_kw"])
                    else float(spec["charger_power_kw"])
                )
                syn_grid = make_norm_grid(
                    synth,
                    reconstruction=spec["method"],
                    charger_power_kw=charger_power,
                )
                metrics = load_profile_metrics(syn_grid, real_grids[spec["key"]])
                for metric_key, metric_value in metrics.items():
                    lp[f"{spec['key']}_{metric_key}"] = metric_value

            lp.update(
                {
                    metric_key: lp[f"uniform_{metric_key}"]
                    for metric_key, _label, _better in [
                        ("rmse", "", ""),
                        ("ks_peak", "", ""),
                        ("cv_err", "", ""),
                        ("peak_rel_err", "", ""),
                    ]
                }
            )
            pairwise = {
                f"pairwise_{x_col}_vs_{y_col}_tvc": pairwise_2d_tv_complement(
                    real_loc, synth, x_col, y_col
                )
                for x_col, y_col in PAIRWISE_COLS
            }
>>>>>>> ee9996e (added new codes/data used in adding new analysis)

            rows.append({
                "model":    model,
                "location": loc,
                "seed":     seed,
                "frobenius": frob,
                **lp,
<<<<<<< HEAD
=======
                **pairwise,
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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

<<<<<<< HEAD
frob_d = sdf[sdf["model"] == "diffusion"]["frobenius"]
frob_c = sdf[sdf["model"] == "ctgan"]["frobenius"]
print(f"{'Metric':<30}{'Diffusion':>22}{'CTGAN':>22}")
print("-" * 74)
row5 = {"Metric": "Frobenius Norm",
        "Diffusion": ms(frob_d), "CTGAN": ms(frob_c), "better": "lower"}
print(f"{'Frobenius Norm':<30}{ms(frob_d):>22}{ms(frob_c):>22}")
=======
row5 = {"Metric": "Frobenius Norm", "better": "lower"}
for model_tag in MODEL_NAMES:
    row5[MODEL_LABELS[model_tag]] = ms(sdf[sdf["model"] == model_tag]["frobenius"])
header5 = f"{'Metric':<30}" + "".join(f"{MODEL_LABELS[model_tag]:>22}" for model_tag in MODEL_NAMES)
print(header5)
print("-" * len(header5))
print(
    f"{'Frobenius Norm':<30}"
    + "".join(f"{row5[MODEL_LABELS[model_tag]]:>22}" for model_tag in MODEL_NAMES)
)
>>>>>>> ee9996e (added new codes/data used in adding new analysis)

table5_df = pd.DataFrame([row5])
out5 = BASE_DIR / "table5_frobenius.csv"
table5_df.to_csv(out5, index=False)
print(f"\nSaved → {out5}")

<<<<<<< HEAD
=======
location_frobenius_rows = []
pooled_real = ensure_plugout_hour(real_all.copy())
pooled_real_mat = assoc_matrix(pooled_real)
max_frobenius_4var = float(np.sqrt(48.0))
for loc in LOCATIONS:
    real_loc = ensure_plugout_hour(real_all[real_all["location_group"] == loc].copy())
    if real_loc.empty:
        continue
    baseline = frob_norm(assoc_matrix(real_loc), pooled_real_mat)
    row = {
        "location": loc,
        "session_count": int(len(real_loc)),
        "real_vs_pooled_real_frobenius": baseline,
        "theoretical_max_frobenius": max_frobenius_4var,
    }
    for model_tag in MODEL_NAMES:
        vals = sdf[(sdf["model"] == model_tag) & (sdf["location"] == loc)]["frobenius"].dropna()
        row[f"{model_tag}_frobenius"] = ms(vals) if len(vals) else "n/a"
        row[f"{model_tag}_frobenius_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{model_tag}_frobenius_std"] = float(vals.std()) if len(vals) else np.nan
        row[f"{model_tag}_n_runs"] = int(len(vals))
        row[f"{model_tag}_share_of_theoretical_max"] = (
            float(vals.mean() / max_frobenius_4var) if len(vals) else np.nan
        )
    location_frobenius_rows.append(row)

location_frobenius_df = pd.DataFrame(location_frobenius_rows)
out5_loc = BASE_DIR / "table5_frobenius_by_location.csv"
location_frobenius_df.to_csv(out5_loc, index=False)

print("\nFrobenius norm by location group with real-data baseline")
loc_frob_header = (
    f"{'Location':<10}{'Sessions':>10}{'Real-pooled':>14}"
    + "".join(f"{MODEL_LABELS[model_tag]:>22}" for model_tag in MODEL_NAMES)
)
print(loc_frob_header)
print("-" * len(loc_frob_header))
for _, row in location_frobenius_df.iterrows():
    print(
        f"{int(row['location']):<10}{int(row['session_count']):>10}"
        f"{row['real_vs_pooled_real_frobenius']:>14.3f}"
        + "".join(f"{row[f'{model_tag}_frobenius']:>22}" for model_tag in MODEL_NAMES)
    )
print(f"\nSaved → {out5_loc}")

pairwise_rows = []
for x_col, y_col in PAIRWISE_COLS:
    col = f"pairwise_{x_col}_vs_{y_col}_tvc"
    p_value = paired_wilcoxon_pvalue(sdf, col)
    row = {
        "Pair": f"{x_col} vs {y_col}",
        "Metric": "2D TV Complement",
        "p_value_raw": p_value,
        "better": "higher",
    }
    for model_tag in MODEL_NAMES:
        vals = sdf[sdf["model"] == model_tag][col].dropna()
        row[MODEL_LABELS[model_tag]] = ms(vals) if len(vals) else "n/a"
    pairwise_rows.append(row)

pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df["p_value_holm"] = holm_adjust_pvalues(pairwise_df["p_value_raw"].tolist())
pairwise_df = pairwise_df[
    ["Pair", "Metric"] + model_table_cols + ["p_value_holm", "p_value_raw", "better"]
]
out_pairwise = BASE_DIR / "table9_pairwise_2d_similarity.csv"
pairwise_df.to_csv(out_pairwise, index=False)

print("\nPairwise 2D distribution similarity across all location groups")
pair_header = (
    f"{'Pair':<42}{'Metric':<20}"
    + "".join(f"{label:>22}" for label in model_table_cols)
    + f"{'Holm p':>12}"
)
print(pair_header)
print("-" * len(pair_header))
for _, row in pairwise_df.iterrows():
    print(
        f"{row['Pair']:<42}{row['Metric']:<20}"
        + "".join(f"{row[label]:>22}" for label in model_table_cols)
        + f"{fmt_pvalue(row['p_value_holm']):>12}"
    )
print(f"\nSaved → {out_pairwise}")

>>>>>>> ee9996e (added new codes/data used in adding new analysis)
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
<<<<<<< HEAD
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
=======

table6_rows = []
for col, label, better in LP_METRICS:
    p_value = paired_wilcoxon_pvalue(sdf, col)
    row = {"Metric": label, "p_value": p_value, "better": better}
    for model_tag in MODEL_NAMES:
        row[MODEL_LABELS[model_tag]] = ms(sdf[sdf["model"] == model_tag][col])
    table6_rows.append(row)

table6_df = pd.DataFrame(table6_rows)
table6_df["p_value_holm"] = holm_adjust_pvalues(table6_df["p_value"].tolist())
table6_df = table6_df.rename(
    columns={
        "p_value": "p_value_raw",
        "p_value_holm": "p_value",
    }
)
table6_df = table6_df[
    ["Metric"] + model_table_cols + ["p_value", "p_value_raw", "better"]
]

header6 = (
    f"{'Metric':<26}"
    + "".join(f"{label:>22}" for label in model_table_cols)
    + f"{'Holm p':>12}{'raw p':>12}"
)
print(header6)
print("-" * len(header6))
for _, row in table6_df.iterrows():
    print(
        f"{row['Metric']:<26}"
        + "".join(f"{row[label]:>22}" for label in model_table_cols)
        + f"{fmt_pvalue(row['p_value']):>12}{fmt_pvalue(row['p_value_raw']):>12}"
    )

>>>>>>> ee9996e (added new codes/data used in adding new analysis)
out6 = BASE_DIR / "table6_load_profile_metrics.csv"
table6_df.to_csv(out6, index=False)
print(f"\nSaved → {out6}")

<<<<<<< HEAD
=======
# ---------------------------------------------------------------------------
# TABLE 10 — Load-profile reconstruction sensitivity
# ---------------------------------------------------------------------------

print("\n" + "="*90)
print("TABLE 10 — Load-profile reconstruction sensitivity (mean ± std)")
print("="*90)

sensitivity_rows = []
for spec in LOAD_RECONSTRUCTIONS:
    for col, label, better in LP_METRICS:
        metric_col = f"{spec['key']}_{col}"
        p_value = paired_wilcoxon_pvalue(sdf, metric_col)
        row = {
            "Reconstruction": spec["label"],
            "Metric": label,
            "charger_power_kw": spec["charger_power_kw"],
            "p_value_raw": p_value,
            "better": better,
        }
        model_means = {}
        for model_tag in MODEL_NAMES:
            vals = sdf[sdf["model"] == model_tag][metric_col].dropna()
            row[MODEL_LABELS[model_tag]] = ms(vals) if len(vals) else "n/a"
            model_means[model_tag] = float(vals.mean()) if len(vals) else np.nan

        if better == "lower":
            available = {k: v for k, v in model_means.items() if not np.isnan(v)}
            row["best_model"] = min(available, key=available.get) if available else "n/a"
        else:
            available = {k: v for k, v in model_means.items() if not np.isnan(v)}
            row["best_model"] = max(available, key=available.get) if available else "n/a"
        sensitivity_rows.append(row)

sensitivity_df = pd.DataFrame(sensitivity_rows)
for reconstruction, idx in sensitivity_df.groupby("Reconstruction").groups.items():
    sensitivity_df.loc[idx, "p_value_holm"] = holm_adjust_pvalues(
        sensitivity_df.loc[idx, "p_value_raw"].tolist()
    )
sensitivity_df = sensitivity_df[
    [
        "Reconstruction",
        "Metric",
        "charger_power_kw",
        "Diffusion",
        "CTGAN",
        "TabDDPM",
        "best_model",
        "p_value_holm",
        "p_value_raw",
        "better",
    ]
]
out_sensitivity = BASE_DIR / "table10_load_profile_reconstruction_sensitivity.csv"
sensitivity_df.to_csv(out_sensitivity, index=False)

sens_header = (
    f"{'Reconstruction':<24}{'Metric':<22}"
    f"{'Diffusion':>20}{'CTGAN':>20}{'Best':>14}{'Holm p':>12}"
)
print(sens_header)
print("-" * len(sens_header))
for _, row in sensitivity_df.iterrows():
    print(
        f"{row['Reconstruction']:<24}{row['Metric']:<22}"
        f"{row['Diffusion']:>20}{row['CTGAN']:>20}"
        f"{row['best_model']:>14}{fmt_pvalue(row['p_value_holm']):>12}"
    )
print(f"\nSaved → {out_sensitivity}")

>>>>>>> ee9996e (added new codes/data used in adding new analysis)
print("\nDone.")
