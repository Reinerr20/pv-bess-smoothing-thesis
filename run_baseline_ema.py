# Baseline EMA Smoothing Script (Final)
# ----------------------------------
# Scenario 1: Baseline smoothing (no SoC, no limits)
# EMA with alpha equivalent to SMA window: alpha = 2 / (N + 1)

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# CONFIG
# =========================
PV_CSV_PATH = "Data PV.csv"
RECOMMENDED_DAYS_PATH = "recommended_days.csv"

OUT_DIR = "outputs_baseline_ema"
os.makedirs(OUT_DIR, exist_ok=True)

# NEW: per-run profiles directory for Level-2 runner
PROFILES_DIR = os.path.join(OUT_DIR, "baseline_profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

INVERTER_1_COL = "inverter_1_ac_power_(kw)_inv_150143"
INVERTER_2_COL = "inverter_2_ac_power_(kw)_inv_150144"
TIME_COL = "measured_on"

DT_MIN = 5
DT_HOURS = DT_MIN / 60.0

# SMA-equivalent windows (EMA alpha derived from these)
EMA_WINDOWS = {
    "10min": 2,
    "20min": 4,
    "30min": 6,
    "60min": 12,
}

# windows shown in time-series plot
PLOT_WINDOWS = ["10min", "30min", "60min"]

# =========================
# DATA LOADING & PREPROCESS
# =========================

def load_and_preprocess_pv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    for col in [INVERTER_1_COL, INVERTER_2_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    df = df.set_index(TIME_COL).sort_index()
    df = df.groupby(df.index).mean(numeric_only=True)

    df["Ppv_kw"] = df[INVERTER_1_COL].fillna(0) + df[INVERTER_2_COL].fillna(0)

    return df[["Ppv_kw"]]


def load_recommended_days(path: str) -> pd.DataFrame:
    rec = pd.read_csv(path)
    rec["date"] = pd.to_datetime(rec["date"]).dt.date

    rec = rec[rec["label"].isin(["clear", "medium", "cloudy"])].copy()

    order = {"clear": 0, "medium": 1, "cloudy": 2}
    rec["label_order"] = rec["label"].map(order)
    rec = rec.sort_values(["label_order", "date"]).drop(columns="label_order")

    return rec[["label", "date"]]


def slice_one_day(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def active_window_mask(pv_day: pd.Series) -> pd.Series:
    pmax = pv_day.max()
    if pd.isna(pmax) or pmax <= 0:
        return pv_day * 0 > 1
    return pv_day > 0.05 * pmax


# =========================
# CORE COMPUTATION
# =========================

def ema_alpha_from_N(N: int) -> float:
    return 2.0 / (N + 1)


def compute_ema(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False, min_periods=int(round(2 / alpha - 1))).mean()


def ramp_p95_kw(psmooth: pd.Series) -> float:
    dP = psmooth.diff().abs().dropna()
    return float(dP.quantile(0.95)) if not dP.empty else np.nan


def throughput_kwh(pbatt: pd.Series) -> float:
    x = pbatt.abs().dropna()
    return float(x.sum() * DT_HOURS) if not x.empty else np.nan


# =========================
# PLOTTING
# =========================

def plot_raw_vs_smoothed(day_df: pd.DataFrame, label: str, day_str: str, smoothed: Dict[str, pd.Series]):
    fig, ax = plt.subplots()

    ax.plot(day_df.index, day_df["Ppv_kw"], label="Ppv_raw")

    for w in PLOT_WINDOWS:
        if w in smoothed:
            ax.plot(day_df.index, smoothed[w], label=f"EMA_{w}")

    ax.set_title(f"Raw vs Smoothed (EMA) — {label} — {day_str}")
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Power (kW)")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.autofmt_xdate()
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"raw_vs_smoothed_EMA_{label}_{day_str}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)


# =========================
# NEW: EXPORT PROFILES (for Level-2)
# =========================

def save_profile_for_level2(
    day_index: pd.DatetimeIndex,
    pv_day: pd.Series,
    psmooth: pd.Series,
    mask_active: pd.Series,
    label: str,
    day_str: str,
    window_name: str,
):
    """Save 1 CSV per run: timestamp, P_pv_kw, P_smooth_kw, is_active."""
    prof = pd.DataFrame({
        "timestamp": day_index,
        "P_pv_kw": pv_day.values,
        "P_smooth_kw": psmooth.values,
        "is_active": mask_active.astype(int).values,
    })
    fname = f"EMA_{label}_{day_str}_{window_name}.csv"
    prof.to_csv(os.path.join(PROFILES_DIR, fname), index=False)


def plot_tradeoff(df: pd.DataFrame):
    fig, ax = plt.subplots()

    for (label, date), g in df.groupby(["label", "date"]):
        ax.scatter(g["ramp_p95_kw"], g["throughput_kwh"], label=f"{label} ({date})")
        for _, r in g.iterrows():
            ax.annotate(r["window_name"], (r["ramp_p95_kw"], r["throughput_kwh"]))

    ax.set_title("Trade-off (EMA): RampP95 vs Throughput")
    ax.set_xlabel("RampP95 of Psmooth (kW / 5 min)")
    ax.set_ylabel("Battery Throughput (kWh)")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "tradeoff_EMA_rampP95_vs_throughput.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)


# =========================
# MAIN
# =========================

def main():
    pv = load_and_preprocess_pv(PV_CSV_PATH)
    rec = load_recommended_days(RECOMMENDED_DAYS_PATH)

    rows: List[Dict] = []

    for _, r in rec.iterrows():
        label = r["label"]
        day = r["date"]
        day_ts = pd.Timestamp(day)
        day_str = day_ts.strftime("%Y-%m-%d")

        day_df = slice_one_day(pv, day_ts)
        if day_df.empty:
            continue

        pv_day = day_df["Ppv_kw"]
        mask = active_window_mask(pv_day)

        smoothed_map = {}

        for name, N in EMA_WINDOWS.items():
            alpha = ema_alpha_from_N(N)
            ps = compute_ema(pv_day, alpha)
            smoothed_map[name] = ps

            pbatt_req = ps - pv_day

            ps_eval = ps[mask].dropna()
            pb_eval = pbatt_req[mask].dropna()

            rows.append({
                "label": label,
                "date": day_str,
                "method": "EMA",
                "param_name": "alpha",
                "param_value": alpha,
                "window_name": name,
                "ramp_p95_kw": ramp_p95_kw(ps_eval),
                "throughput_kwh": throughput_kwh(pb_eval),
                "energy_day_kwh_active_window": float(pv_day[mask].sum() * DT_HOURS),
                "n_points_active_window": int(mask.sum()),
            })

            # NEW: export profile for Level-2 (1 file per run)
            save_profile_for_level2(
                day_index=day_df.index,
                pv_day=pv_day,
                psmooth=ps,
                mask_active=mask,
                label=label,
                day_str=day_str,
                window_name=name,
            )

        plot_raw_vs_smoothed(day_df, label, day_str, smoothed_map)

    results = pd.DataFrame(rows).sort_values(["label", "param_value"])

    out_csv = os.path.join(OUT_DIR, "smoothing_results_ema.csv")
    results.to_csv(out_csv, index=False)

    if not results.empty:
        plot_tradeoff(results)

    print(f"Baseline EMA results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
