# plot_compare_sma_ema_active_window.py
# ---------------------------------------------------
# Plot 1-day example: EMA (red) vs SMA (blue) with ACTIVE WINDOW
# Data source & column names follow your baseline SMA script.
# Author: Reiner (helper script)

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# CONFIG (match your SMA script)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PV_CSV_PATH = os.path.join(BASE_DIR, "Data PV.csv")
RECOMMENDED_DAYS_PATH = os.path.join(BASE_DIR, "recommended_days.csv")

INVERTER_1_COL = "inverter_1_ac_power_(kw)_inv_150143"
INVERTER_2_COL = "inverter_2_ac_power_(kw)_inv_150144"
TIME_COL = "measured_on"

WINDOWS = {
    "10min": 2,
    "20min": 4,
    "30min": 6,
    "60min": 12,
}

OUT_DIR = os.path.join(BASE_DIR, "outputs_compare_sma_ema")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def load_and_preprocess_pv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PV CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if TIME_COL not in df.columns:
        raise KeyError(f"Missing TIME_COL='{TIME_COL}' in PV CSV columns. Found: {df.columns.tolist()}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    for col in [INVERTER_1_COL, INVERTER_2_COL]:
        if col not in df.columns:
            raise KeyError(f"Missing inverter column '{col}' in PV CSV columns.")
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    df = df.set_index(TIME_COL).sort_index()

    # handle duplicate timestamps (same as your SMA script)
    df = df.groupby(df.index).mean(numeric_only=True)

    df["Ppv_kw"] = df[INVERTER_1_COL].fillna(0) + df[INVERTER_2_COL].fillna(0)

    return df[["Ppv_kw"]]


def load_recommended_days(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"recommended_days.csv not found: {path}")

    rec = pd.read_csv(path)
    if "label" not in rec.columns or "date" not in rec.columns:
        raise KeyError(f"recommended_days.csv must have columns ['label','date'], got: {rec.columns.tolist()}")

    rec["label"] = rec["label"].astype(str)
    rec["date"] = pd.to_datetime(rec["date"], errors="coerce").dt.date
    rec = rec.dropna(subset=["date"])
    return rec[["label", "date"]]


def slice_one_day(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def active_window_mask(pv_day: pd.Series, threshold_ratio: float = 0.05) -> pd.Series:
    """
    Active window = PV > threshold_ratio * Pmax(day).
    Default threshold_ratio=0.05 matches your SMA baseline script.
    """
    pmax = pv_day.max()
    if pd.isna(pmax) or pmax <= 0:
        return pv_day * 0 > 1  # all False
    return pv_day > (threshold_ratio * pmax)


def compute_sma(series: pd.Series, N: int) -> pd.Series:
    return series.rolling(window=N, min_periods=N).mean()


def compute_ema(series: pd.Series, N: int) -> pd.Series:
    # EMA alpha commonly used: 2/(N+1)
    alpha = 2.0 / (N + 1.0)
    # min_periods=N so SMA vs EMA start validity is comparable
    return series.ewm(alpha=alpha, adjust=False, min_periods=N).mean()


# =========================
# PLOT
# =========================
def plot_one_day_compare_active(
    day_df: pd.DataFrame,
    label: str,
    day_str: str,
    window_name: str,
    show_raw: bool,
    threshold_ratio: float,
):
    if window_name not in WINDOWS:
        raise ValueError(f"window must be one of {list(WINDOWS.keys())}")

    N = WINDOWS[window_name]
    pv = day_df["Ppv_kw"]

    mask = active_window_mask(pv, threshold_ratio=threshold_ratio)

    # compute on full day (keeps filter behavior consistent), then mask for plotting
    ps_sma_full = compute_sma(pv, N)
    ps_ema_full = compute_ema(pv, N)

    # apply mask for plotting
    day_df_plot = day_df.loc[mask].copy()
    if day_df_plot.empty:
        raise RuntimeError(
            f"Active window is empty for {day_str}. "
            f"Try lowering threshold_ratio (current={threshold_ratio})."
        )

    pv_plot = pv.loc[mask]
    ps_sma = ps_sma_full.loc[mask]
    ps_ema = ps_ema_full.loc[mask]

    fig, ax = plt.subplots(figsize=(10, 4))

    if show_raw:
        ax.plot(day_df_plot.index, pv_plot, label="Ppv_raw", linewidth=1.0, alpha=0.5)

    ax.plot(day_df_plot.index, ps_ema, label=f"EMA_{window_name}", color="red", linewidth=2.0)
    ax.plot(day_df_plot.index, ps_sma, label=f"SMA_{window_name}", color="blue", linewidth=2.0)

    thr_pct = int(round(threshold_ratio * 100))
    ax.set_title(f"SMA vs EMA (window={window_name}, active>{thr_pct}% Pmax) — {label} — {day_str}")
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Power (kW)")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"compare_SMA_vs_EMA_{label}_{day_str}_{window_name}_active{thr_pct}.png")
    fig.savefig(out, dpi=250)
    plt.close(fig)

    print(f"[OK] Saved plot: {out}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="cloudy", help="cloudy / normal / medium (must match recommended_days.csv)")
    parser.add_argument("--window", type=str, default="30min", help="10min / 20min / 30min / 60min")
    parser.add_argument("--show_raw", action="store_true", help="overlay raw PV")
    parser.add_argument("--date", type=str, default="", help="override date, e.g. 2024-04-03 (recommended)")
    parser.add_argument("--thr", type=float, default=0.05, help="active window threshold ratio, default 0.05 (5%% Pmax)")
    args = parser.parse_args()

    pv = load_and_preprocess_pv(PV_CSV_PATH)

    # Prefer date override (more deterministic for PPT)
    if args.date.strip():
        day_ts = pd.Timestamp(args.date.strip())
        label = args.label
    else:
        rec = load_recommended_days(RECOMMENDED_DAYS_PATH)
        rec["label_lc"] = rec["label"].str.lower().str.strip()
        target = args.label.lower().strip()
        match = rec[rec["label_lc"] == target]
        if match.empty:
            raise ValueError(f"Label '{args.label}' not found in recommended_days.csv")
        row = match.iloc[0]
        label = row["label"]
        day_ts = pd.Timestamp(row["date"])

    day_str = day_ts.strftime("%Y-%m-%d")
    day_df = slice_one_day(pv, day_ts)
    if day_df.empty:
        raise RuntimeError(f"No PV data found for date {day_str}. Check CSV timestamps / timezone.")

    plot_one_day_compare_active(
        day_df=day_df,
        label=str(label),
        day_str=day_str,
        window_name=str(args.window),
        show_raw=bool(args.show_raw),
        threshold_ratio=float(args.thr),
    )


if __name__ == "__main__":
    main()