import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PV_CSV_PATH = "Data PV.csv"
RECOMMENDED_DAYS_PATH = "recommended_days.csv"

INVERTER_1_COL = "inverter_1_ac_power_(kw)_inv_150143"
INVERTER_2_COL = "inverter_2_ac_power_(kw)_inv_150144"
TIME_COL = "measured_on"

WINDOWS = {
    "10min": 2,
    "20min": 4,
    "30min": 6,
    "60min": 12,
}

OUT_DIR = "outputs_compare_sma_ema"
os.makedirs(OUT_DIR, exist_ok=True)


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
    rec["date"] = pd.to_datetime(rec["date"], errors="coerce").dt.date
    rec = rec.dropna(subset=["date"])
    rec = rec[rec["label"].isin(["clear", "medium", "cloudy"])].copy()
    return rec[["label", "date"]]


def slice_one_day(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def active_window_mask(pv_day: pd.Series, threshold_ratio: float = 0.05) -> pd.Series:
    pmax = pv_day.max()
    if pd.isna(pmax) or pmax <= 0:
        return pv_day * 0 > 1
    return pv_day > (threshold_ratio * pmax)


def compute_sma(series: pd.Series, N: int) -> pd.Series:
    return series.rolling(window=N, min_periods=N).mean()


def compute_ema(series: pd.Series, N: int) -> pd.Series:
    alpha = 2.0 / (N + 1.0)
    return series.ewm(alpha=alpha, adjust=False, min_periods=N).mean()


def plot_one_day_4panel(day_df: pd.DataFrame, label: str, day_str: str, threshold_ratio: float = 0.05):
    pv = day_df["Ppv_kw"]
    mask = active_window_mask(pv, threshold_ratio=threshold_ratio)
    if not mask.any():
        raise RuntimeError(f"Active window empty for {label} {day_str}")

    pv_plot = pv.loc[mask]

    sma_map = {}
    ema_map = {}
    pbatt_sma_map = {}
    pbatt_ema_map = {}

    for wname, N in WINDOWS.items():
        ps_sma = compute_sma(pv, N).loc[mask]
        ps_ema = compute_ema(pv, N).loc[mask]

        sma_map[wname] = ps_sma
        ema_map[wname] = ps_ema
        pbatt_sma_map[wname] = ps_sma - pv_plot
        pbatt_ema_map[wname] = ps_ema - pv_plot

    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    # (a) Raw + SMA
    axs[0].plot(pv_plot.index, pv_plot.values, label="Ppv_raw", color="black", linewidth=1.2, alpha=0.7)
    for wname, s in sma_map.items():
        axs[0].plot(s.index, s.values, label=f"SMA_{wname}", linewidth=1.5)
    axs[0].set_ylabel("Power (kW)")
    axs[0].set_title(f"Level-1 baseline smoothing — {label} — {day_str}")

    # (b) Pbatt_req SMA
    for wname, s in pbatt_sma_map.items():
        axs[1].plot(s.index, s.values, label=f"SMA_{wname}", linewidth=1.5)
    axs[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axs[1].set_ylabel("$P_{batt,req}$ (kW)")

    # (c) Raw + EMA
    axs[2].plot(pv_plot.index, pv_plot.values, label="Ppv_raw", color="black", linewidth=1.2, alpha=0.7)
    for wname, s in ema_map.items():
        axs[2].plot(s.index, s.values, label=f"EMA_{wname}", linewidth=1.5)
    axs[2].set_ylabel("Power (kW)")

    # (d) Pbatt_req EMA
    for wname, s in pbatt_ema_map.items():
        axs[3].plot(s.index, s.values, label=f"EMA_{wname}", linewidth=1.5)
    axs[3].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axs[3].set_ylabel("$P_{batt,req}$ (kW)")
    axs[3].set_xlabel("Time (HH:MM)")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.autofmt_xdate()
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"fig_level1_temporal_{label}_{day_str}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved plot: {out}")


def main():
    pv = load_and_preprocess_pv(PV_CSV_PATH)
    rec = load_recommended_days(RECOMMENDED_DAYS_PATH)

    for _, row in rec.iterrows():
        label = str(row["label"])
        day_ts = pd.Timestamp(row["date"])
        day_str = day_ts.strftime("%Y-%m-%d")

        day_df = slice_one_day(pv, day_ts)
        if day_df.empty:
            print(f"[SKIP] No data for {label} {day_str}")
            continue

        plot_one_day_4panel(day_df, label=label, day_str=day_str, threshold_ratio=0.05)


if __name__ == "__main__":
    main()