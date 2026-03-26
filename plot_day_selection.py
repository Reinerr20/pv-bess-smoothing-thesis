import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ====== CONFIG ======
DATA_FILE = "Data PV.csv"
RECOMMENDED_FILE = "recommended_days.csv"
DAILY_FILE = "daily_indicators.csv"

TIME_COL = "measured_on"
INV1 = "inverter_1_ac_power_(kw)_inv_150143"
INV2 = "inverter_2_ac_power_(kw)_inv_150144"

FREQ_MIN = 5
DT_H = FREQ_MIN / 60.0

# plot settings
THRESH_FRAC = 0.05  # window siang: PV > 5% max harian


def load_pv_total() -> pd.Series:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).set_index(TIME_COL)

    pv = df[[INV1, INV2]].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    P = pv.sum(axis=1)
    P.name = "P_pv_total_kW"

    # robust: handle duplicate timestamps
    P = P.groupby(P.index).mean()
    return P


def get_day_series(P_all: pd.Series, day_str: str) -> pd.Series:
    day_start = pd.to_datetime(day_str)
    day_end = day_start + pd.Timedelta(days=1)
    P_day = P_all[(P_all.index >= day_start) & (P_all.index < day_end)].copy()
    return P_day


def apply_day_window(P_day: pd.Series) -> pd.Series:
    # threshold-based window (siang): PV > 5% max
    if P_day.empty:
        return P_day
    day_max = P_day.max()
    if not np.isfinite(day_max) or day_max <= 0:
        return P_day
    thr = THRESH_FRAC * day_max
    P_win = P_day[P_day > thr].copy()
    # fallback kalau terlalu sedikit titik
    if len(P_win) < 60:  # ~5 jam
        return P_day
    return P_win


def plot_timeseries_three_days(P_all: pd.Series, rec: pd.DataFrame):
    # urutkan label biar rapi
    order = ["clear", "cloudy", "medium"]
    rec = rec.set_index("label").loc[order].reset_index()

    plt.figure(figsize=(10, 8))

    for i, row in enumerate(rec.itertuples(index=False), start=1):
        label = row.label
        day = str(pd.to_datetime(row.date).date())

        P_day = get_day_series(P_all, day)
        P_win = apply_day_window(P_day)

        ax = plt.subplot(3, 1, i)
        ax.plot(P_win.index, P_win.values)
        ax.set_title(f"Raw PV Power — {label.upper()} day ({day})")
        ax.set_ylabel("kW")
        if i == 3:
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_selection_timeseries.png", dpi=200)
    plt.close()
    print("Saved: fig_selection_timeseries.png")


def plot_hist_abs_ramp(rec: pd.DataFrame, P_all: pd.Series):
    # histogram |ΔP| untuk 3 hari
    order = ["clear", "cloudy", "medium"]
    rec = rec.set_index("label").loc[order].reset_index()

    plt.figure(figsize=(10, 5))

    for row in rec.itertuples(index=False):
        label = row.label
        day = str(pd.to_datetime(row.date).date())

        P_day = get_day_series(P_all, day)
        P_win = apply_day_window(P_day)

        dP = P_win.diff().dropna()
        abs_dP = np.abs(dP.values)

        # plot histogram normalized (density)
        plt.hist(abs_dP, bins=60, density=True, alpha=0.5, label=f"{label} ({day})")

    plt.xlabel(f"|ΔP| (kW per {FREQ_MIN} min)")
    plt.ylabel("Density")
    plt.title("Ramp Distribution Proof — Histogram of |ΔP|")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_selection_hist_abs_ramp.png", dpi=200)
    plt.close()
    print("Saved: fig_selection_hist_abs_ramp.png")


def plot_scatter_all_days_with_highlight():
    daily = pd.read_csv(DAILY_FILE)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    rec = pd.read_csv(RECOMMENDED_FILE)
    rec["date"] = pd.to_datetime(rec["date"], errors="coerce")

    plt.figure(figsize=(7, 6))
    # all days
    plt.scatter(daily["P95_abs_ramp_kW"], daily["E_day_kWh"], alpha=0.4, label="All days")

    # highlight selected
    for row in rec.itertuples(index=False):
        if row.label not in ["clear", "cloudy", "medium"]:
            continue
        plt.scatter([row.P95_abs_ramp_kW], [row.E_day_kWh], s=120, label=f"{row.label} ({str(row.date.date())})")

    plt.xlabel("P95(|ΔP|) (kW per 5 min)")
    plt.ylabel("Energy in active window (kWh)")
    plt.title("All Days Map — Energy vs Ramp (Selected Days Highlighted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_selection_scatter_all_days.png", dpi=200)
    plt.close()
    print("Saved: fig_selection_scatter_all_days.png")


def main():
    # load recommendation
    rec = pd.read_csv(RECOMMENDED_FILE)
    # keep only the main 3 labels
    rec = rec[rec["label"].isin(["clear", "cloudy", "medium"])].copy()

    # load PV total once
    P_all = load_pv_total()

    # 1) timeseries proof
    plot_timeseries_three_days(P_all, rec)

    # 2) histogram |dP|
    plot_hist_abs_ramp(rec, P_all)

    # 3) scatter all days + highlight
    plot_scatter_all_days_with_highlight()

    print("\nDONE. 3 figures generated for day-selection proof.")


if __name__ == "__main__":
    main()
