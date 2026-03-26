import argparse
from dataclasses import dataclass

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    input_path: str = "Data PV.csv"
    time_col: str = "measured_on"
    freq: str = "5min"                 # resolusi asli data
    day_start: str = "08:00"           # dipakai kalau mode fixed window
    day_end: str = "16:00"
    use_threshold_window: bool = True  # recommended
    threshold_frac: float = 0.05       # 5% dari puncak harian
    max_missing_frac: float = 0.02     # 2% missing pada window -> drop hari itu (kecuali outage case)
    min_points_window: int = 60        # minimal titik pada window (5min => 60 pts = 5 jam)
    output_daily_csv: str = "daily_indicators.csv"
    output_recommendation_csv: str = "recommended_days.csv"


def find_inverter_level_ac_power_cols(df: pd.DataFrame) -> List[str]:
    """
    Cari kolom inverter-level AC power (kW), hindari module-level.
    Heuristik:
    - mengandung 'inverter_' dan 'ac_power'
    - TIDAK mengandung 'module'
    """
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if ("inverter_" in cl) and ("ac_power" in cl) and ("module" not in cl):
            cols.append(c)
    return cols


def longest_true_streak(mask: np.ndarray) -> int:
    """Cari streak True terpanjang."""
    if mask.size == 0:
        return 0
    # hitung run length
    max_len = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return int(max_len)


def compute_daily_indicators(
    P: pd.Series,
    cfg: Config
) -> pd.DataFrame:
    """
    Hitung indikator harian:
    - E_day_kWh (pada window siang)
    - P95_abs_ramp_kW (pada window siang)
    - missing_frac_window
    - zero_streak_max_pts_window
    - n_points_window
    Window siang: threshold-based (recommended) atau fixed time.
    """
    # Pastikan index datetime dan freq rapi
    P = P.sort_index()

    # Reindex ke grid 5-min (biar missing terdeteksi jelas)
    full_idx = pd.date_range(P.index.min().floor("D"), P.index.max().ceil("D"), freq=cfg.freq)
    P = P.reindex(full_idx)

    dt_h = pd.Timedelta(cfg.freq).total_seconds() / 3600.0

    rows = []
    for day, g in P.groupby(P.index.date):
        s = g.copy()
        s.index = pd.to_datetime(s.index)

        # Definisikan window siang
        if cfg.use_threshold_window:
            # threshold = frac * puncak harian (abaikan NaN)
            day_max = np.nanmax(s.values) if np.isfinite(np.nanmax(s.values)) else np.nan
            if not np.isfinite(day_max) or day_max <= 0:
                continue  # hari kosong
            thr = cfg.threshold_frac * day_max
            win = s[s > thr]
        else:
            start = pd.to_datetime(f"{day} {cfg.day_start}")
            end = pd.to_datetime(f"{day} {cfg.day_end}")
            win = s[(s.index >= start) & (s.index <= end)]

        n_total = len(win)
        if n_total == 0:
            continue

        missing_frac = win.isna().mean()
        n_valid = win.notna().sum()

        # Kalau valid terlalu sedikit, skip
        if n_total < cfg.min_points_window:
            # window terlalu pendek -> skip
            continue

        # Hitung metrik hanya pada data valid
        win_valid = win.dropna()
        if len(win_valid) < cfg.min_points_window:
            continue

        # Energy (kWh) = sum(P_kW * dt_h)
        E_day_kWh = float((win_valid.clip(lower=0) * dt_h).sum())

        # Ramp P95 (kW per 5 min) dari smoothness metric
        ramp = win_valid.diff()
        P95_abs_ramp = float(np.nanpercentile(np.abs(ramp.values), 95))

        # Zero streak (untuk outage detection) pada window fixed / threshold
        # Definisi "zero" di sini = <= 0.01*max_hari agar robust noise
        day_max2 = np.nanmax(s.values) if np.isfinite(np.nanmax(s.values)) else 0.0
        zero_thr = 0.01 * day_max2
        # Buat mask berdasarkan win (bukan win_valid) supaya missing ikut kebaca terpisah
        win_for_streak = win.copy()
        zero_mask = (win_for_streak.fillna(-999) <= zero_thr).to_numpy()
        zero_streak_pts = longest_true_streak(zero_mask)

        rows.append({
            "date": pd.to_datetime(day),
            "E_day_kWh": E_day_kWh,
            "P95_abs_ramp_kW": P95_abs_ramp,
            "missing_frac_window": float(missing_frac),
            "n_points_window": int(n_total),
            "n_valid_points_window": int(n_valid),
            "zero_streak_max_pts_window": int(zero_streak_pts),
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def pick_representative_days(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Pilih:
    - clear day: energy tinggi, ramp rendah
    - cloudy day: ramp tinggi (dengan energy tidak terlalu kecil)
    - medium day: skor di tengah
    + optional: outage day: zero streak terbesar (di window)
    """
    d = daily.copy()

    # Filter kualitas data untuk 3 hari representatif
    # (outage dipilih terpisah, jadi di sini kita bersihkan dulu)
    d_good = d[(d["missing_frac_window"] <= cfg.max_missing_frac)].copy()
    if len(d_good) < 3:
        # fallback: pakai semua (kalau dataset sample kecil)
        d_good = d.copy()

    # Normalisasi untuk scoring
    # clear: E tinggi, ramp rendah
    E = d_good["E_day_kWh"].to_numpy()
    R = d_good["P95_abs_ramp_kW"].to_numpy()

    E_n = (E - np.nanmin(E)) / (np.nanmax(E) - np.nanmin(E) + 1e-9)
    R_n = (R - np.nanmin(R)) / (np.nanmax(R) - np.nanmin(R) + 1e-9)

    clear_score = E_n - R_n                 # makin besar makin clear
    cloudy_score = R_n + 0.2 * (1 - E_n)    # ramp tinggi, tapi jangan energy terlalu kecil

    d_good["clear_score"] = clear_score
    d_good["cloudy_score"] = cloudy_score

    # pick clear
    clear_day = d_good.sort_values("clear_score", ascending=False).head(1)

    # pick cloudy (pastikan tidak sama tanggal)
    cloudy_day = d_good[~d_good["date"].isin(clear_day["date"])].sort_values("cloudy_score", ascending=False).head(1)

    # pick medium: dekat median ramp & median energy (jarak euclidean kecil)
    medE = np.nanmedian(E_n)
    medR = np.nanmedian(R_n)
    d_good["mid_dist"] = np.sqrt((E_n - medE)**2 + (R_n - medR)**2)
    medium_day = d_good[~d_good["date"].isin(pd.concat([clear_day["date"], cloudy_day["date"]]))].sort_values("mid_dist").head(1)

    # outage day (optional): zero streak terbesar, tapi pastikan bukan salah satu dari 3 hari
    outage_day = daily.copy()
    outage_day = outage_day[~outage_day["date"].isin(pd.concat([clear_day["date"], cloudy_day["date"], medium_day["date"]]))]
    outage_day = outage_day.sort_values("zero_streak_max_pts_window", ascending=False).head(1)

    rec = []
    if len(clear_day) == 1:
        rec.append(("clear", clear_day.iloc[0]))
    if len(cloudy_day) == 1:
        rec.append(("cloudy", cloudy_day.iloc[0]))
    if len(medium_day) == 1:
        rec.append(("medium", medium_day.iloc[0]))
    if len(outage_day) == 1 and outage_day.iloc[0]["zero_streak_max_pts_window"] > 0:
        rec.append(("outage_optional", outage_day.iloc[0]))

    rec_df = pd.DataFrame([{
        "label": lbl,
        **row.to_dict()
    } for lbl, row in rec])

    # rapihin kolom
    cols_front = ["label", "date", "E_day_kWh", "P95_abs_ramp_kW", "missing_frac_window", "zero_streak_max_pts_window"]
    cols_rest = [c for c in rec_df.columns if c not in cols_front]
    rec_df = rec_df[cols_front + cols_rest].sort_values("label")
    return rec_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data PV.csv", help="Path file excel/csv")
    ap.add_argument("--time_col", default="measured_on", help="Nama kolom timestamp")
    args = ap.parse_args()

    cfg = Config(input_path=args.input, time_col=args.time_col)

    # Load data
    if cfg.input_path.lower().endswith(".csv"):
        df = pd.read_csv(cfg.input_path, low_memory=False)
    else:
        df = pd.read_excel(cfg.input_path)

    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).set_index(cfg.time_col)

    inv_cols = find_inverter_level_ac_power_cols(df)
    if len(inv_cols) == 0:
        raise ValueError("Tidak ketemu kolom inverter-level AC power. Cek nama kolom (harus mengandung 'inverter_' dan 'ac_power', bukan 'module').")


    print("Detected inverter-level AC power columns:")
    for c in inv_cols:
        print(" -", c)

    # Build PV total (kW)
    P = df[inv_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0).sum(axis=1)
    P.name = "P_pv_total_kW"

    # handle duplicate timestamps
    P = P.groupby(P.index).mean()

    # Compute daily indicators
    daily = compute_daily_indicators(P, cfg)
    if daily.empty:
        raise RuntimeError("Daily indicators kosong. Cek apakah datanya benar, timestamp OK, dan ada power > 0.")

    daily.to_csv(cfg.output_daily_csv, index=False)
    print(f"\nSaved daily indicators to: {cfg.output_daily_csv}")

    # Pick representative days
    rec = pick_representative_days(daily, cfg)
    rec.to_csv(cfg.output_recommendation_csv, index=False)
    print(f"Saved recommended days to: {cfg.output_recommendation_csv}")

    print("\n=== Recommended Days ===")
    print(rec[["label", "date", "E_day_kWh", "P95_abs_ramp_kW", "missing_frac_window", "zero_streak_max_pts_window"]].to_string(index=False))


if __name__ == "__main__":
    main()
