from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# =========================
# 0) CONFIG
# =========================
FEASIBILITY_PROFILE_DIR = Path("outputs_level2_feasibility") / "feasibility_profiles"
FEASIBILITY_SUMMARY_CSV = Path("outputs_level2_feasibility") / "level2_feasibility_results.csv"

OUT_DIR = Path("outputs_level2_aging")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGING_SUMMARY_CSV = OUT_DIR / "aging_summary.csv"
AGING_CYCLES_DETAIL_CSV = OUT_DIR / "aging_cycles_detail.csv"
AGING_SUMMARY_WITH_FEAS_CSV = OUT_DIR / "aging_summary_with_feasibility.csv"

# DoD binning
DOD_BIN_STEP = 0.05
DOD_BINS = np.arange(0.0, 1.0 + DOD_BIN_STEP, DOD_BIN_STEP)
DOD_BIN_MID = (DOD_BINS[:-1] + DOD_BINS[1:]) / 2

PROFILE_RE = re.compile(
    r"^(SMA|EMA)_(clear|cloudy|medium)_(\d{4}-\d{2}-\d{2})_(\d+min)_(small|large)\.csv$",
    re.IGNORECASE,
)


# =========================
# 1) WANG2011-DERIVED N(DoD)
# =========================
# Reference basis:
# - Wang et al., J. Power Sources 2011
# - graphite–LiFePO4 cells
# - C/2 case
# - T = 25°C
# - EOL defined at 80% remaining capacity (Q_loss = 20%)
#
# Ah-throughput relation in the paper:
#   Ah = cycle_number * DoD * (2 Ah)
# Thus:
#   N(DoD) = Ah_EOL / (2 * DoD)
#
# This script uses the derived N(DoD) curve ONLY as a relative aging proxy basis.

R_GAS = 8.314
T_REF_K = 298.15
QLOSS_EOL = 20.0

A_C2 = 30330.0
EA_C2 = 31500.0
B_C2 = 0.552

Ah_EOL_REF = (QLOSS_EOL / (A_C2 * np.exp(-EA_C2 / (R_GAS * T_REF_K)))) ** (1.0 / B_C2)

DOD_POINTS = np.array([0.10, 0.20, 0.50, 0.80, 0.90], dtype=float)
N_CYCLES_POINTS = Ah_EOL_REF / (2.0 * DOD_POINTS)


def N_of_DoD(dod: np.ndarray) -> np.ndarray:
    """
    Interpolate N(DoD) from Wang2011-derived reference points.
    Values are clipped to the reference DoD range to avoid extrapolation.
    """
    d = np.asarray(dod, dtype=float)
    d = np.clip(d, DOD_POINTS.min(), DOD_POINTS.max())
    return np.interp(d, DOD_POINTS, N_CYCLES_POINTS)


# =========================
# 2) RAINFLOW HELPERS
# =========================
def turning_points(x: np.ndarray) -> np.ndarray:
    """
    Extract turning points (peaks/valleys) from a 1D SoC array.
    Consecutive duplicates are removed first.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 3:
        return x

    keep = [0]
    for i in range(1, len(x)):
        if x[i] != x[i - 1]:
            keep.append(i)
    x = x[keep]

    if x.size < 3:
        return x

    tp = [x[0]]
    for i in range(1, len(x) - 1):
        prev_, cur_, next_ = x[i - 1], x[i], x[i + 1]
        if (cur_ - prev_) == 0:
            continue
        if (cur_ - prev_) * (next_ - cur_) <= 0:
            tp.append(cur_)
    tp.append(x[-1])

    return np.array(tp, dtype=float)


def rainflow_cycles(tp: np.ndarray) -> list[tuple[float, float]]:
    """
    Lightweight rainflow counter.
    Returns list of (range, count), with count = 1.0 for full cycle, 0.5 for half cycle.
    Range is in SoC fraction units.
    """
    stack: list[float] = []
    cycles: list[tuple[float, float]] = []

    for v in tp:
        stack.append(float(v))
        while len(stack) >= 3:
            x0, x1, x2 = stack[-3], stack[-2], stack[-1]
            r1 = abs(x1 - x0)
            r2 = abs(x2 - x1)

            if r2 < r1:
                break

            cycles.append((r1, 1.0))
            stack.pop(-2)

    # remaining half cycles
    for i in range(len(stack) - 1):
        r = abs(stack[i + 1] - stack[i])
        if r > 0:
            cycles.append((r, 0.5))

    return cycles


def bin_cycles_to_DoD(cycles: list[tuple[float, float]]) -> pd.DataFrame:
    """
    Convert rainflow ranges into DoD bins.

    Returns columns:
    - dod_bin_low
    - dod_bin_high
    - dod_mid
    - cycle_count
    """
    if not cycles:
        return pd.DataFrame({
            "dod_bin_low": DOD_BINS[:-1],
            "dod_bin_high": DOD_BINS[1:],
            "dod_mid": DOD_BIN_MID,
            "cycle_count": np.zeros(len(DOD_BIN_MID)),
        })

    ranges = np.array([c[0] for c in cycles], dtype=float)
    counts = np.array([c[1] for c in cycles], dtype=float)

    ranges = np.clip(ranges, 0.0, 1.0)

    bin_idx = np.digitize(ranges, DOD_BINS, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(DOD_BIN_MID) - 1)

    agg = np.zeros(len(DOD_BIN_MID), dtype=float)
    for b, c in zip(bin_idx, counts):
        agg[b] += c

    return pd.DataFrame({
        "dod_bin_low": DOD_BINS[:-1],
        "dod_bin_high": DOD_BINS[1:],
        "dod_mid": DOD_BIN_MID,
        "cycle_count": agg,
    })


def miner_damage_from_binned_cycles(df_bins: pd.DataFrame) -> float:
    """
    Compute relative damage index:
        D_rel = sum_i n_i / N(DoD_i)
    """
    n = df_bins["cycle_count"].values
    dod = df_bins["dod_mid"].values

    valid = n > 0
    if not np.any(valid):
        return 0.0

    N = N_of_DoD(dod[valid])
    return float(np.sum(n[valid] / N))


# =========================
# 3) DISCOVERY + PARSING
# =========================
def discover_feasibility_profiles() -> list[Path]:
    if not FEASIBILITY_PROFILE_DIR.exists():
        raise FileNotFoundError(
            f"Feasibility profile folder not found: {FEASIBILITY_PROFILE_DIR}\n"
            f"Run run_level2_feasibility.py first."
        )
    return sorted(FEASIBILITY_PROFILE_DIR.glob("*.csv"))


def parse_profile_name(path: Path) -> dict:
    m = PROFILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected feasibility profile filename: {path.name}")

    method = m.group(1).upper()
    label = m.group(2).lower()
    date_str = m.group(3)
    window_name = m.group(4)
    scenario = m.group(5).lower()

    return {
        "method": method,
        "label": label,
        "date": date_str,
        "window_name": window_name,
        "scenario": scenario,
    }


def window_to_param(method: str, window_name: str) -> tuple[str, float]:
    wmin = int(window_name.replace("min", ""))
    N = {10: 2, 20: 4, 30: 6, 60: 12}[wmin]

    if method.upper() == "SMA":
        return ("N", float(N))

    alpha = 2.0 / (N + 1.0)
    return ("alpha", float(alpha))


# =========================
# 4) MAIN
# =========================
def main():
    profiles = discover_feasibility_profiles()
    print(f"Found {len(profiles)} feasibility profile files.")
    print(f"[INFO] Wang2011-derived Ah_EOL at 25°C (C/2): {Ah_EOL_REF:.1f} Ah")
    print(f"[INFO] Derived N(DoD) points: {list(zip(DOD_POINTS, np.round(N_CYCLES_POINTS).astype(int)))}")
    print("[INFO] Aging output is intended for relative comparison only.")

    summary_rows: list[dict] = []
    cycles_detail_rows: list[dict] = []

    for prof_path in profiles:
        meta = parse_profile_name(prof_path)
        method = meta["method"]
        label = meta["label"]
        date_str = meta["date"]
        window_name = meta["window_name"]
        scenario = meta["scenario"]

        param_name, param_value = window_to_param(method, window_name)

        df = pd.read_csv(prof_path)
        required_cols = ["timestamp", "SoC_actual", "is_active"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{prof_path.name} missing columns: {missing_cols}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        soc = pd.to_numeric(df["SoC_actual"], errors="coerce")
        mask = pd.to_numeric(df["is_active"], errors="coerce").fillna(0).astype(int) == 1

        soc_active = soc[mask].dropna().values

        if soc_active.size == 0:
            tp = np.array([], dtype=float)
            cycles = []
            df_bins = bin_cycles_to_DoD(cycles)
            damage = 0.0
            soc_min_actual = np.nan
            soc_max_actual = np.nan
            soc_span_actual = np.nan
            n_tp = 0
            n_cycles_total = 0.0
            dominant_dod_mid = np.nan
            dominant_cycle_count = 0.0
        else:
            tp = turning_points(soc_active)
            cycles = rainflow_cycles(tp)
            df_bins = bin_cycles_to_DoD(cycles)
            damage = miner_damage_from_binned_cycles(df_bins)

            soc_min_actual = float(np.min(soc_active))
            soc_max_actual = float(np.max(soc_active))
            soc_span_actual = float(soc_max_actual - soc_min_actual)
            n_tp = int(len(tp))
            n_cycles_total = float(df_bins["cycle_count"].sum())

            if (df_bins["cycle_count"] > 0).any():
                idx_dom = df_bins["cycle_count"].idxmax()
                dominant_dod_mid = float(df_bins.loc[idx_dom, "dod_mid"])
                dominant_cycle_count = float(df_bins.loc[idx_dom, "cycle_count"])
            else:
                dominant_dod_mid = np.nan
                dominant_cycle_count = 0.0

        # summary row
        summary_rows.append({
            "label": label,
            "date": date_str,
            "method": method,
            "window_name": window_name,
            "param_name": param_name,
            "param_value": param_value,
            "scenario": scenario,
            "active_points": int(mask.sum()),
            "soc_min_actual": soc_min_actual,
            "soc_max_actual": soc_max_actual,
            "soc_span_actual": soc_span_actual,
            "n_turning_points": n_tp,
            "n_cycles_total": n_cycles_total,
            "dominant_dod_mid": dominant_dod_mid,
            "dominant_cycle_count": dominant_cycle_count,
            "damage_index": float(damage),
        })

        # bin-level detail rows
        for _, r in df_bins.iterrows():
            if r["cycle_count"] <= 0:
                continue
            cycles_detail_rows.append({
                "label": label,
                "date": date_str,
                "method": method,
                "window_name": window_name,
                "param_name": param_name,
                "param_value": param_value,
                "scenario": scenario,
                "dod_bin_low": float(r["dod_bin_low"]),
                "dod_bin_high": float(r["dod_bin_high"]),
                "dod_mid": float(r["dod_mid"]),
                "cycle_count": float(r["cycle_count"]),
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["scenario", "label", "date", "method", "window_name"]
    ).reset_index(drop=True)
    summary_df.to_csv(AGING_SUMMARY_CSV, index=False)

    # Save cycle details
    cycles_df = pd.DataFrame(cycles_detail_rows)
    if not cycles_df.empty:
        cycles_df = cycles_df.sort_values(
            ["scenario", "label", "date", "method", "window_name", "dod_mid"]
        ).reset_index(drop=True)
    cycles_df.to_csv(AGING_CYCLES_DETAIL_CSV, index=False)

    print(f"Saved aging summary      : {AGING_SUMMARY_CSV}")
    print(f"Saved cycles detail      : {AGING_CYCLES_DETAIL_CSV}")

    # Optional merge with feasibility summary
    if FEASIBILITY_SUMMARY_CSV.exists():
        feas = pd.read_csv(FEASIBILITY_SUMMARY_CSV)

        merge_keys = ["label", "date", "method", "window_name", "scenario"]
        merged = pd.merge(
            summary_df,
            feas,
            on=merge_keys,
            how="left",
            suffixes=("", "_feas"),
        )

        merged.to_csv(AGING_SUMMARY_WITH_FEAS_CSV, index=False)
        print(f"Saved merged summary     : {AGING_SUMMARY_WITH_FEAS_CSV}")

    # Quick sanity checks
    if not summary_df.empty:
        print("\nQuick sanity check:")
        print(f"- damage_index >= 0 : {(summary_df['damage_index'] >= -1e-12).all()}")
        print(f"- active_points > 0 : {(summary_df['active_points'] > 0).all()}")
        print(f"- soc_span_actual >= 0 : {(summary_df['soc_span_actual'].fillna(0) >= -1e-12).all()}")


if __name__ == "__main__":
    main()