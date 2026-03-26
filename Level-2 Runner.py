from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) CONFIG
# =========================
DT_MIN = 5
DT_HR = DT_MIN / 60.0

# Baseline profiles folders (from your patched baseline scripts)
PROFILE_DIRS = [
    Path("outputs_baseline_sma") / "baseline_profiles",
    Path("outputs_baseline_ema") / "baseline_profiles",
]

OUT_DIR = Path("outputs_level2")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LEVEL2_RESULTS_CSV = OUT_DIR / "level2_results.csv"
CYCLES_DETAIL_CSV = OUT_DIR / "rainflow_cycles_detail.csv"

# BESS scenarios (Proposal 1 locked)
SCENARIOS = {
    "small": {
        "Ecap_kwh": 2000.0,   # 2 MWh
        "Pbatt_max_kw": 500.0,
        "soc_min": 0.20,
        "soc_max": 0.80,
        "soc0": 0.50,
    },
    "large": {
        "Ecap_kwh": 6000.0,   # 6 MWh
        "Pbatt_max_kw": 1500.0,
        "soc_min": 0.20,
        "soc_max": 0.80,
        "soc0": 0.50,
    },
}

# DoD binning (5%)
DOD_BIN_STEP = 0.05
DOD_BINS = np.arange(0.0, 1.0 + DOD_BIN_STEP, DOD_BIN_STEP)
DOD_BIN_MID = (DOD_BINS[:-1] + DOD_BINS[1:]) / 2

# =========================
# 1) N(DoD) TABLE (DERIVED FROM WANG et al., J Power Sources 2011)
# =========================
# Reference:
#   J. Wang et al., "Cycle-life model for graphite–LiFePO4 cells", J. Power Sources, 2011.
# Model used: C/2 case from Table 1, EOL = 80% remaining capacity (Q_loss = 20%).
# We set a nominal temperature T = 25°C (298.15 K) to obtain a reference Ah_EoL.
# The paper defines Ah-throughput as:
#   Ah = cycle_number * DoD * (2 Ah)
# Hence cycles-to-EOL at a given DoD is:
#   N(DoD) = Ah_EoL / (2 * DoD)
# This yields a reference N(DoD) curve for comparative (relative) aging tendency analysis.

R_GAS = 8.314  # J/mol/K
T_REF_K = 298.15  # 25°C nominal
QLOSS_EOL = 20.0  # % capacity loss => 80% remaining capacity

# C/2 model coefficients from Table 1:
# Q_loss = 30330 * exp(-31500/(R*T)) * (Ah)^0.552
A_C2 = 30330.0
EA_C2 = 31500.0
B_C2 = 0.552

# Solve for Ah_EoL at reference temperature
Ah_EOL_REF = (QLOSS_EOL / (A_C2 * np.exp(-EA_C2 / (R_GAS * T_REF_K)))) ** (1.0 / B_C2)

# Use DoD points commonly referenced in the paper experiments (and convenient for interpolation)
DOD_POINTS = np.array([0.10, 0.20, 0.50, 0.80, 0.90], dtype=float)

# Convert Ah_EoL to cycles-to-EOL (Table values for N(DoD))
# NOTE: 2 Ah is the nominal capacity used in Wang et al. for Ah-throughput definition.
N_CYCLES_POINTS = Ah_EOL_REF / (2.0 * DOD_POINTS)

# If you want to hardcode rounded integers instead (same numbers), uncomment below:
# N_CYCLES_POINTS = np.array([86005, 43003, 17201, 10751, 9556], dtype=float)

print(f"[INFO] Wang2011-derived Ah_EOL at 25°C (C/2): {Ah_EOL_REF:.1f} Ah")
print(f"[INFO] Derived N(DoD) points: {list(zip(DOD_POINTS, np.round(N_CYCLES_POINTS).astype(int)))}")


def N_of_DoD(dod: np.ndarray) -> np.ndarray:
    """Interpolate N(DoD) from reference points (linear interpolation in DoD domain).

    If dod is outside [min,max], it is clipped to avoid extrapolation.
    """
    d = np.asarray(dod, dtype=float)
    d = np.clip(d, DOD_POINTS.min(), DOD_POINTS.max())
    return np.interp(d, DOD_POINTS, N_CYCLES_POINTS)


# =========================
# 2) BASIC METRICS
# =========================
def ramp_p95_kw(series_kw: pd.Series, mask: pd.Series | None = None) -> float:
    s = series_kw.copy()
    if mask is not None:
        s = s[mask]
    s = s.dropna()
    if len(s) < 2:
        return float("nan")
    d = s.diff().abs().dropna()
    if d.empty:
        return float("nan")
    return float(np.percentile(d.values, 95))


def throughput_kwh(pbatt_kw: pd.Series, mask: pd.Series | None = None) -> float:
    p = pbatt_kw.copy()
    if mask is not None:
        p = p[mask]
    p = p.dropna()
    if p.empty:
        return float("nan")
    return float(np.sum(np.abs(p.values)) * DT_HR)


def energy_mismatch_kwh(psmooth_kw: pd.Series, pout_kw: pd.Series, mask: pd.Series) -> float:
    # mismatch is computed on active window
    e = (psmooth_kw - pout_kw).abs()
    e = e[mask].dropna()
    if e.empty:
        return float("nan")
    return float(np.sum(e.values) * DT_HR)


# =========================
# 3) CONSTRAINED BESS SIMULATION
# =========================

def simulate_bess_constraints(
    *,
    pv_kw: pd.Series,
    psmooth_kw: pd.Series,
    soc0: float,
    soc_min: float,
    soc_max: float,
    ecap_kwh: float,
    pbatt_max_kw: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Pbatt > 0 : discharge  -> SoC decreases
    Pbatt < 0 : charge     -> SoC increases
    """
    pv = pv_kw.copy().astype(float)
    ps = psmooth_kw.reindex(pv.index).astype(float)

    pbatt_req = ps - pv
    pbatt_act = pd.Series(index=pv.index, dtype=float)
    soc = pd.Series(index=pv.index, dtype=float)

    soc_state = float(soc0)

    for i, t in enumerate(pv.index):
        preq = pbatt_req.loc[t]
        soc.loc[t] = soc_state

        if not np.isfinite(preq):
            pbatt_act.loc[t] = np.nan
            continue

        # 1) power clip
        pact = float(np.clip(preq, -pbatt_max_kw, pbatt_max_kw))

        # 2) predict next SoC
        soc_next = soc_state - (pact * DT_HR) / ecap_kwh

        # 3) SoC bound correction (hit bounds)
        if soc_next > soc_max:
            pact = -(soc_max - soc_state) * ecap_kwh / DT_HR
            pact = float(np.clip(pact, -pbatt_max_kw, pbatt_max_kw))
            soc_next = soc_state - (pact * DT_HR) / ecap_kwh

        elif soc_next < soc_min:
            pact = (soc_state - soc_min) * ecap_kwh / DT_HR
            pact = float(np.clip(pact, -pbatt_max_kw, pbatt_max_kw))
            soc_next = soc_state - (pact * DT_HR) / ecap_kwh

        pbatt_act.loc[t] = pact
        soc_state = soc_next  # update state for next step

    pout_act = pv + pbatt_act
    return pbatt_req, pbatt_act, soc, pout_act



# =========================
# 4) RAINFLOW (simple implementation) + BINNING
# =========================

def turning_points(x: np.ndarray) -> np.ndarray:
    """Extract turning points (peaks/valleys) for rainflow.

    Uses a simple method: remove consecutive duplicates, then keep points where slope changes sign.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return x

    # remove consecutive duplicates
    keep = [0]
    for i in range(1, len(x)):
        if x[i] != x[i - 1]:
            keep.append(i)
    x = x[keep]
    if x.size < 3:
        return x

    tp = [x[0]]
    for i in range(1, len(x) - 1):
        prev, cur, nxt = x[i - 1], x[i], x[i + 1]
        if (cur - prev) == 0:
            continue
        if (cur - prev) * (nxt - cur) <= 0:
            tp.append(cur)
    tp.append(x[-1])
    return np.array(tp, dtype=float)


def rainflow_cycles(tp: np.ndarray) -> list[tuple[float, float]]:
    """Very small rainflow counter.

    Returns list of (range, count) where range is peak-to-valley amplitude.
    count is 1.0 for full cycle, 0.5 for half cycle.

    This is a lightweight implementation suitable for relative comparison.
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
            # count a full cycle of range r1
            cycles.append((r1, 1.0))
            # remove the middle point
            stack.pop(-2)

    # remaining are half cycles
    for i in range(len(stack) - 1):
        r = abs(stack[i + 1] - stack[i])
        if r > 0:
            cycles.append((r, 0.5))

    return cycles


def bin_cycles_to_DoD(cycles: list[tuple[float, float]]) -> pd.DataFrame:
    """Convert rainflow ranges (SoC amplitude) to DoD bins.

    cycles: list of (range, count). range is in SoC units (0..1)

    Returns DataFrame with columns:
    - dod_bin_low, dod_bin_high, dod_mid, count
    """
    if not cycles:
        return pd.DataFrame({
            "dod_bin_low": DOD_BINS[:-1],
            "dod_bin_high": DOD_BINS[1:],
            "dod_mid": DOD_BIN_MID,
            "count": np.zeros(len(DOD_BIN_MID)),
        })

    ranges = np.array([c[0] for c in cycles], dtype=float)
    counts = np.array([c[1] for c in cycles], dtype=float)

    # Clip DoD to [0,1]
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
        "count": agg,
    })


def miner_damage_from_binned_cycles(df_bins: pd.DataFrame) -> float:
    """Compute Miner damage index from binned DoD cycles."""
    n = df_bins["count"].values
    dod = df_bins["dod_mid"].values
    # ignore zero cycles
    m = n > 0
    if not np.any(m):
        return 0.0
    N = N_of_DoD(dod[m])
    return float(np.sum(n[m] / N))


# =========================
# 5) PROFILE DISCOVERY + PARSING
# =========================

def discover_profiles() -> list[Path]:
    files: list[Path] = []
    for d in PROFILE_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.csv")))
    return files


PROFILE_RE = re.compile(r"^(SMA|EMA)_(clear|cloudy|medium)_(\d{4}-\d{2}-\d{2})_(\d+min)\.csv$", re.IGNORECASE)


def parse_profile_name(path: Path) -> dict:
    m = PROFILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected profile filename: {path.name}")
    method, label, date_str, window_name = m.group(1).upper(), m.group(2).lower(), m.group(3), m.group(4)
    return {"method": method, "label": label, "date": date_str, "window_name": window_name}


def window_to_param(method: str, window_name: str) -> tuple[str, float]:
    wmin = int(window_name.replace("min", ""))
    N = {10: 2, 20: 4, 30: 6, 60: 12}[wmin]
    if method.upper() == "SMA":
        return ("N", float(N))
    alpha = 2.0 / (N + 1)
    return ("alpha", float(alpha))


# =========================
# 6) MAIN
# =========================

rows: list[dict] = []
cycles_detail_rows: list[dict] = []

profiles = discover_profiles()
if not profiles:
    raise FileNotFoundError(
        "No baseline profile CSVs found. Make sure you ran patched baseline SMA/EMA scripts and they exported to:\n"
        "- outputs_baseline_sma/baseline_profiles\n"
        "- outputs_baseline_ema/baseline_profiles"
    )

print(f"Found {len(profiles)} profile files.")

# For plots: pick a single representative case
PLOT_ONE_EXAMPLE = True
example_plotted = False

for prof_path in profiles:
    meta = parse_profile_name(prof_path)
    method = meta["method"]
    label = meta["label"]
    date_str = meta["date"]
    window_name = meta["window_name"]

    df = pd.read_csv(prof_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    pv = df["P_pv_kw"].astype(float)
    ps = df["P_smooth_kw"].astype(float)
    is_active = df["is_active"].astype(int)
    mask = is_active == 1

    param_name, param_value = window_to_param(method, window_name)

    # baseline-required signal
    pbatt_req = ps - pv

    # baseline target metrics
    ramp_target = ramp_p95_kw(ps, mask=mask)
    thr_req = throughput_kwh(pbatt_req, mask=mask)

    for scen_name, scen in SCENARIOS.items():
        pbatt_req2, pbatt_act, soc, pout = simulate_bess_constraints(
            pv_kw=pv,
            psmooth_kw=ps,
            soc0=scen["soc0"],
            soc_min=scen["soc_min"],
            soc_max=scen["soc_max"],
            ecap_kwh=scen["Ecap_kwh"],
            pbatt_max_kw=scen["Pbatt_max_kw"],
        )

        # feasibility metrics (active window)
        active_idx = mask & pbatt_act.notna() & soc.notna() & pout.notna() & ps.notna()
        if active_idx.any():
            power_lim_active = (pbatt_act[active_idx].abs() >= (scen["Pbatt_max_kw"] - 1e-9)).mean()
            soc_lim_active = ((soc[active_idx] <= (scen["soc_min"] + 1e-9)) | (soc[active_idx] >= (scen["soc_max"] - 1e-9))).mean()
        else:
            power_lim_active = float("nan")
            soc_lim_active = float("nan")

        mismatch_kwh = energy_mismatch_kwh(psmooth_kw=ps, pout_kw=pout, mask=mask)

        # performance actual
        ramp_actual = ramp_p95_kw(pout, mask=mask)
        thr_act = throughput_kwh(pbatt_act, mask=mask)

        # EFC
        efc = thr_act / (2.0 * scen["Ecap_kwh"]) if np.isfinite(thr_act) else float("nan")

        # rainflow on SoC (active window only)
        soc_active = soc[mask].dropna().values
        tp = turning_points(soc_active)
        cycles = rainflow_cycles(tp)
        df_bins = bin_cycles_to_DoD(cycles)
        damage = miner_damage_from_binned_cycles(df_bins)

        # save cycles detail (per run + scenario)
        for _, r in df_bins.iterrows():
            if r["count"] <= 0:
                continue
            cycles_detail_rows.append({
                "method": method,
                "label": label,
                "date": date_str,
                "window_name": window_name,
                "param_name": param_name,
                "param_value": param_value,
                "scenario": scen_name,
                "dod_bin_low": float(r["dod_bin_low"]),
                "dod_bin_high": float(r["dod_bin_high"]),
                "dod_mid": float(r["dod_mid"]),
                "cycle_count": float(r["count"]),
            })

        rows.append({
            "label": label,
            "date": date_str,
            "method": method,
            "window_name": window_name,
            "param_name": param_name,
            "param_value": param_value,
            "scenario": scen_name,
            "Ecap_kwh": scen["Ecap_kwh"],
            "Pbatt_max_kw": scen["Pbatt_max_kw"],
            "soc_min": scen["soc_min"],
            "soc_max": scen["soc_max"],
            "soc0": scen["soc0"],
            "ramp_p95_kw_target": ramp_target,
            "ramp_p95_kw_actual": ramp_actual,
            "throughput_req_kwh": thr_req,
            "throughput_act_kwh": thr_act,
            "energy_mismatch_kwh": mismatch_kwh,
            "power_limit_frac": float(power_lim_active),
            "soc_limit_frac": float(soc_lim_active),
            "efc_act": float(efc),
            "damage_index": float(damage),
        })

        # minimal example plots
        if PLOT_ONE_EXAMPLE and (not example_plotted) and (label == "cloudy") and (window_name == "60min") and (scen_name == "small"):
            example_plotted = True
            # SoC plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(soc.index, soc.values, label="SoC")
            ax.axhline(scen["soc_min"], linestyle="--", linewidth=1, label="SoC min")
            ax.axhline(scen["soc_max"], linestyle="--", linewidth=1, label="SoC max")
            ax.set_title(f"SoC(t) — {method} {label} {date_str} {window_name} — {scen_name}")
            ax.set_xlabel("Time")
            ax.set_ylabel("SoC")
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "example_soc.png", dpi=200)
            plt.close(fig)

            # DoD histogram plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(df_bins["dod_mid"], df_bins["count"], width=DOD_BIN_STEP * 0.9)
            ax.set_title(f"Rainflow DoD Histogram — {method} {label} {date_str} {window_name} — {scen_name}")
            ax.set_xlabel("DoD (fraction)")
            ax.set_ylabel("Cycle count")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "example_dod_hist.png", dpi=200)
            plt.close(fig)

# Save outputs
res_df = pd.DataFrame(rows)
res_df = res_df.sort_values(["scenario", "label", "date", "method", "window_name"]).reset_index(drop=True)
res_df.to_csv(LEVEL2_RESULTS_CSV, index=False)

cycles_df = pd.DataFrame(cycles_detail_rows)
if not cycles_df.empty:
    cycles_df = cycles_df.sort_values(["scenario", "label", "date", "method", "window_name", "dod_mid"]).reset_index(drop=True)
cycles_df.to_csv(CYCLES_DETAIL_CSV, index=False)

print(f"Saved Level-2 results : {LEVEL2_RESULTS_CSV}")
print(f"Saved cycles detail   : {CYCLES_DETAIL_CSV}")
print(f"Saved example plots   : {PLOTS_DIR}")

# Quick sanity print
print("\nNOTE: N(DoD) table is currently PLACEHOLDER. Replace DOD_POINTS/N_CYCLES_POINTS with values from your reference.")
