from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# =========================
# 0) CONFIG
# =========================
DT_MIN = 5
DT_HR = DT_MIN / 60.0

PROFILE_DIRS = [
    Path("outputs_baseline_sma") / "baseline_profiles",
    Path("outputs_baseline_ema") / "baseline_profiles",
]

OUT_DIR = Path("outputs_level2_feasibility")
PROFILES_OUT_DIR = OUT_DIR / "feasibility_profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_OUT_DIR.mkdir(parents=True, exist_ok=True)

LEVEL2_RESULTS_CSV = OUT_DIR / "level2_feasibility_results.csv"

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


# =========================
# 1) BASIC METRICS
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
    e = (psmooth_kw - pout_kw).abs()
    e = e[mask].dropna()
    if e.empty:
        return float("nan")
    return float(np.sum(e.values) * DT_HR)


# =========================
# 2) CONSTRAINED BESS SIMULATION
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

    Returned SoC is post-step SoC_actual, i.e. SoC after applying Pbatt_act at each timestamp.
    """
    pv = pv_kw.copy().astype(float)
    ps = psmooth_kw.reindex(pv.index).astype(float)

    pbatt_req = ps - pv
    pbatt_act = pd.Series(index=pv.index, dtype=float)
    soc = pd.Series(index=pv.index, dtype=float)

    soc_state = float(soc0)

    for t in pv.index:
        preq = pbatt_req.loc[t]

        if not np.isfinite(preq):
            pbatt_act.loc[t] = np.nan
            soc.loc[t] = soc_state
            continue

        # 1) power clipping
        pact = float(np.clip(preq, -pbatt_max_kw, pbatt_max_kw))

        # 2) predict next SoC
        soc_next = soc_state - (pact * DT_HR) / ecap_kwh

        # 3) SoC bound correction
        if soc_next > soc_max:
            pact = -(soc_max - soc_state) * ecap_kwh / DT_HR
            pact = float(np.clip(pact, -pbatt_max_kw, pbatt_max_kw))
            soc_next = soc_state - (pact * DT_HR) / ecap_kwh

        elif soc_next < soc_min:
            pact = (soc_state - soc_min) * ecap_kwh / DT_HR
            pact = float(np.clip(pact, -pbatt_max_kw, pbatt_max_kw))
            soc_next = soc_state - (pact * DT_HR) / ecap_kwh

        pbatt_act.loc[t] = pact
        soc.loc[t] = soc_next     # post-step SoC_actual
        soc_state = soc_next

    pout_act = pv + pbatt_act
    return pbatt_req, pbatt_act, soc, pout_act


# =========================
# 3) PROFILE DISCOVERY + PARSING
# =========================
PROFILE_RE = re.compile(
    r"^(SMA|EMA)_(clear|cloudy|medium)_(\d{4}-\d{2}-\d{2})_(\d+min)\.csv$",
    re.IGNORECASE,
)


def discover_profiles() -> list[Path]:
    files: list[Path] = []
    for d in PROFILE_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.csv")))
    return files


def parse_profile_name(path: Path) -> dict:
    m = PROFILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected profile filename: {path.name}")
    method, label, date_str, window_name = (
        m.group(1).upper(),
        m.group(2).lower(),
        m.group(3),
        m.group(4),
    )
    return {
        "method": method,
        "label": label,
        "date": date_str,
        "window_name": window_name,
    }


def window_to_param(method: str, window_name: str) -> tuple[str, float]:
    wmin = int(window_name.replace("min", ""))
    N = {10: 2, 20: 4, 30: 6, 60: 12}[wmin]
    if method.upper() == "SMA":
        return ("N", float(N))
    alpha = 2.0 / (N + 1)
    return ("alpha", float(alpha))


def save_feasibility_profile(
    *,
    timestamp_index: pd.Index,
    meta: dict,
    scenario: str,
    pv: pd.Series,
    ps: pd.Series,
    pbatt_req: pd.Series,
    pbatt_act: pd.Series,
    soc: pd.Series,
    pout: pd.Series,
    is_active: pd.Series,
):
    out_df = pd.DataFrame({
        "timestamp": timestamp_index,
        "label": meta["label"],
        "date": meta["date"],
        "method": meta["method"],
        "window_name": meta["window_name"],
        "scenario": scenario,
        "P_pv_kw": pv.values,
        "P_smooth_kw": ps.values,
        "P_batt_req_kw": pbatt_req.values,
        "P_batt_act_kw": pbatt_act.values,
        "SoC_actual": soc.values,
        "P_out_act_kw": pout.values,
        "is_active": is_active.astype(int).values,
    })

    fname = f"{meta['method']}_{meta['label']}_{meta['date']}_{meta['window_name']}_{scenario}.csv"
    out_df.to_csv(PROFILES_OUT_DIR / fname, index=False)


# =========================
# 4) MAIN
# =========================
def main():
    rows: list[dict] = []

    profiles = discover_profiles()
    if not profiles:
        raise FileNotFoundError(
            "No baseline profile CSVs found. Make sure baseline SMA/EMA scripts already exported:\n"
            "- outputs_baseline_sma/baseline_profiles\n"
            "- outputs_baseline_ema/baseline_profiles"
        )

    print(f"Found {len(profiles)} baseline profile files.")

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

        # Level-1 required signal
        pbatt_req = ps - pv

        # Level-1 target metrics
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

            # active-window metrics
            active_idx = mask & pbatt_act.notna() & soc.notna() & pout.notna() & ps.notna()

            if active_idx.any():
                power_lim_active = (
                    pbatt_act[active_idx].abs() >= (scen["Pbatt_max_kw"] - 1e-9)
                ).mean()

                soc_lim_active = (
                    (soc[active_idx] <= (scen["soc_min"] + 1e-9)) |
                    (soc[active_idx] >= (scen["soc_max"] - 1e-9))
                ).mean()
            else:
                power_lim_active = float("nan")
                soc_lim_active = float("nan")

            mismatch_kwh = energy_mismatch_kwh(psmooth_kw=ps, pout_kw=pout, mask=mask)
            ramp_actual = ramp_p95_kw(pout, mask=mask)
            thr_act = throughput_kwh(pbatt_act, mask=mask)
            efc = thr_act / (2.0 * scen["Ecap_kwh"]) if np.isfinite(thr_act) else float("nan")

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
            })

            save_feasibility_profile(
                timestamp_index=pv.index,
                meta=meta,
                scenario=scen_name,
                pv=pv,
                ps=ps,
                pbatt_req=pbatt_req2,
                pbatt_act=pbatt_act,
                soc=soc,
                pout=pout,
                is_active=is_active,
            )

    res_df = pd.DataFrame(rows)
    res_df = res_df.sort_values(
        ["scenario", "label", "date", "method", "window_name"]
    ).reset_index(drop=True)
    res_df.to_csv(LEVEL2_RESULTS_CSV, index=False)

    print(f"Saved feasibility summary : {LEVEL2_RESULTS_CSV}")
    print(f"Saved feasibility profiles: {PROFILES_OUT_DIR}")

    # quick sanity checks
    if not res_df.empty:
        print("\nQuick sanity check:")
        print(f"- throughput_act <= throughput_req : {(res_df['throughput_act_kwh'] <= res_df['throughput_req_kwh'] + 1e-9).all()}")
        print(f"- mismatch >= 0                    : {(res_df['energy_mismatch_kwh'] >= -1e-9).all()}")
        print(f"- 0 <= fp <= 1                    : {((res_df['power_limit_frac'] >= -1e-9) & (res_df['power_limit_frac'] <= 1 + 1e-9)).all()}")
        print(f"- 0 <= fsoc <= 1                  : {((res_df['soc_limit_frac'] >= -1e-9) & (res_df['soc_limit_frac'] <= 1 + 1e-9)).all()}")


if __name__ == "__main__":
    main()