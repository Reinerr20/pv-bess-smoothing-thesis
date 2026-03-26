# Compare Baseline SMA vs EMA
# --------------------------
# This script loads SMA & EMA baseline results and produces:
# 1) Combined CSV (SMA + EMA)
# 2) Trade-off plot: RampP95 vs Throughput (SMA vs EMA)

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SMA_RESULTS_PATH = "outputs_baseline_sma/smoothing_results_sma.csv"
EMA_RESULTS_PATH = "outputs_baseline_ema/smoothing_results_ema.csv"

OUT_DIR = "outputs_baseline_compare"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================

def load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# =========================
# MAIN
# =========================

def main():
    sma = load_results(SMA_RESULTS_PATH)
    ema = load_results(EMA_RESULTS_PATH)

    # consistency check
    required_cols = [
        "label", "date", "method", "window_name",
        "ramp_p95_kw", "throughput_kwh"
    ]
    for c in required_cols:
        if c not in sma.columns or c not in ema.columns:
            raise ValueError(f"Missing column '{c}' in SMA or EMA results")

    # combine
    combined = pd.concat([sma, ema], ignore_index=True)

    # save combined CSV
    out_csv = os.path.join(OUT_DIR, "smoothing_results_baseline_compare.csv")
    combined.to_csv(out_csv, index=False)

    # =========================
    # TRADE-OFF PLOT (SMA vs EMA)
    # =========================
    fig, ax = plt.subplots()

    markers = {"SMA": "o", "EMA": "^"}
    colors = {
        "clear": "tab:blue",
        "medium": "tab:orange",
        "cloudy": "tab:green",
    }

    for (method, label), g in combined.groupby(["method", "label"]):
        ax.scatter(
            g["ramp_p95_kw"],
            g["throughput_kwh"],
            marker=markers.get(method, "o"),
            color=colors.get(label, None),
            label=f"{method} – {label}",
        )

        # annotate window name
        for _, r in g.iterrows():
            ax.annotate(
                r["window_name"],
                (r["ramp_p95_kw"], r["throughput_kwh"]),
                fontsize=8,
            )

    ax.set_title("Baseline Trade-off: SMA vs EMA")
    ax.set_xlabel("RampP95 of Psmooth (kW / 5 min)")
    ax.set_ylabel("Battery Throughput (kWh)")
    ax.legend(ncol=2)

    fig.tight_layout()
    out_plot = os.path.join(OUT_DIR, "tradeoff_SMA_vs_EMA.png")
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)

    print("Comparison results saved to:")
    print(f"- {out_csv}")
    print(f"- {out_plot}")


if __name__ == "__main__":
    main()
