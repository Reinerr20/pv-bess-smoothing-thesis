import os
import pandas as pd
import matplotlib.pyplot as plt

SMA_RESULTS_PATH = "outputs_baseline_sma/smoothing_results_sma.csv"
EMA_RESULTS_PATH = "outputs_baseline_ema/smoothing_results_ema.csv"

OUT_DIR = "outputs_baseline_compare"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_ORDER = {"10min": 10, "20min": 20, "30min": 30, "60min": 60}
MARKERS = {"SMA": "o", "EMA": "^"}
COLORS = {
    "clear": "tab:green",
    "medium": "tab:orange",
    "cloudy": "tab:red",
}


def load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def prepare_combined() -> pd.DataFrame:
    sma = load_results(SMA_RESULTS_PATH)
    ema = load_results(EMA_RESULTS_PATH)

    required_cols = [
        "label", "date", "method", "window_name",
        "raw_ramp_p95_kw", "ramp_p95_kw", "throughput_kwh", "ramp_reduction_pct"
    ]
    for c in required_cols:
        if c not in sma.columns or c not in ema.columns:
            raise ValueError(f"Missing column '{c}' in SMA or EMA results")

    combined = pd.concat([sma, ema], ignore_index=True)
    combined["window_order"] = combined["window_name"].map(WINDOW_ORDER)
    DAY_ORDER = {"clear": 1, "medium": 2, "cloudy": 3}
    combined["day_order"] = combined["label"].map(DAY_ORDER)
    combined = combined.sort_values(["day_order", "date", "method", "window_order"]).reset_index(drop=True)
    return combined


def save_summary_table(combined: pd.DataFrame):
    summary_cols = [
        "label", "date", "method", "window_name",
        "raw_ramp_p95_kw", "ramp_p95_kw", "ramp_reduction_pct", "throughput_kwh"
    ]
    summary = combined[summary_cols].copy()
    out_csv = os.path.join(OUT_DIR, "level1_summary_table.csv")
    summary.to_csv(out_csv, index=False)
    return out_csv


def plot_tradeoff(combined: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    for (label, method), g in combined.groupby(["label", "method"]):
        g = g.sort_values("window_order")

        ax.plot(
            g["ramp_p95_kw"],
            g["throughput_kwh"],
            linewidth=1.2,
            alpha=0.8,
            color=COLORS.get(label, None),
        )

        ax.scatter(
            g["ramp_p95_kw"],
            g["throughput_kwh"],
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(label, None),
            edgecolor="black",
            linewidth=0.5,
            s=60,
            label=f"{label} - {method}",
        )

        for _, r in g.iterrows():
            ax.annotate(
                r["window_name"].replace("min", ""),
                (r["ramp_p95_kw"], r["throughput_kwh"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_title("Level-1 Trade-off: RampP95 vs Throughput")
    ax.set_xlabel("RampP95 of $P_{smooth}$ (kW / 5 min)")
    ax.set_ylabel("Required battery throughput (kWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    out_plot = os.path.join(OUT_DIR, "fig_level1_tradeoff.png")
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)
    return out_plot


def main():
    combined = prepare_combined()

    out_csv = os.path.join(OUT_DIR, "smoothing_results_baseline_compare.csv")
    combined.to_csv(out_csv, index=False)

    summary_csv = save_summary_table(combined)
    out_plot = plot_tradeoff(combined)

    print("Comparison outputs saved to:")
    print(f"- {out_csv}")
    print(f"- {summary_csv}")
    print(f"- {out_plot}")


if __name__ == "__main__":
    main()