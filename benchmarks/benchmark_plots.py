import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
RESULTS_CSV = os.path.join(HERE, "results.csv")
OUT_PATH = os.path.join(HERE, "runtime_scaling.png")

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e3e2df"

# Fixed categorical order/colors: identity, not rank, so a legend entry always
# maps to the same implementation regardless of who's fastest at a given n.
SERIES = [
    ("c_time_s", "C (native)", "#2a78d6", "o"),
    ("statsmodels_time_s", "statsmodels", "#eb6834", "s"),
    ("python_time_s", "Manual Python", "#1baf7a", "^"),
]

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.linewidth": 1.2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def plot_runtime_scaling(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for col, label, color, marker in SERIES:
        ax.plot(
            df["n_obs"], df[col],
            label=label, color=color, marker=marker,
            markersize=9, linewidth=3, markeredgewidth=1.5,
            markeredgecolor="white", zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of observations processed")
    ax.set_ylabel("Median time to run (s)")

    fig.text(
        0.06, 0.98, "The hand-written C filter is over 100x faster than Python",
        fontsize=18, fontweight="bold", color=INK, ha="left", va="top",
    )
    fig.text(
        0.06, 0.925,
        "Median runtime of the same local-level Kalman filter, 3 implementations, log-log scale",
        fontsize=11.5, color=MUTED, ha="left", va="top",
    )

    ax.grid(True, which="major", axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)

    handles = [
        Line2D([0], [0], color=color, marker=marker, markersize=8,
               linewidth=3, label=label)
        for _, label, color, marker in SERIES
    ]
    legend = ax.legend(
        handles=handles, loc="upper left", frameon=False, fontsize=12,
        handlelength=1.6, labelspacing=0.6,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    # Direct-labeled speedup callouts at the widest point of separation.
    last = df.iloc[-1]
    ax.annotate(
        f"{last['speedup_vs_python']:.0f}x faster",
        xy=(last["n_obs"], last["python_time_s"]),
        xytext=(-90, -6), textcoords="offset points",
        fontsize=12, fontweight="bold", color="#1baf7a",
    )
    ax.annotate(
        f"{last['speedup_vs_statsmodels']:.0f}x faster",
        xy=(last["n_obs"], last["statsmodels_time_s"]),
        xytext=(-98, -6), textcoords="offset points",
        fontsize=12, fontweight="bold", color="#eb6834",
    )

    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return fig


def main():
    df = pd.read_csv(RESULTS_CSV)

    with plt.rc_context(STYLE):
        fig = plot_runtime_scaling(df)
        fig.savefig(OUT_PATH, dpi=200)
        print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
