"""The three README charts.  python -m pace.charts  (run pace.backtest first)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pace.backtest import REPORTS, summarise  # noqa: E402
from pace.data import load_hotels  # noqa: E402
from pace.models import reference_days  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COLORS = {"pace": "#2a78d6", "gradient_boosting": "#eb6834", "weekday_average": "#1baf7a"}
LABELS = {"pace": "Booking pace (rule)", "gradient_boosting": "Gradient boosting (ML)",
          "weekday_average": "Weekday average"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def error_by_horizon(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (hotel, g) in zip(axes, summary.groupby("hotel")):
        for m, color in COLORS.items():
            ax.plot(g.horizon, g[m], color=color, lw=2, marker="o", ms=6, label=LABELS[m])
            ax.annotate(f"{g[m].iloc[-1]:.0f}", (g.horizon.iloc[-1], g[m].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)
        ax.set_title(f"{hotel}  ·  avg {g.average_guests.iloc[0]:.0f} guests/day", loc="left",
                     color=INK, fontsize=12)
        ax.set_xticks(g.horizon)
        ax.set_xlabel("Days ahead the forecast was made")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Average error (guests per day)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.91), ncol=3, fontsize=10)
    fig.suptitle("Reading the reservation book cuts forecast error by 53-68% a week ahead",
                 x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def forecast_vs_actual(test: pd.DataFrame, out: Path, hotel="City Hotel", horizon=7,
                       start="2017-05-01", end="2017-06-30") -> None:
    g = test[(test.hotel == hotel) & (test.horizon == horizon)
             & test.day.between(pd.Timestamp(start), pd.Timestamp(end))]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(g.day, g.actual, color=INK, lw=2, label="What actually happened")
    ax.plot(g.day, g.pace, color=COLORS["pace"], lw=2, label=LABELS["pace"])
    ax.plot(g.day, g.weekday_average, color=COLORS["weekday_average"], lw=2, ls=(0, (4, 2)),
            label=LABELS["weekday_average"])
    ax.set_ylabel("Guests arriving")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.suptitle(f"{hotel}, forecast made {horizon} days ahead ({pd.Timestamp(start):%b}-"
                 f"{pd.Timestamp(end):%b %Y})", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def booking_curve(test: pd.DataFrame, hotels, out: Path, hotel="City Hotel", horizon=7) -> None:
    """The day where the weekday average was furthest off, and how the book showed it early."""
    g = test[(test.hotel == hotel) & (test.horizon == horizon)].copy()
    g["miss"] = (g.weekday_average - g.actual).abs() - (g.pace - g.actual).abs()
    day = g.sort_values("miss").day.iloc[-1]
    row = g[g.day == day].iloc[0]
    h = hotels[hotel]
    ks = np.arange(60, -1, -1)
    refs = reference_days(h, day, horizon)
    this = [h.booked(day, k) for k in ks]
    normal = [np.mean([h.booked(r, k) for r in refs]) for k in ks]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(-ks, normal, color=MUTED, lw=2, label=f"A normal {day:%A} (last {len(refs)} weeks)")
    ax.plot(-ks, this, color=COLORS["pace"], lw=2, label=f"{day:%A %d %B %Y}")
    ax.axvline(-(horizon - 1), color=INK2, lw=1, ls=(0, (2, 2)))
    ax.text(-(horizon - 1) - 0.8, 12, f"forecast made here\n({horizon} days ahead)",
            ha="right", va="bottom", color=INK2, fontsize=10)
    ax.set_xlabel("Days before arrival")
    ax.set_ylabel("Guests in the reservation book")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(f"The book was running {row.booked_now - row.booked_normally:+.0f} guests ahead of normal. "
                 f"Weekday average said {row.weekday_average:.0f}, pace said {row.pace:.0f}, "
                 f"{row.actual} came.", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    test = pd.read_csv(REPORTS / "forecasts.csv.gz", parse_dates=["day"])
    summary = summarise(test)
    error_by_horizon(summary, REPORTS / "error_by_horizon.png")
    forecast_vs_actual(test, REPORTS / "forecast_vs_actual.png")
    booking_curve(test, load_hotels(), REPORTS / "booking_curve.png")
    print("charts written to", REPORTS)


if __name__ == "__main__":
    main()
