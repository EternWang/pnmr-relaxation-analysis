from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .models import t1_magnitude_model, t2_envelope_model

BLUE = "#2F6B9A"
ORANGE = "#D97935"
GREEN = "#5B8C5A"
GRAY = "#4A5568"
LIGHT = "#EEF2F6"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
        }
    )


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="white",
        edgecolor="#CBD5E1",
        linewidth=1.0,
    )
    ax.add_patch(box)
    title_y = y + h - 0.075 if body else y + h / 2
    title_va = "top" if body else "center"
    ax.text(x + 0.035, title_y, title, ha="left", va=title_va, fontsize=10.2, weight="bold", color=color)
    if body:
        ax.text(x + 0.035, y + h - 0.215, body, ha="left", va="top", fontsize=8.45, color="#172033", linespacing=1.18)


def plot_t1_fit(
    delays_ms: np.ndarray,
    amplitudes_mV: np.ndarray,
    sigma_mV: np.ndarray,
    fit_params: tuple[float, float, float],
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    grid = np.linspace(np.min(delays_ms), np.max(delays_ms), 500)
    fit = t1_magnitude_model(grid, *fit_params)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(delays_ms, amplitudes_mV, yerr=sigma_mV, fmt="o", color=BLUE, ecolor="#7FA7C7", capsize=3, label="Measured amplitudes")
    ax.plot(grid, fit, color=ORANGE, lw=2.4, label="Weighted nonlinear fit")
    ax.set_xlabel("Delay $\\tau$ (ms)")
    ax.set_ylabel("Baseline-subtracted amplitude (mV)")
    ax.set_title("Inversion-recovery fit for T1")
    ax.text(
        0.03,
        0.95,
        f"T1 = {fit_params[2]:.2f} ms",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )
    ax.legend()
    save_figure(fig, save_path)


def plot_t2_fit(
    time_ms: np.ndarray,
    amplitude_mV: np.ndarray,
    sigma_mV: np.ndarray,
    fit_params: tuple[float, float],
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    grid = np.linspace(np.min(time_ms), np.max(time_ms), 600)
    fit = t2_envelope_model(grid, *fit_params)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(time_ms, amplitude_mV, yerr=sigma_mV, fmt="o", ms=4, color=BLUE, ecolor="#7FA7C7", capsize=2, label="Detected echo peaks")
    ax.plot(grid, fit, color=ORANGE, lw=2.4, label="Exponential fit")
    ax.set_xlabel("Echo time (ms)")
    ax.set_ylabel("Amplitude above baseline (mV)")
    ax.set_title("Meiboom-Gill echo-envelope fit for T2")
    ax.text(
        0.03,
        0.95,
        f"T2 = {fit_params[1]:.2f} ms",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )
    ax.legend()
    save_figure(fig, save_path)


def plot_waveform_with_peaks(
    time_ms: np.ndarray,
    amplitude_mV: np.ndarray,
    peak_times_ms: np.ndarray,
    peak_amplitudes_mV: np.ndarray,
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(time_ms, amplitude_mV, color=GRAY, lw=0.8, alpha=0.9, label="Baseline-subtracted waveform")
    ax.scatter(peak_times_ms, peak_amplitudes_mV, color=ORANGE, s=22, zorder=3, label=f"Detected peaks (n={len(peak_times_ms)})")
    ax.set_xlim(-2, 75)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude above baseline (mV)")
    ax.set_title("Echo detection on raw scope waveform")
    ax.legend(loc="upper right")
    save_figure(fig, save_path)


def plot_results_dashboard(
    delays_ms: np.ndarray,
    t1_amplitudes_mV: np.ndarray,
    t1_sigma_mV: np.ndarray,
    t1_params: tuple[float, float, float],
    t2_time_ms: np.ndarray,
    t2_amplitudes_mV: np.ndarray,
    t2_sigma_mV: np.ndarray,
    t2_params: tuple[float, float],
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    grid_t1 = np.linspace(np.min(delays_ms), np.max(delays_ms), 500)
    axes[0].errorbar(delays_ms, t1_amplitudes_mV, yerr=t1_sigma_mV, fmt="o", color=BLUE, ecolor="#7FA7C7", capsize=2)
    axes[0].plot(grid_t1, t1_magnitude_model(grid_t1, *t1_params), color=ORANGE, lw=2.4)
    axes[0].set_xlabel("Delay $\\tau$ (ms)")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("T1 inversion recovery")
    axes[0].text(
        0.04,
        0.94,
        f"T1 = {t1_params[2]:.2f} ms",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )

    grid_t2 = np.linspace(np.min(t2_time_ms), np.max(t2_time_ms), 600)
    axes[1].errorbar(t2_time_ms, t2_amplitudes_mV, yerr=t2_sigma_mV, fmt="o", ms=3.4, color=BLUE, ecolor="#7FA7C7", capsize=2)
    axes[1].plot(grid_t2, t2_envelope_model(grid_t2, *t2_params), color=ORANGE, lw=2.4)
    axes[1].set_xlabel("Echo time (ms)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_title("T2 echo envelope")
    axes[1].text(
        0.04,
        0.94,
        f"T2 = {t2_params[1]:.2f} ms",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )

    save_figure(fig, save_path)


def plot_research_snapshot(summary: dict[str, float | int], save_path: str | Path) -> None:
    set_style()
    save_path = Path(save_path)
    fig = plt.figure(figsize=(11.2, 5.4), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.08], width_ratios=[1.0, 1.0, 1.1])

    ax_cards = fig.add_subplot(grid[0, :])
    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    fig.suptitle("Pulsed NMR relaxation analysis", x=0.04, y=0.985, ha="left", fontsize=17, weight="bold", color="#172033")
    fig.text(
        0.04,
        0.925,
        "Raw oscilloscope traces and timing records reduced to reproducible T1/T2 estimates with uncertainty checks.",
        ha="left",
        fontsize=10.5,
        color=GRAY,
    )

    _card(
        ax_cards,
        0.02,
        0.08,
        0.28,
        0.68,
        "Signal extraction",
        f"{int(summary['n_echo_peaks'])} detected echo peaks\n"
        f"baseline RMS {summary['baseline_rms_mV']:.3f} mV\n"
        "pre-trigger noise",
        BLUE,
    )
    _card(
        ax_cards,
        0.36,
        0.08,
        0.28,
        0.68,
        "Relaxation estimates",
        f"T1 fit {summary['t1_fit_ms']:.2f} +/- {summary['t1_fit_sigma_ms']:.2f} ms\n"
        f"T1 zero {summary['t1_zero_crossing_ms']:.2f} +/- {summary['t1_zero_crossing_sigma_ms']:.2f} ms\n"
        f"T2 fit {summary['t2_fit_ms']:.2f} +/- {summary['t2_fit_sigma_ms']:.2f} ms",
        ORANGE,
    )
    _card(
        ax_cards,
        0.70,
        0.08,
        0.28,
        0.68,
        "Reproducible outputs",
        "echo table\nJSON summary\nfigures + tests",
        GREEN,
    )

    ax_bar = fig.add_subplot(grid[1, :2])
    labels = ["T1 zero crossing", "T1 nonlinear fit", "T2 envelope fit"]
    values = np.array(
        [
            summary["t1_zero_crossing_ms"],
            summary["t1_fit_ms"],
            summary["t2_fit_ms"],
        ],
        dtype=float,
    )
    errors = np.array(
        [
            summary["t1_zero_crossing_sigma_ms"],
            summary["t1_fit_sigma_ms"],
            summary["t2_fit_sigma_ms"],
        ],
        dtype=float,
    )
    ypos = np.arange(len(labels))
    ax_bar.barh(ypos, values, xerr=errors, color=[BLUE, ORANGE, GREEN], ecolor="#7FA7C7", capsize=4)
    ax_bar.set_yticks(ypos, labels)
    ax_bar.set_xlabel("Relaxation time (ms)")
    ax_bar.set_title("Independent relaxation-time summaries", weight="bold", loc="left")
    ax_bar.set_xlim(0, max(values + errors) + 8)
    for y, value, error in zip(ypos, values, errors):
        ax_bar.text(value + error + 1.0, y, f"{value:.2f} +/- {error:.2f}", va="center", color=GRAY, fontsize=9.5)

    ax_flow = fig.add_subplot(grid[1, 2])
    ax_flow.axis("off")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.set_title("Analysis chain", weight="bold", loc="left", pad=8)
    steps = ["raw waveform", "baseline + peaks", "weighted fits", "tables + figures"]
    y_positions = [0.82, 0.60, 0.38, 0.16]
    for idx, (step, y) in enumerate(zip(steps, y_positions)):
        _card(ax_flow, 0.08, y, 0.78, 0.13, f"{idx + 1}. {step}", "", [BLUE, ORANGE, GREEN, GRAY][idx])
        if idx < len(steps) - 1:
            ax_flow.annotate("", xy=(0.47, y - 0.015), xytext=(0.47, y - 0.07), arrowprops={"arrowstyle": "->", "color": "#94A3B8"})

    fig.tight_layout(rect=[0.03, 0.02, 0.99, 0.9])
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
