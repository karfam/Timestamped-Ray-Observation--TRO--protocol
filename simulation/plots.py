"""Matplotlib plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=300)
    fig.savefig(figure_dir / f"{name}.pdf")
    plt.close(fig)


def line_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    output_dir: Path,
    name: str,
    xlabel: str,
    ylabel: str,
    group: str | None = None,
) -> None:
    """Create a publication-style line plot with optional grouping."""
    if data.empty or x not in data or y not in data:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if group and group in data:
        for label, subset in data.groupby(group):
            subset = subset.sort_values(x)
            ax.plot(subset[x], subset[y], marker="o", linewidth=2.0, label=str(label))
        ax.legend(frameon=False)
    else:
        subset = data.sort_values(x)
        ax.plot(subset[x], subset[y], marker="o", linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, name)


def plot_packet_loss(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "packet_loss", "rmse_m", output_dir, "rmse_vs_packet_loss", "Packet loss probability", "RMSE (m)", "method")
    line_plot(summary, "packet_loss", "estimate_availability_pct", output_dir, "availability_vs_packet_loss", "Packet loss probability", "Availability (%)", "method")


def plot_test1a_packet_loss(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "packet_loss", "rmse_m", output_dir, "test1a_rmse_vs_packet_loss", "Packet loss probability", "RMSE (m)", "method")
    line_plot(summary, "packet_loss", "estimate_availability_pct", output_dir, "test1a_availability_vs_packet_loss", "Packet loss probability", "Availability (%)", "method")


def plot_test1b_heterogeneous_rates(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "packet_loss", "rmse_m", output_dir, "test1b_rmse_vs_packet_loss", "Packet loss probability", "RMSE (m)", "method")
    line_plot(summary, "packet_loss", "estimate_availability_pct", output_dir, "test1b_availability_vs_packet_loss", "Packet loss probability", "Availability (%)", "method")
    line_plot(summary, "packet_loss", "mean_active_rays", output_dir, "test1b_rays_vs_packet_loss", "Packet loss probability", "Mean active rays", "method")


def plot_test2a_delay_sweep(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "delay_s", "rmse_m", output_dir, "test2a_rmse_vs_delay", "Communication delay (s)", "RMSE (m)", "method")
    line_plot(summary, "delay_s", "p95_error_m", output_dir, "test2a_p95_error_vs_delay", "Communication delay (s)", "95th percentile error (m)", "method")
    line_plot(summary, "delay_s", "mean_observation_age_s", output_dir, "test2a_age_vs_delay", "Communication delay (s)", "Mean observation age (s)", "method")
    line_plot(summary, "delay_s", "stale_rejection_rate", output_dir, "test2a_stale_rejection_vs_delay", "Communication delay (s)", "Stale rejection rate", "method")
    line_plot(summary, "delay_s", "estimate_availability_pct", output_dir, "test2a_availability_vs_delay", "Communication delay (s)", "Availability (%)", "method")


def plot_test2b_jitter_sweep(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "jitter_s", "rmse_m", output_dir, "test2b_rmse_vs_jitter", "Delay jitter std. dev. (s)", "RMSE (m)", "method")
    line_plot(summary, "jitter_s", "p95_error_m", output_dir, "test2b_p95_error_vs_jitter", "Delay jitter std. dev. (s)", "95th percentile error (m)", "method")
    line_plot(summary, "jitter_s", "mean_observation_age_s", output_dir, "test2b_age_vs_jitter", "Delay jitter std. dev. (s)", "Mean observation age (s)", "method")
    line_plot(summary, "jitter_s", "estimate_availability_pct", output_dir, "test2b_availability_vs_jitter", "Delay jitter std. dev. (s)", "Availability (%)", "method")


def plot_delay(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "delay_s", "rmse_m", output_dir, "rmse_vs_delay", "Communication delay (s)", "RMSE (m)", "timing_mode")
    line_plot(summary, "delay_s", "mean_observation_age_s", output_dir, "capture_vs_arrival_age", "Communication delay (s)", "Mean observation age (s)", "timing_mode")


def plot_window(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "sliding_window_s", "rmse_m", output_dir, "rmse_vs_window", "Sliding window duration (s)", "RMSE (m)", "target_speed_mps")


def plot_bandwidth(summary: pd.DataFrame, output_dir: Path) -> None:
    line_plot(summary, "num_uavs", "kbit_per_s", output_dir, "bandwidth_vs_uavs", "Number of UAVs", "Payload bandwidth (kbit/s)", "label")


def plot_trajectory(time_series: pd.DataFrame, output_dir: Path, name: str = "trajectory") -> None:
    """Plot true and estimated target trajectories in the horizontal plane."""
    if time_series.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(time_series["truth_x_m"], time_series["truth_y_m"], linewidth=2.0, label="Truth")
    valid = time_series[time_series["valid_estimate"] == True]
    if not valid.empty:
        ax.plot(valid["estimate_x_m"], valid["estimate_y_m"], linewidth=1.5, label="Estimate")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, output_dir, name)


def grouped_bar_plot(
    data: pd.DataFrame,
    category: str,
    value: str,
    group: str,
    output_dir: Path,
    name: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Create a grouped bar chart."""
    if data.empty or category not in data or value not in data or group not in data:
        return
    pivot = data.pivot_table(index=category, columns=group, values=value, aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.autofmt_xdate(rotation=0)
    _save(fig, output_dir, name)


def plot_sync_vs_tro(summary: pd.DataFrame, output_dir: Path) -> None:
    """Generate required synchronized bearing fusion versus TRO comparison plots."""
    if summary.empty:
        return

    ideal = summary[summary["experiment_name"] == "1A_ideal_synchronized"]
    grouped_bar_plot(
        ideal,
        "condition_name",
        "rmse_m",
        "method",
        output_dir,
        "ideal_rmse_comparison",
        "Ideal synchronized condition",
        "RMSE (m)",
    )
    grouped_bar_plot(
        ideal,
        "condition_name",
        "availability_percent",
        "method",
        output_dir,
        "ideal_availability_comparison",
        "Ideal synchronized condition",
        "Availability (%)",
    )

    packet_loss = summary[summary["experiment_name"] == "1D_packet_loss_sweep"]
    line_plot(
        packet_loss,
        "packet_loss",
        "availability_percent",
        output_dir,
        "availability_vs_packet_loss",
        "Packet loss probability",
        "Availability (%)",
        "method",
    )
    line_plot(
        packet_loss,
        "packet_loss",
        "rmse_m",
        output_dir,
        "rmse_vs_packet_loss",
        "Packet loss probability",
        "RMSE (m)",
        "method",
    )

    delay = summary[summary["experiment_name"] == "1C_delay_sweep"]
    line_plot(
        delay,
        "delay",
        "availability_percent",
        output_dir,
        "availability_vs_communication_delay",
        "Communication delay (s)",
        "Availability (%)",
        "method",
    )
    line_plot(
        delay,
        "delay",
        "rmse_m",
        output_dir,
        "rmse_vs_communication_delay",
        "Communication delay (s)",
        "RMSE (m)",
        "method",
    )

    heterogeneous = summary[summary["experiment_name"] == "1B_heterogeneous_update_rates"]
    grouped_bar_plot(
        heterogeneous,
        "condition_name",
        "availability_percent",
        "method",
        output_dir,
        "heterogeneous_availability_comparison",
        "Heterogeneous-rate condition",
        "Availability (%)",
    )
