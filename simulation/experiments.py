"""Experiment runners for the TRO simulator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import ExperimentConfig, FusionConfig, NetworkConfig, SimulationConfig
from .fusion_node import FusionNode
from .metrics import MetricsRecorder, aggregate_runs, summarize_time_series
from .network_model import NetworkModel
from .plots import plot_bandwidth, plot_delay, plot_packet_loss, plot_trajectory, plot_window
from .scenario import Scenario
from .tro_message import payload_bandwidth
from .utils import ensure_output_dirs, make_rng


def run_single_simulation(
    sim_config: SimulationConfig,
    network_config: NetworkConfig,
    fusion_config: FusionConfig,
    seed: int,
    payload_bytes: int = 64,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run one scenario realization and return time series plus summary."""
    sim_config = replace(sim_config, seed=seed)
    sim_config.validate()
    network_config.validate()
    fusion_config.validate()

    rng = make_rng(seed)
    scenario = Scenario(sim_config, rng)
    network = NetworkModel(network_config, rng)
    fusion = FusionNode(fusion_config)
    recorder = MetricsRecorder()

    next_observation_times = np.zeros(sim_config.num_uavs, dtype=float)
    observation_periods = np.array([1.0 / rate for rate in sim_config.observation_rates_hz], dtype=float)
    sequences = np.zeros(sim_config.num_uavs, dtype=int)
    next_fusion_time = 0.0
    fusion_period = 1.0 / sim_config.fusion_rate_hz

    current_time = 0.0
    while current_time <= sim_config.duration_s + 1.0e-12:
        for uav_id in range(sim_config.num_uavs):
            while next_observation_times[uav_id] <= current_time + 1.0e-12:
                capture_time = float(next_observation_times[uav_id])
                if capture_time <= sim_config.duration_s + 1.0e-12:
                    message = scenario.make_message(uav_id, int(sequences[uav_id]), capture_time)
                    network.transmit(message)
                    sequences[uav_id] += 1
                next_observation_times[uav_id] += observation_periods[uav_id]

        for message in network.deliver(current_time):
            fusion.receive(message, current_time)

        while next_fusion_time <= current_time + 1.0e-12:
            truth = scenario.target_position(float(next_fusion_time))
            result = fusion.fuse(float(next_fusion_time))
            recorder.record(result, truth, fusion.stats, network.stats)
            next_fusion_time += fusion_period

        current_time += sim_config.dt_s

    time_series = recorder.dataframe()
    summary = summarize_time_series(time_series)
    generated_messages = int(sequences.sum())
    bytes_per_s, kbit_per_s = payload_bandwidth(
        sim_config.num_uavs,
        float(np.mean(sim_config.observation_rates_hz)),
        payload_bytes,
    )
    summary.update(network.stats.as_dict())
    summary.update(fusion.stats.as_dict())
    summary.update(
        {
            "generated_messages": generated_messages,
            "payload_bytes": payload_bytes,
            "payload_bytes_per_s": bytes_per_s,
            "payload_kbit_per_s": kbit_per_s,
            "stale_rejection_rate": fusion.stats.stale_rejected_count / max(fusion.stats.received_count, 1),
        }
    )
    return time_series, summary


def run_ideal_baseline(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Experiment 1: ideal communication baseline."""
    base_sim = SimulationConfig(moving_target=False, target_speed_mps=0.0)
    network = NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=0.0)
    fusion = FusionConfig(sliding_window_s=1.0, timing_mode="capture_time")
    return _run_conditions(
        "ideal_baseline",
        exp_config,
        [("ideal", base_sim, network, fusion, {"condition": "ideal", "method": "tro"})],
        group_columns=["experiment", "condition", "method"],
        plotter=None,
    )


def run_packet_loss_sweep(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Experiment 2: packet loss sweep with sliding-window and latest-only methods."""
    conditions = []
    losses = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    for packet_loss in losses:
        for method, buffer_mode in [("tro_sliding_window", "sliding_window"), ("latest_only", "latest_only")]:
            sim = SimulationConfig(moving_target=False, target_speed_mps=0.0)
            network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=0.0)
            fusion = FusionConfig(sliding_window_s=1.0, timing_mode="capture_time", buffer_mode=buffer_mode)
            conditions.append((f"loss_{packet_loss}_{method}", sim, network, fusion, {"packet_loss": packet_loss, "method": method}))
    return _run_conditions("packet_loss", exp_config, conditions, ["experiment", "packet_loss", "method"], plot_packet_loss)


def run_delay_sweep(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Experiment 3: delay sweep comparing capture-time and arrival-time modes."""
    conditions = []
    for delay_s in [0.0, 0.05, 0.10, 0.25, 0.50]:
        for timing_mode in ["capture_time", "arrival_time"]:
            sim = SimulationConfig(moving_target=True, target_speed_mps=10.0)
            network = NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=delay_s)
            fusion = FusionConfig(sliding_window_s=1.0, timing_mode=timing_mode)
            conditions.append((f"delay_{delay_s}_{timing_mode}", sim, network, fusion, {"delay_s": delay_s, "timing_mode": timing_mode, "method": timing_mode}))
    return _run_conditions("delay", exp_config, conditions, ["experiment", "delay_s", "timing_mode", "method"], plot_delay)


def run_window_sweep(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Experiment 4: sliding-window duration sweep across target speeds."""
    conditions = []
    for speed in [0.0, 2.0, 10.0]:
        for window_s in [0.2, 0.5, 1.0, 2.0]:
            sim = SimulationConfig(moving_target=speed > 0.0, target_speed_mps=speed)
            network = NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=0.05)
            fusion = FusionConfig(sliding_window_s=window_s, timing_mode="capture_time")
            conditions.append((f"window_{window_s}_speed_{speed}", sim, network, fusion, {"sliding_window_s": window_s, "target_speed_mps": speed, "method": "tro"}))
    return _run_conditions("window", exp_config, conditions, ["experiment", "sliding_window_s", "target_speed_mps", "method"], plot_window)


def run_heterogeneous_rates(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Experiment 5: equal versus heterogeneous observation rates."""
    conditions = []
    rate_sets = [("equal_5hz", [5.0, 5.0, 5.0, 5.0]), ("heterogeneous", [10.0, 5.0, 2.0, 1.0])]
    for label, rates in rate_sets:
        sim = SimulationConfig(num_uavs=4, observation_rates_hz=rates, moving_target=True, target_speed_mps=2.0)
        network = NetworkConfig(packet_loss=0.1, delay_mode="uniform", uniform_delay_range_s=(0.0, 0.1))
        fusion = FusionConfig(sliding_window_s=1.0, timing_mode="capture_time")
        conditions.append((label, sim, network, fusion, {"rate_case": label, "method": "tro"}))
    return _run_conditions("heterogeneous_rates", exp_config, conditions, ["experiment", "rate_case", "method"], None)


def run_outlier_rejection(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Experiment 6: residual-gating robustness under false-ray outliers."""
    conditions = []
    for outlier_rate in [0.0, 0.05, 0.10, 0.20]:
        for gating in [True, False]:
            sim = SimulationConfig(moving_target=False, target_speed_mps=0.0, outlier_rate=outlier_rate)
            network = NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=0.0)
            fusion = FusionConfig(sliding_window_s=1.0, residual_gating=gating)
            method = "gating_on" if gating else "gating_off"
            conditions.append((f"outlier_{outlier_rate}_{method}", sim, network, fusion, {"outlier_rate": outlier_rate, "method": method}))
    return _run_conditions("outliers", exp_config, conditions, ["experiment", "outlier_rate", "method"], None)


def run_bandwidth_calculation(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Experiment 7: payload bandwidth table."""
    rows: list[dict[str, object]] = []
    for payload_bytes in [48, 64]:
        for num_uavs in [2, 4, 8, 16]:
            for rate_hz in [2, 5, 10]:
                bytes_per_s, kbit_per_s = payload_bandwidth(num_uavs, rate_hz, payload_bytes)
                rows.append(
                    {
                        "experiment": "bandwidth",
                        "payload_bytes": payload_bytes,
                        "num_uavs": num_uavs,
                        "rate_hz": rate_hz,
                        "bytes_per_s": bytes_per_s,
                        "kbit_per_s": kbit_per_s,
                        "label": f"{payload_bytes} B, {rate_hz} Hz",
                    }
                )
    output_dir = exp_config.output_dir
    ensure_output_dirs(output_dir)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "bandwidth_summary.csv", index=False)
    if exp_config.make_plots:
        plot_bandwidth(frame, output_dir)
    return frame


def run_smoke(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Run a short smoke experiment for runtime validation."""
    short_config = replace(exp_config, monte_carlo_runs=1)
    sim = SimulationConfig(duration_s=4.0, dt_s=0.02, moving_target=True, target_speed_mps=2.0)
    network = NetworkConfig(packet_loss=0.05, delay_mode="uniform", uniform_delay_range_s=(0.0, 0.05))
    fusion = FusionConfig(sliding_window_s=0.5, timing_mode="capture_time")
    return _run_conditions("smoke", short_config, [("smoke", sim, network, fusion, {"condition": "smoke", "method": "tro"})], ["experiment", "condition", "method"], None)


def run_validation_checks() -> None:
    """Run simple internal consistency checks requested for the simulator."""
    ideal_sim = SimulationConfig(duration_s=3.0, angular_noise_std_deg=0.0, position_noise_std_m=0.0)
    ideal_network = NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=0.0)
    ideal_fusion = FusionConfig(sliding_window_s=1.0, residual_gating=False)
    ideal_ts, ideal_summary = run_single_simulation(ideal_sim, ideal_network, ideal_fusion, seed=123)
    if not ideal_ts.empty:
        direction_norm_columns_ok = True
    else:
        direction_norm_columns_ok = False
    if not direction_norm_columns_ok or ideal_summary["rmse_m"] > 1.0e-5:
        raise AssertionError("ideal no-noise localization check failed")

    loss_ts, _ = run_single_simulation(
        SimulationConfig(duration_s=2.0),
        NetworkConfig(packet_loss=0.5, delay_mode="fixed", fixed_delay_s=0.0),
        FusionConfig(sliding_window_s=1.0),
        seed=124,
    )
    if loss_ts["packet_loss_count"].iloc[-1] <= 0:
        raise AssertionError("packet loss counter check failed")

    stale_ts, stale_summary = run_single_simulation(
        SimulationConfig(duration_s=2.0),
        NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=0.4),
        FusionConfig(sliding_window_s=0.2),
        seed=125,
    )
    if stale_summary["stale_rejected_count"] <= 0 or stale_ts["stale_rejected_count"].iloc[-1] <= 0:
        raise AssertionError("stale rejection check failed")

    moving_sim = SimulationConfig(duration_s=5.0, moving_target=True, target_speed_mps=10.0)
    capture_ts, capture_summary = run_single_simulation(
        moving_sim,
        NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=0.3),
        FusionConfig(sliding_window_s=1.0, timing_mode="capture_time"),
        seed=126,
    )
    arrival_ts, arrival_summary = run_single_simulation(
        moving_sim,
        NetworkConfig(packet_loss=0.0, delay_mode="fixed", fixed_delay_s=0.3),
        FusionConfig(sliding_window_s=1.0, timing_mode="arrival_time"),
        seed=126,
    )
    if abs(float(capture_summary["rmse_m"]) - float(arrival_summary["rmse_m"])) < 1.0e-6:
        raise AssertionError("arrival-time baseline did not differ from capture-time mode")


def run_all(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Run all paper-oriented experiments and write a consolidated summary."""
    frames = [
        run_ideal_baseline(exp_config),
        run_packet_loss_sweep(exp_config),
        run_delay_sweep(exp_config),
        run_window_sweep(exp_config),
        run_heterogeneous_rates(exp_config),
        run_outlier_rejection(exp_config),
        run_bandwidth_calculation(exp_config),
    ]
    summary = pd.concat(frames, ignore_index=True, sort=False)
    exp_config.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(exp_config.output_dir / "results_summary.csv", index=False)
    return summary


def _run_conditions(
    experiment_name: str,
    exp_config: ExperimentConfig,
    conditions: list[tuple[str, SimulationConfig, NetworkConfig, FusionConfig, dict[str, object]]],
    group_columns: list[str],
    plotter: Callable[[pd.DataFrame, Path], None] | None,
) -> pd.DataFrame:
    exp_config.validate()
    output_dir = exp_config.output_dir
    ensure_output_dirs(output_dir)
    run_summaries: list[dict[str, object]] = []
    time_series_frames: list[pd.DataFrame] = []

    total = len(conditions) * exp_config.monte_carlo_runs
    completed = 0
    for condition_index, (condition_name, sim, network, fusion, labels) in enumerate(conditions):
        for run in range(exp_config.monte_carlo_runs):
            seed = exp_config.seed + condition_index * 1000 + run
            completed += 1
            print(f"[{experiment_name}] run {completed}/{total}: {condition_name}, seed={seed}")
            time_series, summary = run_single_simulation(sim, network, fusion, seed=seed)
            row: dict[str, object] = {"experiment": experiment_name, "condition_name": condition_name, "run": run, "seed": seed}
            row.update(labels)
            row.update(summary)
            run_summaries.append(row)
            ts = time_series.copy()
            ts.insert(0, "experiment", experiment_name)
            ts.insert(1, "condition_name", condition_name)
            ts.insert(2, "run", run)
            for key, value in labels.items():
                ts[key] = value
            time_series_frames.append(ts)

    per_run = pd.DataFrame(run_summaries)
    aggregate = aggregate_runs(run_summaries, group_columns)
    per_run.to_csv(output_dir / f"{experiment_name}_per_run_summary.csv", index=False)
    aggregate.to_csv(output_dir / f"{experiment_name}_summary.csv", index=False)
    if time_series_frames:
        pd.concat(time_series_frames, ignore_index=True, sort=False).to_csv(output_dir / f"{experiment_name}_time_series.csv", index=False)
    if exp_config.make_plots and plotter is not None:
        plotter(aggregate, output_dir)
    if exp_config.make_plots and experiment_name in {"ideal_baseline", "smoke"} and time_series_frames:
        plot_trajectory(time_series_frames[0], output_dir, name=f"{experiment_name}_trajectory")
    return aggregate


EXPERIMENTS: dict[str, Callable[[ExperimentConfig], pd.DataFrame]] = {
    "ideal": run_ideal_baseline,
    "ideal_baseline": run_ideal_baseline,
    "packet_loss": run_packet_loss_sweep,
    "delay": run_delay_sweep,
    "window": run_window_sweep,
    "sliding_window": run_window_sweep,
    "rates": run_heterogeneous_rates,
    "heterogeneous_rates": run_heterogeneous_rates,
    "outliers": run_outlier_rejection,
    "outlier_rejection": run_outlier_rejection,
    "bandwidth": run_bandwidth_calculation,
    "smoke": run_smoke,
    "all": run_all,
}
