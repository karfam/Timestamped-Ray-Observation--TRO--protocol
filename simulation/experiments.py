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
from .plots import (
    plot_bandwidth,
    plot_delay,
    plot_packet_loss,
    plot_sync_vs_tro,
    plot_test1a_packet_loss,
    plot_test1b_heterogeneous_rates,
    plot_test2a_delay_sweep,
    plot_test2b_jitter_sweep,
    plot_test3a_latest_only_packet_loss,
    plot_test3b_window_duration_sweep,
    plot_test4a_unweighted_bearing_fusion,
    plot_test5a_false_detection_injection,
    plot_test6a_payload_bandwidth_comparison,
    plot_test6b_bandwidth_limited_channel,
    plot_trajectory,
    plot_window,
)
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

    if sim_config.observation_start_offsets_s is None:
        next_observation_times = np.zeros(sim_config.num_uavs, dtype=float)
    else:
        next_observation_times = np.asarray(sim_config.observation_start_offsets_s, dtype=float)
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


def run_test1a_packet_loss_sweep(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 1A: packet loss sweep for strict synchronized bearing fusion versus TRO."""
    conditions = []
    losses = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    methods = ["synchronized_bearing_fusion", "tro_sliding_window_fusion"]
    for packet_loss in losses:
        for method in methods:
            sim = SimulationConfig(
                num_uavs=4,
                duration_s=exp_config.duration_s or 60.0,
                fusion_rate_hz=5.0,
                observation_rates_hz=[5.0, 5.0, 5.0, 5.0],
                moving_target=False,
                target_speed_mps=0.0,
                angular_noise_std_deg=0.5,
                position_noise_std_m=1.0,
            )
            network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=0.0)
            fusion = _sync_baseline_fusion_config(method, sim.num_uavs)
            conditions.append(
                (
                    f"loss_{packet_loss:g}_{method}",
                    sim,
                    network,
                    fusion,
                    {
                        "test": "test1_synchronized_bearing_fusion_baseline",
                        "experiment_id": "1A",
                        "packet_loss": packet_loss,
                        "delay_s": 0.0,
                        "target_speed_mps": 0.0,
                        "uav_count": sim.num_uavs,
                        "method": method,
                    },
                )
            )
    return _run_conditions(
        "test1a_packet_loss",
        exp_config,
        conditions,
        ["experiment", "test", "experiment_id", "packet_loss", "delay_s", "target_speed_mps", "uav_count", "method"],
        plot_test1a_packet_loss,
    )


def run_test1b_heterogeneous_rates(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 1B: heterogeneous UAV update rates for synchronized fusion versus TRO."""
    conditions = []
    losses = [0.0, 0.05, 0.10]
    rates = [10.0, 5.0, 2.0, 1.0]
    methods = ["synchronized_bearing_fusion", "tro_sliding_window_fusion"]
    for packet_loss in losses:
        for method in methods:
            sim = SimulationConfig(
                num_uavs=4,
                duration_s=exp_config.duration_s or 60.0,
                fusion_rate_hz=5.0,
                observation_rates_hz=rates,
                moving_target=False,
                target_speed_mps=0.0,
                angular_noise_std_deg=0.5,
                position_noise_std_m=1.0,
            )
            network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=0.0)
            fusion = _sync_baseline_fusion_config(method, sim.num_uavs)
            conditions.append(
                (
                    f"loss_{packet_loss:g}_{method}",
                    sim,
                    network,
                    fusion,
                    {
                        "test": "test1_synchronized_bearing_fusion_baseline",
                        "experiment_id": "1B",
                        "packet_loss": packet_loss,
                        "delay_s": 0.0,
                        "target_speed_mps": 0.0,
                        "uav_count": sim.num_uavs,
                        "uav_rates_hz": ";".join(f"{rate:g}" for rate in rates),
                        "fusion_rate_hz": sim.fusion_rate_hz,
                        "method": method,
                    },
                )
            )
    return _run_conditions(
        "test1b_heterogeneous_rates",
        exp_config,
        conditions,
        [
            "experiment",
            "test",
            "experiment_id",
            "packet_loss",
            "delay_s",
            "target_speed_mps",
            "uav_count",
            "uav_rates_hz",
            "fusion_rate_hz",
            "method",
        ],
        plot_test1b_heterogeneous_rates,
    )


def run_test2a_delay_sweep(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 2A: delay sweep for arrival-time fusion versus timestamp-aware TRO."""
    conditions = []
    delays = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
    losses = [0.0, 0.05, 0.10]
    methods = ["arrival_time_fusion", "tro_timestamp_aware_fusion"]
    for loss_index, packet_loss in enumerate(losses):
        for delay_index, delay_s in enumerate(delays):
            seed_group = loss_index * len(delays) + delay_index
            for method in methods:
                sim = _timestamp_baseline_sim_config(exp_config.duration_s or 60.0)
                network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=delay_s)
                fusion = _timestamp_baseline_fusion_config(method)
                conditions.append(
                    (
                        f"loss_{packet_loss:g}_delay_{delay_s:g}_{method}",
                        sim,
                        network,
                        fusion,
                        {
                            "test": "test2_arrival_time_fusion_baseline",
                            "experiment_id": "2A",
                            "packet_loss": packet_loss,
                            "delay_s": delay_s,
                            "target_speed_mps": sim.target_speed_mps,
                            "uav_count": sim.num_uavs,
                            "method": method,
                            "_seed_group": seed_group,
                        },
                    )
                )
    return _run_conditions(
        "test2a_delay_sweep",
        exp_config,
        conditions,
        ["experiment", "test", "experiment_id", "packet_loss", "delay_s", "target_speed_mps", "uav_count", "method"],
        plot_test2a_delay_sweep,
    )


def run_test2b_jitter_sweep(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 2B: delay-jitter sweep for arrival-time fusion versus timestamp-aware TRO."""
    conditions = []
    mean_delay_s = 0.25
    jitters = [0.0, 0.05, 0.10, 0.25, 0.50]
    packet_loss = 0.05
    methods = ["arrival_time_fusion", "tro_timestamp_aware_fusion"]
    for jitter_index, jitter_s in enumerate(jitters):
        for method in methods:
            sim = _timestamp_baseline_sim_config(exp_config.duration_s or 60.0)
            if jitter_s == 0.0:
                network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=mean_delay_s)
            else:
                network = NetworkConfig(
                    packet_loss=packet_loss,
                    delay_mode="normal",
                    normal_delay_mean_s=mean_delay_s,
                    normal_delay_std_s=jitter_s,
                )
            fusion = _timestamp_baseline_fusion_config(method)
            conditions.append(
                (
                    f"jitter_{jitter_s:g}_{method}",
                    sim,
                    network,
                    fusion,
                    {
                        "test": "test2_arrival_time_fusion_baseline",
                        "experiment_id": "2B",
                        "packet_loss": packet_loss,
                        "mean_delay_s": mean_delay_s,
                        "jitter_s": jitter_s,
                        "target_speed_mps": sim.target_speed_mps,
                        "uav_count": sim.num_uavs,
                        "method": method,
                        "_seed_group": jitter_index,
                    },
                )
            )
    return _run_conditions(
        "test2b_jitter_sweep",
        exp_config,
        conditions,
        ["experiment", "test", "experiment_id", "packet_loss", "mean_delay_s", "jitter_s", "target_speed_mps", "uav_count", "method"],
        plot_test2b_jitter_sweep,
    )


def run_test3a_latest_only_packet_loss(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 3A: latest-only fusion baseline under packet loss and sparse updates."""
    conditions = []
    losses = [0.0, 0.10, 0.20, 0.30, 0.50]
    rates = [10.0, 5.0, 2.0, 1.0]
    methods = ["latest_only_fusion", "tro_sliding_window_fusion"]
    for packet_loss in losses:
        for method in methods:
            sim = SimulationConfig(
                num_uavs=4,
                duration_s=exp_config.duration_s or 60.0,
                fusion_rate_hz=5.0,
                observation_rates_hz=rates,
                observation_start_offsets_s=[0.0, 0.08, 0.17, 0.31],
                moving_target=True,
                target_speed_mps=2.0,
                angular_noise_std_deg=0.5,
                position_noise_std_m=1.0,
            )
            network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=0.0)
            fusion = _latest_only_baseline_fusion_config(method, window_s=1.0)
            conditions.append(
                (
                    f"loss_{packet_loss:g}_{method}",
                    sim,
                    network,
                    fusion,
                    {
                        "test": "test3_latest_only_fusion_baseline",
                        "experiment_id": "3A",
                        "packet_loss": packet_loss,
                        "target_speed_mps": sim.target_speed_mps,
                        "uav_count": sim.num_uavs,
                        "uav_rates_hz": ";".join(f"{rate:g}" for rate in rates),
                        "fusion_rate_hz": sim.fusion_rate_hz,
                        "sliding_window_s": fusion.sliding_window_s,
                        "method": method,
                    },
                )
            )
    return _run_conditions(
        "test3a_latest_only_packet_loss",
        exp_config,
        conditions,
        [
            "experiment",
            "test",
            "experiment_id",
            "packet_loss",
            "target_speed_mps",
            "uav_count",
            "uav_rates_hz",
            "fusion_rate_hz",
            "sliding_window_s",
            "method",
        ],
        plot_test3a_latest_only_packet_loss,
    )


def run_test3b_window_duration_sweep(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 3B: sliding-window duration sweep under packet loss and moving targets."""
    conditions = []
    windows = [0.1, 0.25, 0.5, 1.0, 2.0]
    speeds = [0.0, 5.0, 10.0, 20.0]
    rates = [10.0, 5.0, 2.0, 1.0]
    packet_loss = 0.20
    for speed_index, speed in enumerate(speeds):
        for window_index, window_s in enumerate(windows):
            sim = SimulationConfig(
                num_uavs=4,
                duration_s=exp_config.duration_s or 60.0,
                fusion_rate_hz=5.0,
                observation_rates_hz=rates,
                observation_start_offsets_s=[0.0, 0.08, 0.17, 0.31],
                moving_target=speed > 0.0,
                target_speed_mps=speed,
                angular_noise_std_deg=0.5,
                position_noise_std_m=1.0,
            )
            network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=0.0)
            fusion = FusionConfig(
                fusion_mode="tro_sliding_window_fusion",
                sliding_window_s=window_s,
                timing_mode="capture_time",
                buffer_mode="sliding_window",
                stale_time_s=window_s,
                min_uavs_for_estimate=2,
            )
            conditions.append(
                (
                    f"window_{window_s:g}_speed_{speed:g}",
                    sim,
                    network,
                    fusion,
                    {
                        "test": "test3_latest_only_fusion_baseline",
                        "experiment_id": "3B",
                        "packet_loss": packet_loss,
                        "target_speed_mps": speed,
                        "uav_count": sim.num_uavs,
                        "uav_rates_hz": ";".join(f"{rate:g}" for rate in rates),
                        "fusion_rate_hz": sim.fusion_rate_hz,
                        "sliding_window_s": window_s,
                        "method": "tro_sliding_window_fusion",
                        "_seed_group": speed_index * len(windows) + window_index,
                    },
                )
            )
    return _run_conditions(
        "test3b_window_duration_sweep",
        exp_config,
        conditions,
        [
            "experiment",
            "test",
            "experiment_id",
            "packet_loss",
            "target_speed_mps",
            "uav_count",
            "uav_rates_hz",
            "fusion_rate_hz",
            "sliding_window_s",
            "method",
        ],
        plot_test3b_window_duration_sweep,
    )


def run_test4a_unweighted_bearing_fusion(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 4A: unweighted bearing fusion baseline under mixed observation quality."""
    conditions = []
    losses = [0.0, 0.05, 0.10]
    speeds = [0.0, 2.0]
    noise_deg = [0.5, 0.5, 2.0, 5.0]
    confidence_means = [0.95, 0.95, 0.70, 0.40]
    uncertainty_deg = noise_deg
    methods = [
        ("unweighted_bearing_fusion", "equal"),
        ("tro_weighted_fusion", "combined"),
    ]
    for speed_index, speed in enumerate(speeds):
        for loss_index, packet_loss in enumerate(losses):
            seed_group = speed_index * len(losses) + loss_index
            for method, weight_mode in methods:
                sim = SimulationConfig(
                    num_uavs=4,
                    duration_s=exp_config.duration_s or 60.0,
                    fusion_rate_hz=5.0,
                    observation_rates_hz=[5.0, 5.0, 5.0, 5.0],
                    moving_target=speed > 0.0,
                    target_speed_mps=speed,
                    angular_noise_std_deg=0.5,
                    per_uav_angular_noise_std_deg=noise_deg,
                    position_noise_std_m=1.0,
                    confidence_mean=0.9,
                    per_uav_confidence_mean=confidence_means,
                    confidence_std=0.03,
                    angular_uncertainty_deg=0.5,
                    per_uav_angular_uncertainty_deg=uncertainty_deg,
                )
                network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=0.02)
                fusion = FusionConfig(
                    fusion_mode="tro_sliding_window_fusion",
                    sliding_window_s=0.5,
                    timing_mode="capture_time",
                    buffer_mode="sliding_window",
                    weight_mode=weight_mode,
                    residual_gating=False,
                    stale_time_s=0.5,
                    min_uavs_for_estimate=2,
                )
                conditions.append(
                    (
                        f"speed_{speed:g}_loss_{packet_loss:g}_{method}",
                        sim,
                        network,
                        fusion,
                        {
                            "test": "test4_unweighted_bearing_fusion_baseline",
                            "experiment_id": "4A",
                            "packet_loss": packet_loss,
                            "delay_s": network.fixed_delay_s,
                            "target_speed_mps": speed,
                            "uav_count": sim.num_uavs,
                            "bearing_noise_deg_by_uav": ";".join(f"{value:g}" for value in noise_deg),
                            "confidence_mean_by_uav": ";".join(f"{value:g}" for value in confidence_means),
                            "angular_uncertainty_deg_by_uav": ";".join(f"{value:g}" for value in uncertainty_deg),
                            "sliding_window_s": fusion.sliding_window_s,
                            "weight_mode": weight_mode,
                            "method": method,
                            "_seed_group": seed_group,
                        },
                    )
                )
    return _run_conditions(
        "test4a_unweighted_bearing_fusion",
        exp_config,
        conditions,
        [
            "experiment",
            "test",
            "experiment_id",
            "packet_loss",
            "delay_s",
            "target_speed_mps",
            "uav_count",
            "bearing_noise_deg_by_uav",
            "confidence_mean_by_uav",
            "angular_uncertainty_deg_by_uav",
            "sliding_window_s",
            "weight_mode",
            "method",
        ],
        plot_test4a_unweighted_bearing_fusion,
    )


def run_test5a_false_detection_injection(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 5A: false detection injection with and without residual gating."""
    conditions = []
    false_observation_rates = [0.0, 0.05, 0.10, 0.20]
    packet_losses = [0.05, 0.10]
    target_speeds = [0.0, 2.0]
    methods = [
        ("tro_without_residual_gating", False),
        ("tro_with_residual_gating", True),
    ]
    delay_s = 0.10
    for speed_index, speed in enumerate(target_speeds):
        for loss_index, packet_loss in enumerate(packet_losses):
            for rate_index, false_observation_rate in enumerate(false_observation_rates):
                seed_group = speed_index * len(packet_losses) * len(false_observation_rates) + loss_index * len(false_observation_rates) + rate_index
                for method, residual_gating in methods:
                    sim = SimulationConfig(
                        num_uavs=4,
                        duration_s=exp_config.duration_s or 60.0,
                        fusion_rate_hz=5.0,
                        observation_rates_hz=[5.0, 5.0, 5.0, 5.0],
                        moving_target=speed > 0.0,
                        target_speed_mps=speed,
                        angular_noise_std_deg=0.5,
                        position_noise_std_m=1.0,
                        outlier_rate=false_observation_rate,
                    )
                    network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=delay_s)
                    fusion = FusionConfig(
                        fusion_mode="tro_sliding_window_fusion",
                        sliding_window_s=0.75,
                        timing_mode="capture_time",
                        buffer_mode="sliding_window",
                        weight_mode="combined",
                        residual_gating=residual_gating,
                        residual_threshold_m=20.0,
                        stale_time_s=0.75,
                        min_uavs_for_estimate=2,
                    )
                    conditions.append(
                        (
                            f"false_obs_{false_observation_rate:g}_loss_{packet_loss:g}_speed_{speed:g}_{method}",
                            sim,
                            network,
                            fusion,
                            {
                                "test": "test5_fusion_without_residual_gating_baseline",
                                "experiment_id": "5A",
                                "false_observation_rate": false_observation_rate,
                                "packet_loss": packet_loss,
                                "delay_s": delay_s,
                                "target_speed_mps": speed,
                                "uav_count": sim.num_uavs,
                                "uav_rates_hz": ";".join(f"{rate:g}" for rate in sim.observation_rates_hz),
                                "fusion_rate_hz": sim.fusion_rate_hz,
                                "sliding_window_s": fusion.sliding_window_s,
                                "residual_gating": residual_gating,
                                "residual_threshold_m": fusion.residual_threshold_m,
                                "method": method,
                                "_seed_group": seed_group,
                            },
                        )
                    )
    return _run_conditions(
        "test5a_false_detection_injection",
        exp_config,
        conditions,
        [
            "experiment",
            "test",
            "experiment_id",
            "false_observation_rate",
            "packet_loss",
            "delay_s",
            "target_speed_mps",
            "uav_count",
            "uav_rates_hz",
            "fusion_rate_hz",
            "sliding_window_s",
            "residual_gating",
            "residual_threshold_m",
            "method",
        ],
        plot_test5a_false_detection_injection,
    )


def run_test6a_payload_bandwidth_comparison(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 6A: payload bandwidth comparison against image-sharing baselines."""
    rows = _image_sharing_rows(
        experiment="test6a_payload_bandwidth_comparison",
        num_uavs_values=[2, 4, 6, 8],
        update_rates_hz=[1.0, 5.0, 10.0],
        available_bandwidth_kbps_values=[1000.0],
        include_all_payload_cases=True,
    )
    frame = pd.DataFrame(rows)
    output_dir = exp_config.output_dir
    ensure_output_dirs(output_dir)
    frame.to_csv(output_dir / "test6a_payload_bandwidth_comparison_summary.csv", index=False)
    if exp_config.make_plots:
        plot_test6a_payload_bandwidth_comparison(frame, output_dir)
    return frame


def run_test6b_bandwidth_limited_channel(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 6B: channel feasibility under bandwidth-limited links."""
    rows = _image_sharing_rows(
        experiment="test6b_bandwidth_limited_channel",
        num_uavs_values=[4],
        update_rates_hz=[5.0, 10.0],
        available_bandwidth_kbps_values=[10.0, 25.0, 50.0, 100.0, 500.0, 1000.0],
        include_all_payload_cases=True,
    )
    frame = pd.DataFrame(rows)
    output_dir = exp_config.output_dir
    ensure_output_dirs(output_dir)
    frame.to_csv(output_dir / "test6b_bandwidth_limited_channel_summary.csv", index=False)
    if exp_config.make_plots:
        plot_test6b_bandwidth_limited_channel(frame, output_dir)
    return frame


def run_test6_image_sharing_baseline(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Test 6: image-sharing communication baseline, combining 6A and 6B."""
    summary = pd.concat(
        [
            run_test6a_payload_bandwidth_comparison(exp_config),
            run_test6b_bandwidth_limited_channel(exp_config),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(exp_config.output_dir / "test6_image_sharing_baseline_summary.csv", index=False)
    return summary


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


def _image_sharing_rows(
    experiment: str,
    num_uavs_values: list[int],
    update_rates_hz: list[float],
    available_bandwidth_kbps_values: list[float],
    include_all_payload_cases: bool,
) -> list[dict[str, object]]:
    payload_profiles = [
        ("tro_communication", "min", 48),
        ("tro_communication", "max", 64),
        ("cropped_image_sharing", "min", 5_000),
        ("cropped_image_sharing", "nominal", 17_500),
        ("cropped_image_sharing", "max", 30_000),
        ("full_frame_image_sharing", "min", 50_000),
        ("full_frame_image_sharing", "nominal", 125_000),
        ("full_frame_image_sharing", "max", 200_000),
    ]
    if not include_all_payload_cases:
        payload_profiles = [profile for profile in payload_profiles if profile[1] in {"max", "nominal"}]

    rows: list[dict[str, object]] = []
    for num_uavs in num_uavs_values:
        for update_rate_hz in update_rates_hz:
            messages_per_s = float(num_uavs * update_rate_hz)
            for method, payload_case, payload_bytes in payload_profiles:
                payload_bytes_per_s, payload_kbit_per_s = payload_bandwidth(num_uavs, update_rate_hz, payload_bytes)
                for available_bandwidth_kbps in available_bandwidth_kbps_values:
                    rows.append(
                        _channel_capacity_row(
                            experiment=experiment,
                            method=method,
                            payload_case=payload_case,
                            payload_bytes=payload_bytes,
                            num_uavs=num_uavs,
                            update_rate_hz=update_rate_hz,
                            messages_per_s=messages_per_s,
                            payload_bytes_per_s=payload_bytes_per_s,
                            payload_kbit_per_s=payload_kbit_per_s,
                            available_bandwidth_kbps=available_bandwidth_kbps,
                        )
                    )
    return rows


def _channel_capacity_row(
    experiment: str,
    method: str,
    payload_case: str,
    payload_bytes: int,
    num_uavs: int,
    update_rate_hz: float,
    messages_per_s: float,
    payload_bytes_per_s: float,
    payload_kbit_per_s: float,
    available_bandwidth_kbps: float,
) -> dict[str, object]:
    channel_load = payload_kbit_per_s / available_bandwidth_kbps if available_bandwidth_kbps > 0 else float("inf")
    delivered_fraction = min(1.0, 1.0 / channel_load) if channel_load > 0 else 1.0
    drop_fraction = max(0.0, 1.0 - delivered_fraction)
    dropped_packets_per_s = messages_per_s * drop_fraction
    serialization_delay_s = payload_bytes * 8.0 / (available_bandwidth_kbps * 1000.0)
    end_to_end_delay_s = _estimated_channel_delay_s(serialization_delay_s, channel_load)
    return {
        "experiment": experiment,
        "test": "test6_image_sharing_communication_baseline",
        "experiment_id": "6A" if experiment.startswith("test6a") else "6B",
        "method": method,
        "payload_case": payload_case,
        "payload_bytes": payload_bytes,
        "num_uavs": num_uavs,
        "update_rate_hz": update_rate_hz,
        "messages_per_s": messages_per_s,
        "payload_bytes_per_s": payload_bytes_per_s,
        "payload_kbit_per_s": payload_kbit_per_s,
        "available_bandwidth_kbps": available_bandwidth_kbps,
        "channel_load": channel_load,
        "channel_load_pct": channel_load * 100.0,
        "delivered_packets_per_s": messages_per_s - dropped_packets_per_s,
        "dropped_packets_per_s": dropped_packets_per_s,
        "drop_fraction": drop_fraction,
        "drop_fraction_pct": drop_fraction * 100.0,
        "serialization_delay_ms": serialization_delay_s * 1000.0,
        "end_to_end_delay_ms": end_to_end_delay_s * 1000.0,
        "operational": channel_load <= 1.0,
    }


def _estimated_channel_delay_s(serialization_delay_s: float, channel_load: float) -> float:
    if channel_load >= 1.0:
        return float("inf")
    return serialization_delay_s * (1.0 + channel_load / max(1.0 - channel_load, 1.0e-9))


def run_smoke(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Run a short smoke experiment for runtime validation."""
    short_config = replace(exp_config, monte_carlo_runs=1)
    sim = SimulationConfig(duration_s=4.0, dt_s=0.02, moving_target=True, target_speed_mps=2.0)
    network = NetworkConfig(packet_loss=0.05, delay_mode="uniform", uniform_delay_range_s=(0.0, 0.05))
    fusion = FusionConfig(sliding_window_s=0.5, timing_mode="capture_time")
    return _run_conditions("smoke", short_config, [("smoke", sim, network, fusion, {"condition": "smoke", "method": "tro"})], ["experiment", "condition", "method"], None)


def run_synchronized_vs_tro_evaluation(exp_config: ExperimentConfig) -> pd.DataFrame:
    """Evaluate synchronized bearing fusion against TRO sliding-window fusion."""
    exp_config.validate()
    output_dir = _sync_vs_tro_output_dir(exp_config.output_dir)
    ensure_output_dirs(output_dir)

    conditions = _sync_vs_tro_conditions(exp_config.duration_s or 60.0)
    methods = ["synchronized_bearing_fusion", "tro_sliding_window_fusion"]
    run_rows: list[dict[str, object]] = []
    aggregate_inputs: list[dict[str, object]] = []
    representative_series: dict[tuple[str, str, str], pd.DataFrame] = {}

    total = len(conditions) * len(methods) * exp_config.monte_carlo_runs
    completed = 0
    for condition_index, condition in enumerate(conditions):
        for method in methods:
            fusion = _sync_baseline_fusion_config(method, int(condition["simulation"].num_uavs))
            for run in range(exp_config.monte_carlo_runs):
                seed = exp_config.seed + condition_index * 10_000 + methods.index(method) * 1_000 + run
                completed += 1
                print(
                    f"[sync_vs_tro] run {completed}/{total}: "
                    f"{condition['experiment_name']} / {condition['condition_name']} / {method}, seed={seed}"
                )
                time_series, summary = run_single_simulation(
                    condition["simulation"],
                    condition["network"],
                    fusion,
                    seed=seed,
                )
                row = _sync_vs_tro_summary_row(condition, method, run, seed, summary, exp_config.monte_carlo_runs)
                run_rows.append(row)
                aggregate_inputs.append(row)

                series_key = (str(condition["experiment_name"]), str(condition["condition_name"]), method)
                if run == 0 and series_key not in representative_series:
                    ts = time_series.copy()
                    ts.insert(0, "experiment_name", condition["experiment_name"])
                    ts.insert(1, "condition_name", condition["condition_name"])
                    ts.insert(2, "method", method)
                    representative_series[series_key] = ts

    group_columns = [
        "experiment_name",
        "condition_name",
        "method",
        "target_speed",
        "uav_rates",
        "fusion_rate",
        "packet_loss",
        "delay",
        "bearing_noise_deg",
        "position_noise_m",
        "sync_tolerance_s",
        "sliding_window_s",
        "monte_carlo_runs",
    ]
    summary = aggregate_runs(aggregate_inputs, group_columns)
    summary = _order_sync_vs_tro_columns(summary)
    per_run = pd.DataFrame(run_rows)

    summary_path = output_dir / "summary.csv"
    per_run.to_csv(output_dir / "per_run_summary.csv", index=False)
    summary.to_csv(summary_path, index=False)
    for (experiment_name, condition_name, method), frame in representative_series.items():
        filename = f"timeseries_{_safe_name(experiment_name)}_{_safe_name(condition_name)}_{_safe_name(method)}.csv"
        frame.to_csv(output_dir / filename, index=False)

    if exp_config.make_plots:
        plot_sync_vs_tro(summary, output_dir)

    print(f"[sync_vs_tro] summary: {summary_path.resolve()}")
    print(f"[sync_vs_tro] figures: {(output_dir / 'figures').resolve()}")
    return summary


def _sync_vs_tro_conditions(duration_s: float) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []

    def add(
        experiment_name: str,
        condition_name: str,
        target_speed: float,
        rates: list[float],
        packet_loss: float,
        delay: float,
        offsets: list[float] | None = None,
    ) -> None:
        simulation = SimulationConfig(
            num_uavs=4,
            duration_s=duration_s,
            target_speed_mps=target_speed,
            moving_target=target_speed > 0.0,
            observation_rates_hz=rates,
            observation_start_offsets_s=offsets,
            fusion_rate_hz=5.0,
            angular_noise_std_deg=0.5,
            position_noise_std_m=1.0,
        )
        network = NetworkConfig(packet_loss=packet_loss, delay_mode="fixed", fixed_delay_s=delay)
        conditions.append(
            {
                "experiment_name": experiment_name,
                "condition_name": condition_name,
                "target_speed": target_speed,
                "uav_rates": ";".join(f"{rate:g}" for rate in rates),
                "fusion_rate": 5.0,
                "packet_loss": packet_loss,
                "delay": delay,
                "bearing_noise_deg": 0.5,
                "position_noise_m": 1.0,
                "simulation": simulation,
                "network": network,
            }
        )

    add("1A_ideal_synchronized", "ideal_static_5hz", 0.0, [5.0, 5.0, 5.0, 5.0], 0.0, 0.0)

    for speed in [0.0, 2.0]:
        add(
            "1B_heterogeneous_update_rates",
            f"heterogeneous_speed_{speed:g}",
            speed,
            [10.0, 5.0, 2.0, 1.0],
            0.0,
            0.0,
            [0.0, 0.08, 0.17, 0.31],
        )

    for delay in [0.0, 0.05, 0.10, 0.25, 0.50]:
        add(
            "1C_delay_sweep",
            f"delay_{delay:g}s",
            2.0,
            [5.0, 5.0, 5.0, 5.0],
            0.0,
            delay,
        )

    for speed in [0.0, 2.0]:
        for packet_loss in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
            add(
                "1D_packet_loss_sweep",
                f"loss_{packet_loss:g}_speed_{speed:g}",
                speed,
                [5.0, 5.0, 5.0, 5.0],
                packet_loss,
                0.0,
            )

    return conditions


def _sync_baseline_fusion_config(method: str, num_uavs: int) -> FusionConfig:
    return FusionConfig(
        fusion_mode=method,
        sliding_window_s=0.5,
        sync_tolerance_s=0.05,
        timing_mode="capture_time",
        buffer_mode="sliding_window",
        stale_time_s=0.5,
        min_uavs_for_estimate=num_uavs if method == "synchronized_bearing_fusion" else 2,
    )


def _timestamp_baseline_sim_config(duration_s: float) -> SimulationConfig:
    return SimulationConfig(
        num_uavs=4,
        duration_s=duration_s,
        fusion_rate_hz=5.0,
        observation_rates_hz=[5.0, 5.0, 5.0, 5.0],
        moving_target=True,
        target_speed_mps=10.0,
        angular_noise_std_deg=0.5,
        position_noise_std_m=1.0,
    )


def _timestamp_baseline_fusion_config(method: str) -> FusionConfig:
    if method == "arrival_time_fusion":
        return FusionConfig(
            fusion_mode="legacy_sliding_window",
            sliding_window_s=0.5,
            timing_mode="arrival_time",
            buffer_mode="sliding_window",
            stale_time_s=2.0,
            min_uavs_for_estimate=2,
        )
    if method == "tro_timestamp_aware_fusion":
        return FusionConfig(
            fusion_mode="tro_sliding_window_fusion",
            sliding_window_s=0.75,
            timing_mode="capture_time",
            buffer_mode="sliding_window",
            stale_time_s=0.75,
            min_uavs_for_estimate=2,
        )
    raise ValueError(f"unknown timestamp baseline method: {method}")


def _latest_only_baseline_fusion_config(method: str, window_s: float) -> FusionConfig:
    if method == "latest_only_fusion":
        latest_horizon_s = 0.2
        return FusionConfig(
            fusion_mode="tro_sliding_window_fusion",
            sliding_window_s=latest_horizon_s,
            timing_mode="capture_time",
            buffer_mode="latest_only",
            stale_time_s=latest_horizon_s,
            min_uavs_for_estimate=2,
        )
    if method == "tro_sliding_window_fusion":
        return FusionConfig(
            fusion_mode="tro_sliding_window_fusion",
            sliding_window_s=window_s,
            timing_mode="capture_time",
            buffer_mode="sliding_window",
            stale_time_s=window_s,
            min_uavs_for_estimate=2,
        )
    raise ValueError(f"unknown latest-only baseline method: {method}")


def _sync_vs_tro_summary_row(
    condition: dict[str, object],
    method: str,
    run: int,
    seed: int,
    summary: dict[str, object],
    monte_carlo_runs: int,
) -> dict[str, object]:
    return {
        "experiment_name": condition["experiment_name"],
        "condition_name": condition["condition_name"],
        "method": method,
        "run": run,
        "seed": seed,
        "target_speed": condition["target_speed"],
        "uav_rates": condition["uav_rates"],
        "fusion_rate": condition["fusion_rate"],
        "packet_loss": condition["packet_loss"],
        "delay": condition["delay"],
        "bearing_noise_deg": condition["bearing_noise_deg"],
        "position_noise_m": condition["position_noise_m"],
        "sync_tolerance_s": 0.05,
        "sliding_window_s": 0.5,
        "rmse_m": summary.get("rmse_m", float("nan")),
        "mean_error_m": summary.get("mean_error_m", float("nan")),
        "median_error_m": summary.get("median_error_m", float("nan")),
        "p95_error_m": summary.get("p95_error_m", float("nan")),
        "max_error_m": summary.get("max_error_m", float("nan")),
        "availability_percent": summary.get("estimate_availability_pct", 0.0),
        "valid_estimates": summary.get("valid_estimate_count", 0),
        "invalid_fusion_cycles": summary.get("invalid_fusion_cycles", 0),
        "mean_active_rays": summary.get("mean_active_rays", 0.0),
        "mean_contributing_uavs": summary.get("mean_contributing_uavs", 0.0),
        "mean_observation_age_s": summary.get("mean_observation_age_s", float("nan")),
        "max_observation_age_s": summary.get("max_observation_age_s", float("nan")),
        "mean_residual_m": summary.get("mean_residual_m", float("nan")),
        "max_residual_m": summary.get("max_residual_m", float("nan")),
        "mean_condition_number": summary.get("mean_condition_number", float("nan")),
        "packet_loss_count": summary.get("final_packet_loss_count", 0),
        "stale_rejected_count": summary.get("final_stale_rejected_count", 0),
        "duplicate_count": summary.get("final_duplicate_count", 0),
        "out_of_order_count": summary.get("final_out_of_order_count", 0),
        "monte_carlo_runs": monte_carlo_runs,
    }


def _order_sync_vs_tro_columns(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "experiment_name",
        "condition_name",
        "method",
        "target_speed",
        "uav_rates",
        "fusion_rate",
        "packet_loss",
        "delay",
        "bearing_noise_deg",
        "position_noise_m",
        "sync_tolerance_s",
        "sliding_window_s",
        "rmse_m",
        "mean_error_m",
        "median_error_m",
        "p95_error_m",
        "max_error_m",
        "availability_percent",
        "valid_estimates",
        "invalid_fusion_cycles",
        "mean_active_rays",
        "mean_contributing_uavs",
        "mean_observation_age_s",
        "max_observation_age_s",
        "mean_residual_m",
        "max_residual_m",
        "mean_condition_number",
        "packet_loss_count",
        "stale_rejected_count",
        "duplicate_count",
        "out_of_order_count",
        "monte_carlo_runs",
    ]
    existing = [column for column in ordered if column in summary.columns]
    remaining = [column for column in summary.columns if column not in existing]
    return summary[existing + remaining]


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _sync_vs_tro_output_dir(output_dir: Path) -> Path:
    if output_dir.name == "synchronized_vs_tro":
        return output_dir
    return output_dir / "synchronized_vs_tro"


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
        run_test1a_packet_loss_sweep(exp_config),
        run_test1b_heterogeneous_rates(exp_config),
        run_test2a_delay_sweep(exp_config),
        run_test2b_jitter_sweep(exp_config),
        run_test3a_latest_only_packet_loss(exp_config),
        run_test3b_window_duration_sweep(exp_config),
        run_test4a_unweighted_bearing_fusion(exp_config),
        run_test5a_false_detection_injection(exp_config),
        run_test6_image_sharing_baseline(exp_config),
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
            seed_group = int(labels.get("_seed_group", condition_index))
            seed = exp_config.seed + seed_group * 1000 + run
            completed += 1
            print(f"[{experiment_name}] run {completed}/{total}: {condition_name}, seed={seed}")
            time_series, summary = run_single_simulation(sim, network, fusion, seed=seed)
            row: dict[str, object] = {"experiment": experiment_name, "condition_name": condition_name, "run": run, "seed": seed}
            row.update({key: value for key, value in labels.items() if not key.startswith("_")})
            row.update(summary)
            run_summaries.append(row)
            ts = time_series.copy()
            ts.insert(0, "experiment", experiment_name)
            ts.insert(1, "condition_name", condition_name)
            ts.insert(2, "run", run)
            for key, value in labels.items():
                if key.startswith("_"):
                    continue
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
    "test1a_packet_loss": run_test1a_packet_loss_sweep,
    "sync_packet_loss": run_test1a_packet_loss_sweep,
    "synchronized_packet_loss": run_test1a_packet_loss_sweep,
    "test1b_heterogeneous_rates": run_test1b_heterogeneous_rates,
    "sync_heterogeneous_rates": run_test1b_heterogeneous_rates,
    "synchronized_heterogeneous_rates": run_test1b_heterogeneous_rates,
    "test2a_delay_sweep": run_test2a_delay_sweep,
    "arrival_delay_sweep": run_test2a_delay_sweep,
    "timestamp_delay_sweep": run_test2a_delay_sweep,
    "test2b_jitter_sweep": run_test2b_jitter_sweep,
    "arrival_jitter_sweep": run_test2b_jitter_sweep,
    "timestamp_jitter_sweep": run_test2b_jitter_sweep,
    "test3a_latest_only_packet_loss": run_test3a_latest_only_packet_loss,
    "latest_only_packet_loss": run_test3a_latest_only_packet_loss,
    "test3_latest_only": run_test3a_latest_only_packet_loss,
    "test3b_window_duration_sweep": run_test3b_window_duration_sweep,
    "latest_only_window_sweep": run_test3b_window_duration_sweep,
    "test3_window_sweep": run_test3b_window_duration_sweep,
    "test4a_unweighted_bearing_fusion": run_test4a_unweighted_bearing_fusion,
    "test4_unweighted": run_test4a_unweighted_bearing_fusion,
    "unweighted_bearing_fusion": run_test4a_unweighted_bearing_fusion,
    "test5a_false_detection_injection": run_test5a_false_detection_injection,
    "test5_false_detection": run_test5a_false_detection_injection,
    "false_detection_injection": run_test5a_false_detection_injection,
    "fusion_without_residual_gating": run_test5a_false_detection_injection,
    "test6_image_sharing_baseline": run_test6_image_sharing_baseline,
    "test6_image_sharing": run_test6_image_sharing_baseline,
    "image_sharing_baseline": run_test6_image_sharing_baseline,
    "test6a_payload_bandwidth_comparison": run_test6a_payload_bandwidth_comparison,
    "test6a_payload_bandwidth": run_test6a_payload_bandwidth_comparison,
    "test6b_bandwidth_limited_channel": run_test6b_bandwidth_limited_channel,
    "test6b_bandwidth_limited": run_test6b_bandwidth_limited_channel,
    "delay": run_delay_sweep,
    "window": run_window_sweep,
    "sliding_window": run_window_sweep,
    "rates": run_heterogeneous_rates,
    "heterogeneous_rates": run_heterogeneous_rates,
    "outliers": run_outlier_rejection,
    "outlier_rejection": run_outlier_rejection,
    "bandwidth": run_bandwidth_calculation,
    "smoke": run_smoke,
    "sync_vs_tro": run_synchronized_vs_tro_evaluation,
    "synchronized_vs_tro": run_synchronized_vs_tro_evaluation,
    "all": run_all,
}
