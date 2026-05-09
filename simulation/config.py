"""Configuration objects for the TRO simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


ROOT_OUTPUT_DIR = Path("outputs")
FIGURE_DIR = ROOT_OUTPUT_DIR / "figures"

DelayMode = Literal["fixed", "uniform", "normal"]
WeightMode = Literal["equal", "confidence", "uncertainty", "combined"]
TimingMode = Literal["capture_time", "arrival_time"]
BufferMode = Literal["sliding_window", "latest_only"]
GeometryMode = Literal["circle", "square"]
FusionMode = Literal["legacy_sliding_window", "synchronized_bearing_fusion", "tro_sliding_window_fusion"]


@dataclass(slots=True)
class SimulationConfig:
    """Scenario and ray-generation configuration."""

    num_uavs: int = 4
    duration_s: float = 60.0
    dt_s: float = 0.02
    fusion_rate_hz: float = 5.0
    observation_rates_hz: Sequence[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 5.0])
    observation_start_offsets_s: Sequence[float] | None = None
    target_speed_mps: float = 0.0
    moving_target: bool = False
    target_initial_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    uav_altitude_m: float = 80.0
    uav_radius_m: float = 250.0
    uav_geometry: GeometryMode = "circle"
    circular_uav_motion: bool = False
    uav_angular_speed_rad_s: float = 0.02
    angular_noise_std_deg: float = 0.5
    per_uav_angular_noise_std_deg: Sequence[float] | None = None
    position_noise_std_m: float = 0.5
    confidence_mean: float = 0.9
    per_uav_confidence_mean: Sequence[float] | None = None
    confidence_std: float = 0.05
    angular_uncertainty_deg: float = 0.5
    per_uav_angular_uncertainty_deg: Sequence[float] | None = None
    outlier_rate: float = 0.0
    seed: int = 7

    def validate(self) -> None:
        if self.num_uavs < 1:
            raise ValueError("num_uavs must be at least 1")
        if self.duration_s <= 0 or self.dt_s <= 0:
            raise ValueError("duration_s and dt_s must be positive")
        if self.fusion_rate_hz <= 0:
            raise ValueError("fusion_rate_hz must be positive")
        if len(self.observation_rates_hz) != self.num_uavs:
            raise ValueError("observation_rates_hz length must equal num_uavs")
        if self.observation_start_offsets_s is not None and len(self.observation_start_offsets_s) != self.num_uavs:
            raise ValueError("observation_start_offsets_s length must equal num_uavs")
        if self.per_uav_angular_noise_std_deg is not None and len(self.per_uav_angular_noise_std_deg) != self.num_uavs:
            raise ValueError("per_uav_angular_noise_std_deg length must equal num_uavs")
        if self.per_uav_confidence_mean is not None and len(self.per_uav_confidence_mean) != self.num_uavs:
            raise ValueError("per_uav_confidence_mean length must equal num_uavs")
        if self.per_uav_angular_uncertainty_deg is not None and len(self.per_uav_angular_uncertainty_deg) != self.num_uavs:
            raise ValueError("per_uav_angular_uncertainty_deg length must equal num_uavs")
        if any(rate <= 0 for rate in self.observation_rates_hz):
            raise ValueError("all observation rates must be positive")
        if self.observation_start_offsets_s is not None and any(offset < 0 for offset in self.observation_start_offsets_s):
            raise ValueError("observation_start_offsets_s values must be non-negative")
        if self.angular_noise_std_deg < 0 or self.position_noise_std_m < 0:
            raise ValueError("noise values must be non-negative")
        if self.per_uav_angular_noise_std_deg is not None and any(value < 0 for value in self.per_uav_angular_noise_std_deg):
            raise ValueError("per-UAV angular noise values must be non-negative")
        if self.per_uav_angular_uncertainty_deg is not None and any(value < 0 for value in self.per_uav_angular_uncertainty_deg):
            raise ValueError("per-UAV angular uncertainty values must be non-negative")
        if self.per_uav_confidence_mean is not None and any(not 0.0 <= value <= 1.0 for value in self.per_uav_confidence_mean):
            raise ValueError("per-UAV confidence means must be in [0, 1]")
        if not 0.0 <= self.outlier_rate <= 1.0:
            raise ValueError("outlier_rate must be in [0, 1]")


@dataclass(slots=True)
class NetworkConfig:
    """Network impairment model configuration."""

    packet_loss: float = 0.0
    delay_mode: DelayMode = "fixed"
    fixed_delay_s: float = 0.0
    uniform_delay_range_s: tuple[float, float] = (0.0, 0.1)
    normal_delay_mean_s: float = 0.05
    normal_delay_std_s: float = 0.02
    duplicate_probability: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.packet_loss <= 1.0:
            raise ValueError("packet_loss must be in [0, 1]")
        if not 0.0 <= self.duplicate_probability <= 1.0:
            raise ValueError("duplicate_probability must be in [0, 1]")
        if self.fixed_delay_s < 0:
            raise ValueError("fixed_delay_s must be non-negative")
        if self.uniform_delay_range_s[0] < 0 or self.uniform_delay_range_s[1] < self.uniform_delay_range_s[0]:
            raise ValueError("uniform_delay_range_s must be ordered and non-negative")
        if self.normal_delay_std_s < 0:
            raise ValueError("normal_delay_std_s must be non-negative")


@dataclass(slots=True)
class FusionConfig:
    """Fusion-node configuration."""

    sliding_window_s: float = 1.0
    sync_tolerance_s: float = 0.05
    fusion_mode: FusionMode = "legacy_sliding_window"
    timing_mode: TimingMode = "capture_time"
    buffer_mode: BufferMode = "sliding_window"
    weight_mode: WeightMode = "combined"
    residual_gating: bool = True
    residual_threshold_m: float = 20.0
    max_condition_number: float = 1.0e8
    expected_version: int = 1
    expected_msg_type: int = 1
    stale_time_s: float | None = None
    min_uavs_for_estimate: int = 2
    min_geometry_quality: float = 1.0e-3
    weight_epsilon: float = 1.0e-8

    def validate(self) -> None:
        if self.sliding_window_s <= 0:
            raise ValueError("sliding_window_s must be positive")
        if self.sync_tolerance_s < 0:
            raise ValueError("sync_tolerance_s must be non-negative")
        if self.residual_threshold_m <= 0:
            raise ValueError("residual_threshold_m must be positive")
        if self.max_condition_number <= 0:
            raise ValueError("max_condition_number must be positive")
        if self.min_uavs_for_estimate < 2:
            raise ValueError("min_uavs_for_estimate must be at least 2")
        if self.min_geometry_quality < 0:
            raise ValueError("min_geometry_quality must be non-negative")


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment execution options."""

    monte_carlo_runs: int = 20
    output_dir: Path = ROOT_OUTPUT_DIR
    make_plots: bool = True
    seed: int = 7
    duration_s: float | None = None

    def validate(self) -> None:
        if self.monte_carlo_runs < 1:
            raise ValueError("monte_carlo_runs must be at least 1")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration_s must be positive when provided")
