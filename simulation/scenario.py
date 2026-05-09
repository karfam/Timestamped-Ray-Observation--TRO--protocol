"""Scenario generation and noisy ray creation."""

from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig
from .tro_message import TROMessage
from .utils import add_angular_noise, normalize, random_unit_vector


class Scenario:
    """Generates UAV poses, target states, and TRO ray observations."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        config.validate()
        self.config = config
        self.rng = rng
        self._base_uav_positions = self._initial_uav_positions()
        speed = config.target_speed_mps if config.moving_target or config.target_speed_mps > 0 else 0.0
        self._target_velocity = np.array([speed, 0.25 * speed, 0.0], dtype=float)

    def target_position(self, time_s: float) -> np.ndarray:
        """Return target position at the requested simulation time."""
        return np.asarray(self.config.target_initial_position_m, dtype=float) + self._target_velocity * time_s

    def uav_position(self, uav_id: int, time_s: float) -> np.ndarray:
        """Return UAV position at the requested simulation time."""
        base = self._base_uav_positions[uav_id]
        if not self.config.circular_uav_motion:
            return base.copy()
        center = self.target_position(time_s)
        rel = base - np.asarray(self.config.target_initial_position_m, dtype=float)
        angle = self.config.uav_angular_speed_rad_s * time_s
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return center + rotation @ rel

    def uav_positions(self, time_s: float) -> list[np.ndarray]:
        """Return all UAV positions."""
        return [self.uav_position(uav_id, time_s) for uav_id in range(self.config.num_uavs)]

    def make_message(self, uav_id: int, sequence: int, capture_time: float) -> TROMessage:
        """Generate one noisy TRO ray message for a UAV observation."""
        true_origin = self.uav_position(uav_id, capture_time)
        target = self.target_position(capture_time)
        noisy_origin = true_origin + self.rng.normal(0.0, self.config.position_noise_std_m, size=3)

        if self.rng.random() < self.config.outlier_rate:
            direction = random_unit_vector(self.rng)
            flags = 1
            confidence = 0.25
        else:
            ideal_direction = normalize(target - true_origin)
            direction = add_angular_noise(
                ideal_direction,
                math.radians(self.config.angular_noise_std_deg),
                self.rng,
            )
            flags = 0
            confidence = float(
                np.clip(
                    self.rng.normal(self.config.confidence_mean, self.config.confidence_std),
                    0.05,
                    1.0,
                )
            )

        angular_uncertainty = max(math.radians(self.config.angular_uncertainty_deg), 1.0e-6)
        return TROMessage(
            version=1,
            msg_type=1,
            flags=flags,
            sequence=sequence,
            uav_id=uav_id,
            target_id=0,
            capture_time=float(capture_time),
            transmit_time=float(capture_time),
            ray_origin=noisy_origin,
            ray_direction=direction,
            confidence=confidence,
            angular_uncertainty=angular_uncertainty,
        )

    def _initial_uav_positions(self) -> list[np.ndarray]:
        center = np.asarray(self.config.target_initial_position_m, dtype=float)
        radius = self.config.uav_radius_m
        altitude = self.config.uav_altitude_m
        positions: list[np.ndarray] = []
        if self.config.uav_geometry == "square" and self.config.num_uavs == 4:
            offsets = [(-radius, -radius), (radius, -radius), (radius, radius), (-radius, radius)]
            for x_offset, y_offset in offsets:
                positions.append(center + np.array([x_offset, y_offset, altitude], dtype=float))
            return positions

        for uav_id in range(self.config.num_uavs):
            angle = 2.0 * math.pi * uav_id / self.config.num_uavs
            positions.append(
                center
                + np.array(
                    [radius * math.cos(angle), radius * math.sin(angle), altitude],
                    dtype=float,
                )
            )
        return positions
