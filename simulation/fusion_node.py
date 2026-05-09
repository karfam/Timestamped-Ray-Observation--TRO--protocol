"""Fusion node for validating TRO messages and estimating target position."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import FusionConfig
from .models import BufferedObservation, EstimateResult, FusionStats
from .tro_message import TROMessage
from .utils import geometry_quality, normalize


class FusionNode:
    """Validates TRO messages, buffers observations, and fuses rays."""

    def __init__(self, config: FusionConfig):
        config.validate()
        self.config = config
        self.stats = FusionStats()
        self._buffer: list[BufferedObservation] = []
        self._seen_sequences: set[tuple[int, int, int]] = set()
        self._max_sequence_by_uav: dict[tuple[int, int], int] = {}
        self._latest_by_uav: dict[tuple[int, int], BufferedObservation] = {}

    def receive(self, message: TROMessage, current_time: float) -> None:
        """Validate and store a delivered message."""
        self.stats.received_count += 1
        if not self._valid_basic_fields(message):
            self.stats.invalid_count += 1
            return

        key = message.key
        if key in self._seen_sequences:
            self.stats.duplicate_count += 1
            return

        stream_key = (message.uav_id, message.target_id)
        previous_max = self._max_sequence_by_uav.get(stream_key)
        if previous_max is not None:
            if message.sequence < previous_max:
                self.stats.out_of_order_count += 1
            elif message.sequence > previous_max + 1:
                gap = message.sequence - previous_max - 1
                self.stats.sequence_gap_count += 1
                self.stats.estimated_packet_loss_count += gap
        self._max_sequence_by_uav[stream_key] = max(previous_max or message.sequence, message.sequence)
        self._seen_sequences.add(key)

        stale_limit = self.config.stale_time_s or self.config.sliding_window_s
        if current_time - message.capture_time > stale_limit + 1.0e-12:
            self.stats.stale_rejected_count += 1
            return

        timing_key = message.capture_time if self.config.timing_mode == "capture_time" else current_time
        observation = BufferedObservation(message=message, receive_time=current_time, timing_key=timing_key)
        self.stats.accepted_count += 1
        if self.config.buffer_mode == "latest_only":
            self._latest_by_uav[stream_key] = observation
            self._buffer = list(self._latest_by_uav.values())
        else:
            self._buffer.append(observation)

    def fuse(self, current_time: float) -> EstimateResult:
        """Run one weighted multi-ray least-squares fusion cycle."""
        active = self._active_observations(current_time)
        contributing_uavs = len({obs.message.uav_id for obs in active})
        ages = [current_time - obs.message.capture_time for obs in active]
        mean_age = float(np.mean(ages)) if ages else float("nan")
        max_age = float(np.max(ages)) if ages else float("nan")

        if len(active) < 2 or contributing_uavs < self.config.min_uavs_for_estimate:
            return EstimateResult(
                current_time=current_time,
                valid=False,
                estimate=None,
                active_rays=len(active),
                contributing_uavs=contributing_uavs,
                mean_observation_age=mean_age,
                max_observation_age=max_age,
                mean_residual=float("nan"),
                max_residual=float("nan"),
                gated_observations=0,
                condition_number=float("inf"),
                geometry_quality=geometry_quality([obs.message.ray_direction for obs in active]),
                confidence=0.0,
            )

        estimate, condition_number = self._least_squares(active)
        gated_count = 0
        used = active
        if estimate is not None and self.config.residual_gating:
            residuals = self._residuals(estimate, active)
            used = [obs for obs, residual in zip(active, residuals) if residual <= self.config.residual_threshold_m]
            gated_count = len(active) - len(used)
            if len(used) >= 2 and len({obs.message.uav_id for obs in used}) >= self.config.min_uavs_for_estimate:
                estimate, condition_number = self._least_squares(used)
            else:
                estimate = None
                condition_number = float("inf")

        if estimate is None or not np.all(np.isfinite(estimate)) or condition_number > self.config.max_condition_number:
            return EstimateResult(
                current_time=current_time,
                valid=False,
                estimate=None,
                active_rays=len(active),
                contributing_uavs=contributing_uavs,
                mean_observation_age=mean_age,
                max_observation_age=max_age,
                mean_residual=float("nan"),
                max_residual=float("nan"),
                gated_observations=gated_count,
                condition_number=condition_number,
                geometry_quality=geometry_quality([obs.message.ray_direction for obs in active]),
                confidence=0.0,
            )

        residuals = self._residuals(estimate, used)
        used_uavs = len({obs.message.uav_id for obs in used})
        return EstimateResult(
            current_time=current_time,
            valid=True,
            estimate=estimate,
            active_rays=len(active),
            contributing_uavs=used_uavs,
            mean_observation_age=mean_age,
            max_observation_age=max_age,
            mean_residual=float(np.mean(residuals)) if residuals else 0.0,
            max_residual=float(np.max(residuals)) if residuals else 0.0,
            gated_observations=gated_count,
            condition_number=condition_number,
            geometry_quality=geometry_quality([obs.message.ray_direction for obs in used]),
            confidence=1.0 / max(condition_number, 1.0),
            residuals=residuals,
        )

    def _valid_basic_fields(self, message: TROMessage) -> bool:
        if not message.crc_valid:
            return False
        if message.version != self.config.expected_version:
            return False
        if message.msg_type != self.config.expected_msg_type:
            return False
        if not np.all(np.isfinite(message.ray_origin)) or not np.all(np.isfinite(message.ray_direction)):
            return False
        direction_norm = float(np.linalg.norm(message.ray_direction))
        return abs(direction_norm - 1.0) < 1.0e-6

    def _active_observations(self, current_time: float) -> list[BufferedObservation]:
        cutoff = current_time - self.config.sliding_window_s
        if self.config.buffer_mode == "latest_only":
            active = [obs for obs in self._latest_by_uav.values() if obs.timing_key >= cutoff]
            self._buffer = active
            self._latest_by_uav = {(obs.message.uav_id, obs.message.target_id): obs for obs in active}
            return active
        self._buffer = [obs for obs in self._buffer if obs.timing_key >= cutoff]
        return list(self._buffer)

    def _weight(self, message: TROMessage) -> float:
        mode = self.config.weight_mode
        if mode == "equal":
            return 1.0
        if mode == "confidence":
            return max(message.confidence, self.config.weight_epsilon)
        if mode == "uncertainty":
            return 1.0 / max(message.angular_uncertainty**2, self.config.weight_epsilon)
        if mode == "combined":
            return max(message.confidence, self.config.weight_epsilon) / max(
                message.angular_uncertainty**2,
                self.config.weight_epsilon,
            )
        raise ValueError(f"unknown weight mode: {mode}")

    def _least_squares(self, observations: list[BufferedObservation]) -> tuple[np.ndarray | None, float]:
        matrix = np.zeros((3, 3), dtype=float)
        rhs = np.zeros(3, dtype=float)
        identity = np.eye(3)
        for observation in observations:
            message = observation.message
            direction = normalize(message.ray_direction)
            projection = identity - np.outer(direction, direction)
            weight = self._weight(message)
            matrix += weight * projection
            rhs += weight * projection @ message.ray_origin
        try:
            condition_number = float(np.linalg.cond(matrix))
            if not np.isfinite(condition_number) or condition_number > self.config.max_condition_number:
                return None, condition_number
            estimate = np.linalg.solve(matrix, rhs)
            return estimate, condition_number
        except np.linalg.LinAlgError:
            return None, float("inf")

    @staticmethod
    def _residuals(estimate: np.ndarray, observations: list[BufferedObservation]) -> list[float]:
        residuals: list[float] = []
        for observation in observations:
            origin = observation.message.ray_origin
            direction = observation.message.ray_direction
            projection = np.eye(3) - np.outer(direction, direction)
            residuals.append(float(np.linalg.norm(projection @ (estimate - origin))))
        return residuals

    def packet_stats(self) -> dict[str, int]:
        """Return cumulative packet/fusion validation statistics."""
        return self.stats.as_dict()
