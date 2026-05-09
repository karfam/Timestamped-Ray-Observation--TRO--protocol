"""TRO message representation used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PayloadVariant = Literal[48, 64]


@dataclass(slots=True)
class TROMessage:
    """Timestamped Ray Observation message.

    The simulator keeps the message as a Python dataclass instead of serializing
    bytes. The payload-size helper is used for bandwidth calculations.
    """

    version: int
    msg_type: int
    flags: int
    sequence: int
    uav_id: int
    target_id: int
    capture_time: float
    transmit_time: float
    ray_origin: np.ndarray
    ray_direction: np.ndarray
    confidence: float
    angular_uncertainty: float
    crc_valid: bool = True

    def __post_init__(self) -> None:
        self.ray_origin = np.asarray(self.ray_origin, dtype=float).reshape(3)
        self.ray_direction = np.asarray(self.ray_direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(self.ray_direction))
        if norm == 0.0:
            raise ValueError("ray_direction cannot be zero")
        self.ray_direction = self.ray_direction / norm
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.angular_uncertainty <= 0:
            raise ValueError("angular_uncertainty must be positive")

    @property
    def key(self) -> tuple[int, int, int]:
        """Return a duplicate-detection key."""
        return (self.uav_id, self.target_id, self.sequence)


def estimate_payload_size_bytes(variant: PayloadVariant = 64) -> int:
    """Return the nominal compact TRO payload size in bytes."""
    if variant not in (48, 64):
        raise ValueError("variant must be 48 or 64 bytes")
    return int(variant)


def payload_bandwidth(num_uavs: int, rate_hz: float, payload_bytes: int) -> tuple[float, float]:
    """Return aggregate payload bandwidth as bytes/s and kbit/s."""
    bytes_per_second = float(num_uavs * rate_hz * payload_bytes)
    return bytes_per_second, bytes_per_second * 8.0 / 1000.0
