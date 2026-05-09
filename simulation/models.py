"""Shared model dataclasses for fusion and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tro_message import TROMessage


@dataclass(slots=True)
class BufferedObservation:
    """A validated TRO observation stored by the fusion node."""

    message: TROMessage
    receive_time: float
    timing_key: float


@dataclass(slots=True)
class EstimateResult:
    """Output from one fusion cycle."""

    current_time: float
    valid: bool
    estimate: np.ndarray | None
    active_rays: int
    contributing_uavs: int
    mean_observation_age: float
    max_observation_age: float
    mean_residual: float
    max_residual: float
    gated_observations: int
    condition_number: float
    geometry_quality: float
    confidence: float
    residuals: list[float] = field(default_factory=list)


@dataclass(slots=True)
class FusionStats:
    """Cumulative message statistics collected by the fusion node."""

    received_count: int = 0
    accepted_count: int = 0
    invalid_count: int = 0
    sequence_gap_count: int = 0
    estimated_packet_loss_count: int = 0
    duplicate_count: int = 0
    out_of_order_count: int = 0
    stale_rejected_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "received_count": self.received_count,
            "accepted_count": self.accepted_count,
            "invalid_count": self.invalid_count,
            "sequence_gap_count": self.sequence_gap_count,
            "estimated_packet_loss_count": self.estimated_packet_loss_count,
            "duplicate_count": self.duplicate_count,
            "out_of_order_count": self.out_of_order_count,
            "stale_rejected_count": self.stale_rejected_count,
        }


@dataclass(slots=True)
class NetworkStats:
    """Cumulative statistics from the network model."""

    transmitted_count: int = 0
    dropped_count: int = 0
    queued_count: int = 0
    delivered_count: int = 0
    duplicated_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "network_transmitted_count": self.transmitted_count,
            "network_dropped_count": self.dropped_count,
            "network_queued_count": self.queued_count,
            "network_delivered_count": self.delivered_count,
            "network_duplicated_count": self.duplicated_count,
        }
