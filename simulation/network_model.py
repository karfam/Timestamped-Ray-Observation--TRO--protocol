"""Network impairment model for TRO messages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import NetworkConfig
from .models import NetworkStats
from .tro_message import TROMessage


@dataclass(order=True, slots=True)
class _QueuedMessage:
    arrival_time: float
    insertion_index: int
    message: TROMessage = field(compare=False)


class NetworkModel:
    """Simulates packet loss, delay, jitter, duplicates, and reordering."""

    def __init__(self, config: NetworkConfig, rng: np.random.Generator):
        config.validate()
        self.config = config
        self.rng = rng
        self.queue: list[_QueuedMessage] = []
        self.stats = NetworkStats()
        self._counter = 0

    def transmit(self, message: TROMessage) -> None:
        """Send a message into the impaired network."""
        self.stats.transmitted_count += 1
        if self.rng.random() < self.config.packet_loss:
            self.stats.dropped_count += 1
            return
        self._enqueue(message)
        if self.rng.random() < self.config.duplicate_probability:
            self.stats.duplicated_count += 1
            self._enqueue(message)

    def deliver(self, current_time: float) -> list[TROMessage]:
        """Deliver all messages whose random arrival time has elapsed."""
        ready: list[_QueuedMessage] = []
        pending: list[_QueuedMessage] = []
        for item in self.queue:
            if item.arrival_time <= current_time + 1.0e-12:
                ready.append(item)
            else:
                pending.append(item)
        self.queue = pending
        ready.sort()
        delivered = [item.message for item in ready]
        self.stats.delivered_count += len(delivered)
        return delivered

    def _enqueue(self, message: TROMessage) -> None:
        delay = self._sample_delay()
        arrival_time = message.transmit_time + delay
        self.queue.append(_QueuedMessage(arrival_time, self._counter, message))
        self._counter += 1
        self.stats.queued_count += 1

    def _sample_delay(self) -> float:
        mode = self.config.delay_mode
        if mode == "fixed":
            return float(self.config.fixed_delay_s)
        if mode == "uniform":
            low, high = self.config.uniform_delay_range_s
            return float(self.rng.uniform(low, high))
        if mode == "normal":
            return float(max(0.0, self.rng.normal(self.config.normal_delay_mean_s, self.config.normal_delay_std_s)))
        raise ValueError(f"unknown delay mode: {mode}")
