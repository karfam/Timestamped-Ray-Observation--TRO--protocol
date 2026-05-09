"""Utility functions for ray simulation and filesystem output."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_output_dirs(output_dir: Path) -> None:
    """Create output directories used by experiments."""
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)


def make_rng(seed: int | None) -> np.random.Generator:
    """Return a NumPy random generator."""
    return np.random.default_rng(seed)


def normalize(vector: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Return a unit vector, raising if the norm is too small."""
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("cannot normalize near-zero vector")
    return vector / norm


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Sample a random 3-D unit vector."""
    return normalize(rng.normal(size=3))


def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector around axis with Rodrigues' formula."""
    axis = normalize(axis)
    return (
        vector * np.cos(angle_rad)
        + np.cross(axis, vector) * np.sin(angle_rad)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle_rad))
    )


def add_angular_noise(direction: np.ndarray, std_rad: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb a unit direction by a small random angular error."""
    direction = normalize(direction)
    if std_rad <= 0:
        return direction.copy()
    random_axis = random_unit_vector(rng)
    axis = np.cross(direction, random_axis)
    if np.linalg.norm(axis) < 1.0e-10:
        axis = np.cross(direction, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(axis) < 1.0e-10:
        axis = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    angle = float(rng.normal(0.0, std_rad))
    return normalize(rotate_vector(direction, axis, angle))


def geometry_quality(directions: list[np.ndarray]) -> float:
    """Return a simple ray geometry score based on angular diversity."""
    if len(directions) < 2:
        return 0.0
    min_sine = 1.0
    for i, first in enumerate(directions):
        for second in directions[i + 1 :]:
            sine = float(np.linalg.norm(np.cross(normalize(first), normalize(second))))
            min_sine = min(min_sine, sine)
    return min_sine


def kbit_per_second(bytes_per_second: float) -> float:
    """Convert bytes/s to kbit/s."""
    return bytes_per_second * 8.0 / 1000.0
