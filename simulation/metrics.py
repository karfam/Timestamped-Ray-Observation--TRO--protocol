"""Metric recording and summary calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import EstimateResult, FusionStats, NetworkStats


@dataclass(slots=True)
class MetricsRecorder:
    """Collects per-fusion-cycle metrics and run summaries."""

    rows: list[dict[str, float | int | bool]] = field(default_factory=list)

    def record(
        self,
        result: EstimateResult,
        truth_position: np.ndarray,
        fusion_stats: FusionStats,
        network_stats: NetworkStats,
    ) -> None:
        """Store one fusion-cycle metric row."""
        if result.valid and result.estimate is not None:
            error = float(np.linalg.norm(result.estimate - truth_position))
            estimate_x, estimate_y, estimate_z = result.estimate.tolist()
        else:
            error = float("nan")
            estimate_x = estimate_y = estimate_z = float("nan")
        self.rows.append(
            {
                "time_s": result.current_time,
                "truth_x_m": float(truth_position[0]),
                "truth_y_m": float(truth_position[1]),
                "truth_z_m": float(truth_position[2]),
                "estimate_x_m": estimate_x,
                "estimate_y_m": estimate_y,
                "estimate_z_m": estimate_z,
                "valid_estimate": result.valid,
                "localization_error_m": error,
                "active_rays": result.active_rays,
                "contributing_uavs": result.contributing_uavs,
                "mean_observation_age_s": result.mean_observation_age,
                "max_observation_age_s": result.max_observation_age,
                "mean_residual_m": result.mean_residual,
                "max_residual_m": result.max_residual,
                "gated_rejected_observations": result.gated_observations,
                "condition_number": result.condition_number,
                "geometry_quality": result.geometry_quality,
                "packet_loss_count": fusion_stats.estimated_packet_loss_count + network_stats.dropped_count,
                "duplicate_count": fusion_stats.duplicate_count,
                "out_of_order_count": fusion_stats.out_of_order_count,
                "stale_rejected_count": fusion_stats.stale_rejected_count,
            }
        )

    def dataframe(self) -> pd.DataFrame:
        """Return time-series metrics as a DataFrame."""
        return pd.DataFrame(self.rows)


def summarize_time_series(df: pd.DataFrame) -> dict[str, float]:
    """Compute summary statistics from one run's time-series DataFrame."""
    if df.empty:
        return {
            "rmse_m": float("nan"),
            "mean_error_m": float("nan"),
            "median_error_m": float("nan"),
            "p95_error_m": float("nan"),
            "max_error_m": float("nan"),
            "estimate_availability_pct": 0.0,
            "mean_active_rays": 0.0,
            "mean_contributing_uavs": 0.0,
        }
    valid = df[df["valid_estimate"] == True]
    errors = valid["localization_error_m"].dropna().to_numpy(dtype=float)
    if len(errors) == 0:
        rmse = mean_error = median_error = p95_error = max_error = float("nan")
    else:
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_error = float(np.mean(errors))
        median_error = float(np.median(errors))
        p95_error = float(np.percentile(errors, 95))
        max_error = float(np.max(errors))
    return {
        "rmse_m": rmse,
        "mean_error_m": mean_error,
        "median_error_m": median_error,
        "p95_error_m": p95_error,
        "max_error_m": max_error,
        "estimate_availability_pct": float(100.0 * len(valid) / len(df)),
        "valid_estimate_count": int(len(valid)),
        "invalid_fusion_cycles": int(len(df) - len(valid)),
        "mean_active_rays": float(df["active_rays"].mean()),
        "mean_contributing_uavs": float(df["contributing_uavs"].mean()),
        "mean_observation_age_s": float(df["mean_observation_age_s"].mean(skipna=True)),
        "max_observation_age_s": float(df["max_observation_age_s"].max(skipna=True)),
        "mean_residual_m": float(df["mean_residual_m"].mean(skipna=True)),
        "max_residual_m": float(df["max_residual_m"].max(skipna=True)),
        "mean_condition_number": float(df["condition_number"].replace([np.inf, -np.inf], np.nan).mean(skipna=True)),
        "final_packet_loss_count": int(df["packet_loss_count"].iloc[-1]),
        "final_duplicate_count": int(df["duplicate_count"].iloc[-1]),
        "final_out_of_order_count": int(df["out_of_order_count"].iloc[-1]),
        "final_stale_rejected_count": int(df["stale_rejected_count"].iloc[-1]),
    }


def aggregate_runs(summary_rows: list[dict[str, object]], group_columns: list[str]) -> pd.DataFrame:
    """Average Monte Carlo run summaries by condition."""
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return frame
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    excluded_columns = {"run", "seed", *group_columns}
    run_independent_columns = [col for col in numeric_columns if col not in excluded_columns]
    grouped = frame.groupby(group_columns, dropna=False)[run_independent_columns].mean(numeric_only=True).reset_index()
    return grouped
