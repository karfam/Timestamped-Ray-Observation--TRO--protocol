# Timestamped Ray Observation Simulator

This project is a repeatable ray-level simulator for evaluating a Timestamped Ray Observation (TRO) protocol for cooperative multi-UAV vision-based target localization. It does not simulate images. Instead, it simulates UAV positions, target motion, noisy world-frame bearing rays, compact TRO messages, network impairments, and fusion-node localization results.

The simulator is designed to support research-paper experiments around embedded communication constraints: capture timestamps, sequence numbers, sliding-window buffering, confidence and angular-uncertainty weighting, stale packet rejection, residual gating, and compact 48-byte or 64-byte payload sizes.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

## Running Experiments

Run all experiments:

```bash
python -m simulation.main --experiment all
```

Run a selected experiment:

```bash
python -m simulation.main --experiment test1a_packet_loss
python -m simulation.main --experiment test1b_heterogeneous_rates
python -m simulation.main --experiment test2a_delay_sweep
python -m simulation.main --experiment test2b_jitter_sweep
python -m simulation.main --experiment delay
python -m simulation.main --experiment packet_loss
python -m simulation.main --experiment bandwidth
```

Run a short smoke test:

```bash
python -m simulation.main --experiment smoke --runs 1
```

Run internal validation checks:

```bash
python -m simulation.main --experiment validate
```

Useful options:

```bash
python -m simulation.main --experiment all --runs 50 --seed 10
python -m simulation.main --experiment delay --runs 5 --no-plots
python -m simulation.main --experiment all --output-dir outputs_paper
```

## Experiment Set

- `ideal`: no packet loss, no delay, static target, configurable bearing noise.
- `test1a_packet_loss`: Test 1A synchronized bearing fusion baseline, sweeping packet loss from 0% to 50% and comparing strict synchronized fusion with TRO sliding-window fusion.
- `test1b_heterogeneous_rates`: Test 1B heterogeneous UAV update rates `[10, 5, 2, 1]` Hz with 0% to 10% packet loss, comparing strict synchronized fusion with TRO sliding-window fusion.
- `test2a_delay_sweep`: Test 2A arrival-time fusion baseline, sweeping fixed delay from 0 to 1000 ms with 0% to 10% packet loss for a moving target.
- `test2b_jitter_sweep`: Test 2B arrival-time fusion baseline, sweeping delay jitter from 0 to 500 ms around 250 ms mean delay with 5% packet loss.
- `packet_loss`: packet loss sweep from 0 to 50%, comparing TRO sliding-window fusion with a latest-only baseline.
- `delay`: fixed delay sweep comparing capture-time buffering with an arrival-time baseline for a moving target.
- `window`: sliding-window duration sweep for target speeds 0, 2, and 10 m/s.
- `rates`: equal UAV update rates compared with heterogeneous rates `[10, 5, 2, 1]` Hz.
- `outliers`: false-ray injection with residual gating enabled and disabled.
- `bandwidth`: payload bandwidth table for 48-byte and 64-byte messages across UAV counts and rates.

## Output Files

Outputs are written to `outputs/` by default.

Per experiment:

- `<experiment>_per_run_summary.csv`: one row per Monte Carlo run and condition.
- `<experiment>_summary.csv`: condition-level mean summary metrics.
- `<experiment>_time_series.csv`: per-fusion-cycle metrics where applicable.

Consolidated:

- `results_summary.csv`: summary table for the most recently selected experiment, or all experiments when `--experiment all` is used.

Figures:

- Saved as PNG and PDF under `outputs/figures/`.
- Includes RMSE versus packet loss, availability versus packet loss, RMSE versus delay, capture-time versus arrival-time comparisons, RMSE versus sliding-window duration, payload bandwidth versus number of UAVs, and trajectory plots for selected runs.

## Key Metrics

Time-series rows include:

- Ground-truth and estimated target positions.
- Valid estimate flag.
- Localization error.
- Active rays and contributing UAV count.
- Mean and max observation age.
- Mean and max ray residual.
- Number of residual-gated observations.
- Least-squares condition number and geometry quality.
- Packet loss, duplicate, out-of-order, and stale-rejection counters.

Summary rows include:

- RMSE, mean, median, 95th percentile, and max localization error.
- Estimate availability percentage.
- Mean active rays and contributing UAVs.
- Packet and stale rejection statistics.
- Estimated payload bandwidth in bytes/s and kbit/s.

## Package Structure

```text
simulation/
    main.py             CLI entry point
    config.py           Dataclass configuration objects
    scenario.py         UAV and target scenario generation plus noisy ray creation
    models.py           Shared result and statistics dataclasses
    tro_message.py      TROMessage dataclass and payload bandwidth helpers
    network_model.py    Packet loss, delay, jitter, reordering, and duplicate model
    fusion_node.py      Message validation, buffering, weighted multi-ray fusion
    metrics.py          Time-series recording and run summaries
    experiments.py      Monte Carlo experiment definitions
    plots.py            Matplotlib figure generation
    utils.py            Geometry, random, and filesystem helpers
```

## Fusion Method

For ray origin `p_i`, unit direction `d_i`, and weight `w_i`, the fusion node uses the perpendicular projection matrix:

```text
P_i = I - d_i d_i^T
A = sum(w_i P_i)
b = sum(w_i P_i p_i)
x = A^-1 b
```

By default, weights are combined confidence and angular uncertainty weights:

```text
w_i = confidence / max(angular_uncertainty^2, epsilon)
```

The node rejects invalid CRC/version/type messages, tracks sequence gaps, detects duplicates and out-of-order arrivals, rejects stale observations by `capture_time`, and optionally applies residual gating before recomputing the estimate.
