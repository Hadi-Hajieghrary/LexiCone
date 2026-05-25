"""Extract trajectory-smoothness metrics from simulation logs.

For each ``msgpack.xz`` produced by the comparative-effectiveness protocol,
replay it via :class:`NuPlanSimulationLogSource` and compute per-episode
smoothness aggregates from the ego-pose stream:

* ``lat_jerk_rms``  — RMS lateral jerk over the episode
* ``lon_jerk_rms``  — RMS longitudinal jerk
* ``peak_a_y``      — peak absolute lateral acceleration
* ``peak_a_x``      — peak absolute longitudinal acceleration
* ``mean_speed``    — mean ego speed
* ``distance_m``    — total distance travelled along the planned route

Writes a long-form CSV keyed by ``(condition, scenario, seed)``. Consumed by
F7 in the analysis pipeline.

Replay is fast — about 5 s per scenario; no simulator re-run.

Usage::

    python examples/metrics_smoothness.py \\
        --protocol-root examples/outputs/13_protocol \\
        --out examples/outputs/13_protocol/figures/smoothness.csv

    # legacy 12_batch layout
    python examples/metrics_smoothness.py \\
        --legacy-glob 'examples/outputs/12_batch*/*/*.msgpack.xz' \\
        --out /tmp/smoothness.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lexicone.observer.simulation_log_adapter import NuPlanSimulationLogSource  # noqa: E402


EXP_ROOT = Path(os.environ.get("NUPLAN_EXP_ROOT", "/workspace/exp")) / "exp"


def smoothness_from_log(log_path: Path) -> dict:
    """Replay one simulation log; compute ego-trajectory smoothness aggregates."""
    source = NuPlanSimulationLogSource.from_path(log_path, radius_m=20.0)
    snapshots = list(source)
    if len(snapshots) < 3:
        return {"n_ticks": len(snapshots)}

    # Time-stamped ego state arrays.
    t = np.array([s.timestamp_us * 1e-6 for s in snapshots])
    px = np.array([s.ego.pose.x for s in snapshots])
    py = np.array([s.ego.pose.y for s in snapshots])
    psi = np.array([s.ego.pose.heading for s in snapshots])
    v = np.array([s.ego.speed for s in snapshots])
    # ax/ay come from the snapshot's recorded accelerations when present.
    ax = np.array([getattr(s.ego, "ax", 0.0) for s in snapshots])
    ay = np.array([getattr(s.ego, "ay", 0.0) for s in snapshots])

    dt = np.diff(t)
    dt = np.maximum(dt, 1e-3)   # safety floor

    # Body-frame jerk via finite differences. If ax/ay are zero (some sims
    # don't record them), fall back to second derivative of position.
    if np.allclose(ax, 0.0) and np.allclose(ay, 0.0):
        # second derivatives in world frame, then rotate to body frame.
        vx_w = np.diff(px) / dt
        vy_w = np.diff(py) / dt
        ax_w = np.concatenate([[0.0], np.diff(vx_w) / dt[:-1]])
        ay_w = np.concatenate([[0.0], np.diff(vy_w) / dt[:-1]])
        # Rotate world accel into body frame using psi at midpoint.
        cos_p = np.cos(psi[1:])
        sin_p = np.sin(psi[1:])
        ax_b = cos_p * ax_w + sin_p * ay_w
        ay_b = -sin_p * ax_w + cos_p * ay_w
        # Pad to length N.
        ax = np.concatenate([[0.0], ax_b])
        ay = np.concatenate([[0.0], ay_b])

    # Jerk (body frame).
    jx = np.diff(ax) / dt
    jy = np.diff(ay) / dt
    lon_jerk_rms = float(np.sqrt(np.mean(jx ** 2)))
    lat_jerk_rms = float(np.sqrt(np.mean(jy ** 2)))
    peak_a_x = float(np.max(np.abs(ax)))
    peak_a_y = float(np.max(np.abs(ay)))
    mean_speed = float(np.mean(v))
    # Total distance.
    dist = float(np.sum(np.hypot(np.diff(px), np.diff(py))))

    return {
        "n_ticks": len(snapshots),
        "duration_s": float(t[-1] - t[0]),
        "mean_speed_mps": mean_speed,
        "distance_m": dist,
        "lon_jerk_rms": lon_jerk_rms,
        "lat_jerk_rms": lat_jerk_rms,
        "peak_a_x": peak_a_x,
        "peak_a_y": peak_a_y,
    }


def discover_logs_protocol(protocol_root: Path) -> Iterable[Tuple[str, str, int, Path]]:
    """Walk the 13_protocol/<condition>/seed_<n>/<label>__suffix/ layout.

    Yields ``(condition, scenario, seed, msgpack_path)`` tuples. The msgpack
    is found by globbing under EXP_ROOT/exp/demo_12_batch__<label>__<suffix>/
    keyed off the per-scenario subdir name.
    """
    for cond_dir in sorted(protocol_root.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name == "figures":
            continue
        condition = cond_dir.name
        for seed_dir in sorted(cond_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.removeprefix("seed_"))
            for label_dir in sorted(seed_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                # Find latest matching msgpack under EXP_ROOT.
                exp_glob = (EXP_ROOT
                            / f"demo_12_batch__{label_dir.name}"
                            / f"demo_12_batch__{label_dir.name}")
                candidates = list(exp_glob.rglob("*.msgpack.xz")) if exp_glob.is_dir() else []
                if not candidates:
                    continue
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                scenario = label_dir.name.split("__")[0]
                yield condition, scenario, seed, latest


def discover_logs_legacy(legacy_glob: str) -> Iterable[Tuple[str, str, int, Path]]:
    """Walk a glob of legacy ``.msgpack.xz`` paths under EXP_ROOT directly."""
    for log_path in sorted(Path("/").glob(legacy_glob.lstrip("/"))):
        # Heuristic: condition + scenario from path components.
        parts = log_path.parts
        condition = "unknown"
        scenario = "unknown"
        for p in parts:
            if p.startswith("demo_12_batch__"):
                stripped = p.removeprefix("demo_12_batch__")
                # Strip trailing __l1 / __l1_cascade / __l2 suffix
                if "__l1_cascade" in stripped:
                    condition = "C4_cascade_l1"
                    scenario = stripped.split("__l1_cascade")[0]
                elif "__l1" in stripped:
                    condition = "C1_ws_l1"
                    scenario = stripped.split("__l1")[0]
                elif "__l2" in stripped:
                    condition = "C3_ws_l2"
                    scenario = stripped.split("__l2")[0]
                else:
                    condition = "C0_legacy"
                    scenario = stripped
                break
        yield condition, scenario, 0, log_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--protocol-root", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "13_protocol",
    )
    p.add_argument(
        "--legacy-glob", action="append", default=[],
        help="Additional glob patterns to harvest legacy msgpacks from. "
             "May be repeated.",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output CSV path.",
    )
    args = p.parse_args()

    rows: List[dict] = []
    sources: List[Tuple[str, str, int, Path]] = []
    if args.protocol_root.is_dir():
        sources.extend(discover_logs_protocol(args.protocol_root))
    for pattern in args.legacy_glob:
        sources.extend(discover_logs_legacy(pattern))

    if not sources:
        print(f"no simulation logs found", file=sys.stderr)
        return 1
    print(f"[smoothness] {len(sources)} logs to process")

    for condition, scenario, seed, log_path in sources:
        try:
            m = smoothness_from_log(log_path)
        except Exception as exc:
            print(f"  skip {condition}/{scenario}/{seed}: {exc}")
            continue
        m.update({"condition": condition, "scenario": scenario, "seed": seed,
                  "log_path": str(log_path)})
        rows.append(m)

    if not rows:
        print("no successful extractions", file=sys.stderr)
        return 1
    fields = ["condition", "scenario", "seed", "n_ticks", "duration_s",
              "mean_speed_mps", "distance_m",
              "lon_jerk_rms", "lat_jerk_rms",
              "peak_a_x", "peak_a_y", "log_path"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[smoothness] wrote {len(rows)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
