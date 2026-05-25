"""Generate IEEE Transactions-grade per-scenario violation snapshots.

For each scenario, for every rule violated at least once:

1. Locate the most recent simulation log (``msgpack.xz``) under
   ``${NUPLAN_EXP_ROOT}/exp/demo_12_batch__<label>__l1/...``.
2. Load it via ``NuPlanSimulationLogSource`` (no nuPlan re-simulation).
3. Find the *peak*-violation tick from the per-tick CSV log.
4. Render an IEEE-compliant single-frame figure: clean top-down map view
   centred on the ego, with ego footprint + heading line, agent footprints,
   planned trajectory, and traffic-light markers. A colour-coded banner
   above the map identifies the violating rule, its priority level, peak
   rate, integrated violation, and timestamp. A border around the map
   matches the priority colour.
5. Save the snapshot to
   ``examples/outputs/artifacts/violations/<label>/<rule_id>__t<tick>.png``.
6. Composite all per-rule snapshots for the scenario into one gallery
   ``violations/<label>__gallery.png`` (IEEE 2-column page width).

Conforms to IEEE Transactions typography: Times serif, 8 pt body, 300 dpi,
column widths from ``ieee_style`` (3.50 in single / 7.16 in double).

Run from workspace root:

    python scripts/generate_violation_snapshots.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

# ---- IEEE styling -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ieee_style  # noqa: E402

ieee_style.apply()
COL_1 = ieee_style.COL_1   # 3.50 in
COL_2 = ieee_style.COL_2   # 7.16 in

# ---- Workspace + helpers ----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.visualizer import (  # noqa: E402
    AGENT_COLORS, LAYER_STYLE, TL_COLORS,
    _collect_static_features, _draw_static_map, _oriented_box_xy,
)
from lexicone.observer.simulation_log_adapter import NuPlanSimulationLogSource  # noqa: E402

# ---- Paths ------------------------------------------------------------------
BATCH_DIR = ROOT / "examples" / "outputs" / "12_batch_two_level_mpc_planner"
OUT = ROOT / "examples" / "outputs" / "artifacts" / "violations"
OUT.mkdir(parents=True, exist_ok=True)

EXP_ROOT = Path(os.environ.get("NUPLAN_EXP_ROOT", "/workspace/exp")) / "exp"

# ---- Priority levels --------------------------------------------------------
LEVEL_COLOURS = {
    10: "#7f1d1d", 9: "#c0392b", 8: "#e67e22", 7: "#f1c40f",
    3: "#16a085", 2: "#2980b9", 1: "#8e44ad", 0: "#7f8c8d",
}
LEVEL_NAME = {
    10: "Safety (VRU)", 9: "Safety (vehicle/surface)", 8: "Mandatory stop / yield",
    7: "Legal (lane/light)", 3: "Comfort / headway", 2: "Route adherence",
    1: "Priority / lateral", 0: "Comfort (long./lat.)",
}
LEVEL_FROM_RID: Dict[str, int] = {
    "10r0": 10, "10r3": 10, "10r4": 10, "10r5": 10,
    "9r0": 9, "9r1": 9,
    "8r0": 8, "8r1": 8,
    "7r0": 7, "7r1": 7, "7r2": 7, "7r3": 7, "7r4": 7, "7r5": 7,
    "3r0": 3, "3r3": 3, "3r5": 3, "3r6": 3,
    "2r2": 2,
    "1r0": 1, "1r2": 1, "1r5": 1, "1r11": 1,
    "0r2": 0, "0r3": 0,
}

MAP_MARGIN_M = 50.0
FPS = 10


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


@dataclass
class RulePeak:
    rule_id: str
    rule_name: str
    level: int
    peak_tick: int
    peak_rate: float
    integrated: float
    n_violation_ticks: int
    peak_timestamp_s: float


@dataclass
class ScenarioMeta:
    label: str
    msgpack_path: Path
    csv_path: Path
    peaks: List[RulePeak]


def find_latest_msgpack(label: str) -> Optional[Path]:
    """Return the most recent msgpack.xz simulation log for ``label``."""
    label_dir_name = f"demo_12_batch__{label}__l1"
    candidates: List[Path] = []
    # Two layers of the same dir name (Hydra convention).
    parent1 = EXP_ROOT / label_dir_name / label_dir_name
    if parent1.is_dir():
        candidates.extend(parent1.rglob("*.msgpack.xz"))
    # Also try a single-level layout.
    parent2 = EXP_ROOT / label_dir_name
    if parent2.is_dir():
        candidates.extend(p for p in parent2.rglob("*.msgpack.xz") if p not in candidates)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_csv_peaks(csv_path: Path) -> List[RulePeak]:
    per_tick: Dict[float, Dict[str, Tuple[int, int, float]]] = {}
    rule_names: Dict[str, str] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            t = float(row["t_s"])
            rid = row["rule_id"]
            per_tick.setdefault(t, {})[rid] = (
                int(row["applies"]), int(row["is_violated"]), float(row["violation_rate"]),
            )
            rule_names[rid] = row["rule_name"]
    timestamps = sorted(per_tick.keys())
    if not timestamps:
        return []
    dts = np.array([
        (timestamps[i + 1] - timestamps[i]) if i + 1 < len(timestamps) else
        (timestamps[i] - timestamps[i - 1] if i > 0 else 0.1)
        for i in range(len(timestamps))
    ])
    if dts[0] == 0:
        dts[0] = dts[1] if len(dts) > 1 else 0.1

    peaks: List[RulePeak] = []
    for rid, name in rule_names.items():
        rates = np.array([per_tick[t].get(rid, (0, 0, 0.0))[2] for t in timestamps])
        applies = np.array([per_tick[t].get(rid, (0, 0, 0.0))[0] for t in timestamps])
        violated = np.array([per_tick[t].get(rid, (0, 0, 0.0))[1] for t in timestamps], dtype=bool)
        if not violated.any():
            continue
        peak_tick = int(np.argmax(np.where(violated, rates, -1)))
        peaks.append(RulePeak(
            rule_id=rid, rule_name=name,
            level=LEVEL_FROM_RID.get(rid, 0),
            peak_tick=peak_tick,
            peak_rate=float(rates[peak_tick]),
            integrated=float((rates * applies * dts).sum()),
            n_violation_ticks=int(violated.sum()),
            peak_timestamp_s=peak_tick / FPS,
        ))
    peaks.sort(key=lambda p: (-p.level, -p.peak_rate))
    return peaks


# ----------------------------------------------------------------------
# IEEE-compliant single-frame renderer
# ----------------------------------------------------------------------


def render_snapshot(
    snapshot, static_features: dict, lane_connectors,
    peak: RulePeak, label: str, out_path: Path,
) -> None:
    """Render one IEEE-compliant snapshot for the given peak-violation tick."""
    colour = LEVEL_COLOURS.get(peak.level, "#666")

    # IEEE 2-column figure: 7.16 in wide. Banner sized for 10pt text; map ~4.0 in tall.
    fig_w_in = COL_2
    banner_in = 0.42
    map_in = 4.1
    fig_h_in = banner_in + map_in + 0.04
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[banner_in, map_in],
        left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.025,
    )
    ax_banner = fig.add_subplot(gs[0]); ax_banner.set_axis_off()
    ax = fig.add_subplot(gs[1])

    # Banner
    ax_banner.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.004,rounding_size=0.02",
        transform=ax_banner.transAxes, facecolor=colour, edgecolor=colour,
        alpha=0.95, lw=0,
    ))
    # Left side: priority level + rule id + truncated rule name (max 38 chars).
    name_trunc = peak.rule_name if len(peak.rule_name) <= 36 else peak.rule_name[:34] + "…"
    ax_banner.text(
        0.010, 0.55,
        f"L{peak.level}   {peak.rule_id}   ·   {name_trunc}",
        transform=ax_banner.transAxes, va="center", ha="left",
        fontsize=10, fontweight="bold", color="white",
    )
    # Right side: compact one-line metric summary.
    ax_banner.text(
        0.990, 0.55,
        f"peak={peak.peak_rate:.3f}   $\\int{{=}}${peak.integrated:.2f}   "
        f"$t{{=}}${peak.peak_timestamp_s:.1f}$\\,$s   ({peak.n_violation_ticks} ticks)",
        transform=ax_banner.transAxes, va="center", ha="right",
        fontsize=8.5, color="white",
    )

    # Static map layers (drivable area, lanes, crosswalks, walkways, intersections,
    # stop lines, lane direction arrows, centerlines).
    _draw_static_map(ax, static_features)

    # Dynamic layers — ego, agents, planned trajectory, traffic lights.
    ego = snapshot.ego
    ego_box = _oriented_box_xy(ego.pose.x, ego.pose.y, ego.length, ego.width, ego.pose.heading)
    ax.add_collection(PolyCollection(
        [ego_box], facecolors=["#d62728"], edgecolors="#7f1d1d",
        linewidths=0.8, alpha=0.95, zorder=10,
    ))
    cos_h, sin_h = math.cos(ego.pose.heading), math.sin(ego.pose.heading)
    ax.add_collection(LineCollection(
        [[(ego.pose.x, ego.pose.y), (ego.pose.x + 4.0 * cos_h, ego.pose.y + 4.0 * sin_h)]],
        colors="#7f1d1d", linewidths=1.2, zorder=11,
    ))

    # Agents
    boxes, colours = [], []
    for a in snapshot.agents:
        ot = a.object_type.value if hasattr(a.object_type, "value") else str(a.object_type)
        boxes.append(_oriented_box_xy(a.pose.x, a.pose.y, a.length, a.width, a.pose.heading))
        colours.append(AGENT_COLORS.get(ot, "#999"))
    if boxes:
        ax.add_collection(PolyCollection(
            boxes, facecolors=colours, edgecolors="#222",
            linewidths=0.3, alpha=0.85, zorder=9,
        ))

    # Planned trajectory
    if snapshot.planned_trajectory:
        ax.plot(
            [p.pose.x for p in snapshot.planned_trajectory],
            [p.pose.y for p in snapshot.planned_trajectory],
            color="#1f3a5f", linewidth=0.8, linestyle=(0, (3, 1.2)),
            alpha=0.85, zorder=8,
        )

    # Traffic lights
    lc_by_id = {lc.lane_id: lc for lc in lane_connectors}
    tl_x, tl_y, tl_c = [], [], []
    for tl in snapshot.traffic_lights:
        lc = lc_by_id.get(tl.lane_connector_id)
        if lc is None or not lc.centerline:
            continue
        cx = sum(p[0] for p in lc.centerline) / len(lc.centerline)
        cy = sum(p[1] for p in lc.centerline) / len(lc.centerline)
        tl_x.append(cx); tl_y.append(cy)
        state_str = tl.state.value if hasattr(tl.state, "value") else str(tl.state)
        tl_c.append(TL_COLORS.get(state_str, "#888"))
    if tl_x:
        ax.scatter(tl_x, tl_y, s=40, c=tl_c, edgecolors="black", linewidths=0.5, zorder=6)

    # Viewport
    ax.set_xlim(ego.pose.x - MAP_MARGIN_M, ego.pose.x + MAP_MARGIN_M)
    ax.set_ylim(ego.pose.y - MAP_MARGIN_M, ego.pose.y + MAP_MARGIN_M)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Coloured border matching the priority level
    border = mpatches.Rectangle(
        (0.001, 0.001), 0.998, 0.998, transform=ax.transAxes,
        edgecolor=colour, facecolor="none", lw=1.2,
    )
    ax.add_patch(border)

    # Scale bar (10 m, bottom-left of map)
    bar_len_m = 10.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bar_x = x0 + 0.04 * (x1 - x0)
    bar_y = y0 + 0.04 * (y1 - y0)
    ax.plot([bar_x, bar_x + bar_len_m], [bar_y, bar_y],
            color="#1f1f1f", lw=1.0, solid_capstyle="butt", zorder=12)
    ax.text(bar_x + bar_len_m / 2, bar_y + 0.015 * (y1 - y0),
            f"{int(bar_len_m)} m", ha="center", va="bottom",
            fontsize=8, color="#1f1f1f", zorder=12)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------------
# Gallery composite
# ----------------------------------------------------------------------


def render_gallery(label: str, peaks: List[RulePeak],
                   individual_paths: List[Path], out_path: Path) -> None:
    """Composite per-rule snapshots into one IEEE-compliant gallery PNG."""
    n = len(individual_paths)
    if n == 0:
        return
    # Choose grid: 3 cols for 7-9 violations, 2 cols for 4-6, 4 cols for >=10
    if n >= 10:
        cols = 4
    elif n >= 7:
        cols = 3
    elif n >= 4:
        cols = 2
    else:
        cols = n
    rows = (n + cols - 1) // cols

    fig_w_in = COL_2
    tile_w_in = (fig_w_in - 0.04 * (cols - 1) - 0.02 * 2) / cols
    img0 = mpimg.imread(individual_paths[0])
    tile_h_in = tile_w_in * img0.shape[0] / img0.shape[1]
    header_in = 0.38
    fig_h_in = header_in + rows * tile_h_in + 0.04 * (rows - 1) + 0.04

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")
    gs = fig.add_gridspec(
        rows + 1, cols,
        height_ratios=[header_in] + [tile_h_in] * rows,
        left=0.005, right=0.995, top=0.99, bottom=0.005,
        hspace=0.04 / max(tile_h_in, 1.0),
        wspace=0.04 / max(tile_w_in, 1.0),
    )
    ax_h = fig.add_subplot(gs[0, :]); ax_h.set_axis_off()
    ax_h.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.003,rounding_size=0.02",
        transform=ax_h.transAxes, facecolor="#f5f3ee", edgecolor="#b3aea4", lw=0.5,
    ))
    ax_h.text(
        0.010, 0.55,
        f"{label}  ·  {n} violating rules",
        transform=ax_h.transAxes, va="center", ha="left",
        fontsize=11, fontweight="bold", color="#1f1f1f",
    )
    levels_present = sorted({p.level for p in peaks}, reverse=True)
    sw_x = 0.990
    for lv in levels_present:
        ax_h.add_patch(mpatches.FancyBboxPatch(
            (sw_x - 0.030, 0.22), 0.030, 0.56,
            boxstyle="round,pad=0.001,rounding_size=0.01",
            transform=ax_h.transAxes, facecolor=LEVEL_COLOURS[lv],
            edgecolor="#fff", lw=0.5,
        ))
        ax_h.text(sw_x - 0.015, 0.5, f"L{lv}", transform=ax_h.transAxes,
                  ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        sw_x -= 0.034

    for i, ind_path in enumerate(individual_paths):
        r = 1 + i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c]); ax.set_axis_off()
        try:
            ax.imshow(mpimg.imread(ind_path))
        except Exception:
            continue

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


SCENARIO_LABELS = [
    "01_following_slow_lead", "02_near_long_vehicle", "03_near_multiple_vehicles",
    "04_changing_lane", "05_changing_lane_left", "06_changing_lane_right",
    "07_starting_left_turn", "08_starting_right_turn", "09_high_speed_turn",
    "10_low_speed_turn", "11_protected_cross", "12_unprotected_cross",
    "13_high_magnitude_speed", "14_medium_magnitude_speed",
    "15_near_high_speed_vehicle", "16_traversing_intersection",
]


def process_scenario(label: str) -> int:
    sd = BATCH_DIR / f"{label}__l1"
    csv_path = next(sd.glob("*_log.csv"), None)
    if csv_path is None:
        print(f"  {label}: no per-tick CSV")
        return 0
    peaks = load_csv_peaks(csv_path)
    if not peaks:
        print(f"  {label}: no violations")
        return 0

    msgpack = find_latest_msgpack(label)
    if msgpack is None:
        print(f"  {label}: no msgpack on disk under {EXP_ROOT}")
        return 0

    try:
        source = NuPlanSimulationLogSource.from_path(
            msgpack, radius_m=80.0, route_lane_ids=None, include_lane_connectors=True,
        )
        snapshots = list(source)
    except Exception as exc:
        print(f"  {label}: msgpack load failed ({exc})")
        return 0

    if not snapshots:
        print(f"  {label}: msgpack empty")
        return 0

    static_features = _collect_static_features(snapshots)
    lane_connectors = list(static_features.get("lane_connector", {}).values())

    scen_dir = OUT / label
    scen_dir.mkdir(parents=True, exist_ok=True)
    individual_paths: List[Path] = []

    for peak in peaks:
        if peak.peak_tick >= len(snapshots):
            continue
        snap = snapshots[peak.peak_tick]
        ind_path = scen_dir / f"{peak.rule_id}__t{peak.peak_tick:03d}.png"
        try:
            render_snapshot(snap, static_features, lane_connectors,
                            peak, label, ind_path)
            individual_paths.append(ind_path)
        except Exception as exc:
            print(f"    skip {peak.rule_id} (render failed: {exc})")

    gallery_path = OUT / f"{label}__gallery.png"
    render_gallery(label, peaks[:len(individual_paths)], individual_paths, gallery_path)

    print(f"  {label}: {len(individual_paths)} snapshots → gallery")
    return len(individual_paths)


def write_index(per_scenario_counts: Dict[str, int]) -> None:
    total = sum(per_scenario_counts.values())
    lines = [
        "# Violation snapshots — IEEE Transactions grade",
        "",
        "Generated by `scripts/generate_violation_snapshots.py` directly from",
        "the simulation msgpack logs (no MP4 re-extraction). Each frame is",
        "rendered at IEEE Transactions specifications:",
        "",
        "- Figure width: 7.16 in (2-column page width)",
        "- Resolution: 300 dpi",
        "- Body font: Times (serif), 8 pt",
        "- Banner font: 9 pt bold for rule, 7 pt for metrics",
        "- Line widths: 0.5–1.2 pt",
        "",
        "## Files per scenario",
        "",
        "- `<label>__gallery.png` — composite of all violating-rule snapshots, IEEE 2-column",
        "- `<label>/<rule_id>__t<tick>.png` — individual annotated snapshot",
        "",
        "## Colour code (priority level → colour)",
        "",
        "| Level | Group | Colour |",
        "|---|---|---|",
    ]
    for lv in sorted(LEVEL_NAME, reverse=True):
        lines.append(f"| L{lv} | {LEVEL_NAME[lv]} | `{LEVEL_COLOURS[lv]}` |")
    lines += [
        "",
        "## Per-scenario counts",
        "",
        "| Scenario | # violating rules | Gallery |",
        "|---|---|---|",
    ]
    for label in SCENARIO_LABELS:
        cnt = per_scenario_counts.get(label, 0)
        lines.append(f"| {label} | {cnt} | [`{label}__gallery.png`]({label}__gallery.png) |")
    lines += [
        "",
        f"**Total snapshots: {total}**  ·  **Galleries: {sum(1 for v in per_scenario_counts.values() if v > 0)}**",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    counts: Dict[str, int] = {}
    for label in SCENARIO_LABELS:
        counts[label] = process_scenario(label)
    write_index(counts)
    total = sum(counts.values())
    print(f"\nGenerated {total} individual snapshots + "
          f"{sum(1 for v in counts.values() if v > 0)} galleries")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
