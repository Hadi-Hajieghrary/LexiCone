"""Per-tick visualisation for rule-engine demos.

Three artefacts are produced per demo:

1. ``<name>.mp4`` — animated playback. A single frame is laid out as

   .. code::

      ┌──── HEADER  scenario · tick · t · ego v/ax/ay · status ──────┐
      ├──────────────────────────────────────────┬──────────────────┤
      │                                          │  CONTEXT card    │
      │              MAP  (no overlays)          │  lane · v_lim    │
      │                                          │  TL ahead        │
      │                                          │  lead distance   │
      │                                          ├──────────────────┤
      │                                          │  ACTIVE RULES    │
      │                                          │  (sorted, ✗/·)   │
      ├──────────────────────────────────────────┴──────────────────┤
      │  Violation strip (per-rule × tick)              [colorbar]  │
      ├──────────────────────────────────────────────────────────────┤
      │  Map legend (one row)                                       │
      └──────────────────────────────────────────────────────────────┘

   Lane labels appear only on the ego's current lane and its immediate
   neighbours (within ``LANE_LABEL_RADIUS_M`` of the ego), showing only
   the speed limit in mph, in small grey text without a bounding box.
   This keeps the map clean during dense intersection scenarios.

2. ``<name>_summary.png`` — static episode summary: a sorted bar chart of
   integrated violations, a per-rule heatmap timeline, and an
   applicability/violation bar chart.

3. ``<name>_log.csv`` — one row per (tick, rule) for downstream analysis.

The visualiser is fully self-contained — it takes a ``RuleEngine`` and the
corresponding snapshots and writes everything to ``output_dir`` via
:func:`render_episode`.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap

from lexicone.observer import RuleEngine, RuleEvaluation, SceneSnapshot
from lexicone.observer.types import AgentType, TrafficLightState


# ----------------------------------------------------------------------
# Typography & style — sized for full-screen MP4 playback, kept legible
# at 50 % when embedded into a paper figure.
# ----------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
    }
)

# Lane-label heuristic radius: only label lanes whose centerline comes
# within this many metres of the ego at the current tick.
LANE_LABEL_RADIUS_M = 14.0


# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------
AGENT_COLORS = {
    AgentType.VEHICLE.value: "#2ca02c",
    AgentType.PEDESTRIAN.value: "#ff7f0e",
    AgentType.BICYCLE.value: "#9467bd",
    AgentType.MOTORCYCLE.value: "#1f77b4",
    AgentType.BARRIER.value: "#8c564b",
    AgentType.TRAFFIC_CONE.value: "#e377c2",
    AgentType.GENERIC_OBJECT.value: "#7f7f7f",
}

LAYER_STYLE = {
    "drivable": {"fc": "#f0eee8", "ec": "none", "alpha": 0.55, "z": -1},
    "road_lane": {"fc": "#e6e2d8", "ec": "#b3aea4", "alpha": 0.85, "z": 1},
    "lane_connector": {"fc": "#d7d2c5", "ec": "#a59b87", "alpha": 0.85, "z": 1},
    "bike_lane": {"fc": "#cfe2cf", "ec": "#76a276", "alpha": 0.85, "z": 1},
    "intersection": {"fc": "#ddd8cf", "ec": "none", "alpha": 0.65, "z": 1},
    "crosswalk": {"fc": "#f6e6a9", "ec": "#c4a93f", "alpha": 0.85, "z": 2},
    "walkway": {"fc": "#dcefd8", "ec": "#9bbf95", "alpha": 0.55, "z": 2},
    "stop_line": {"color": "#c0392b", "lw": 2.4, "z": 3},
    "centerline": {"color": "#4b6fa5", "lw": 0.9, "alpha": 0.55, "z": 4},
}

TL_COLORS = {
    TrafficLightState.RED.value: "#d62728",
    TrafficLightState.YELLOW.value: "#f1c40f",
    TrafficLightState.GREEN.value: "#2ca02c",
    TrafficLightState.UNKNOWN.value: "#7f7f7f",
}

VIOLATION_CMAP = LinearSegmentedColormap.from_list(
    "violation", [(1.0, 1.0, 1.0), (1.0, 0.92, 0.62), (1.0, 0.36, 0.36), (0.55, 0.0, 0.0)]
)

EGO_FC = "#d62728"
EGO_EC = "#7f1d1d"
TRAIL_C = "#d62728"
PLANNED_C = "#1f3a5f"


# ----------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------


@dataclass
class TickEvaluation:
    timestamp_s: float
    by_rule: Dict[str, RuleEvaluation]


def run_episode(engine: RuleEngine, snapshots: Sequence[SceneSnapshot]) -> List[TickEvaluation]:
    engine.run_replay(snapshots)
    ticks: List[TickEvaluation] = []
    for snap, evals in zip(snapshots, engine.history):
        ticks.append(
            TickEvaluation(
                timestamp_s=snap.timestamp_us * 1e-6,
                by_rule={e.rule_id: e for e in evals},
            )
        )
    return ticks


def violation_matrix(ticks: Sequence[TickEvaluation], rule_ids: Sequence[str]) -> np.ndarray:
    mat = np.zeros((len(rule_ids), len(ticks)), dtype=float)
    for j, t in enumerate(ticks):
        for i, rid in enumerate(rule_ids):
            ev = t.by_rule.get(rid)
            if ev is not None and ev.applies:
                mat[i, j] = ev.violation_rate
    return mat


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def _polys(points_list) -> List[List[Tuple[float, float]]]:
    return [list(p) for p in points_list if p is not None and len(p) >= 3]


def _polyline_length(polyline) -> float:
    total = 0.0
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _point_along_polyline(polyline, frac: float) -> Tuple[float, float, float]:
    frac = max(0.0, min(1.0, frac))
    seg_lens = [
        math.hypot(polyline[i + 1][0] - polyline[i][0], polyline[i + 1][1] - polyline[i][1])
        for i in range(len(polyline) - 1)
    ]
    total = sum(seg_lens)
    if total <= 1e-9:
        a = polyline[0]
        return float(a[0]), float(a[1]), 0.0
    target = total * frac
    walked = 0.0
    for i, d in enumerate(seg_lens):
        if walked + d >= target or i == len(seg_lens) - 1:
            t = (target - walked) / d if d > 1e-9 else 0.0
            a = polyline[i]
            b = polyline[i + 1]
            return (
                float(a[0] + (b[0] - a[0]) * t),
                float(a[1] + (b[1] - a[1]) * t),
                float(math.atan2(b[1] - a[1], b[0] - a[0])),
            )
        walked += d
    a, b = polyline[-2], polyline[-1]
    return float(b[0]), float(b[1]), float(math.atan2(b[1] - a[1], b[0] - a[0]))


def _min_distance_point_to_polyline(px: float, py: float, polyline) -> float:
    best = math.inf
    for i in range(len(polyline) - 1):
        ax_, ay_ = polyline[i]
        bx_, by_ = polyline[i + 1]
        dx, dy = bx_ - ax_, by_ - ay_
        d2 = dx * dx + dy * dy
        if d2 < 1e-9:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax_) * dx + (py - ay_) * dy) / d2))
        qx = ax_ + dx * t
        qy = ay_ + dy * t
        d = math.hypot(px - qx, py - qy)
        if d < best:
            best = d
    return best


def _oriented_box_xy(cx: float, cy: float, length: float, width: float, heading: float):
    L2, W2 = length / 2.0, width / 2.0
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return [
        (cx + dx * cos_h - dy * sin_h, cy + dx * sin_h + dy * cos_h)
        for dx, dy in [(-L2, -W2), (L2, -W2), (L2, W2), (-L2, W2)]
    ]


# ----------------------------------------------------------------------
# Static map drawing
# ----------------------------------------------------------------------


def _collect_static_features(snapshots: Sequence[SceneSnapshot]) -> dict:
    seen: dict = {}

    def collect(kind: str, items, key):
        bucket = seen.setdefault(kind, {})
        for it in items:
            bucket.setdefault(key(it), it)

    for snap in snapshots:
        m = snap.map
        collect("drivable", m.drivable_area, key=lambda da: tuple(da.polygon))
        collect("road_lane", [ln for ln in m.lanes if not ln.is_bike_lane], key=lambda ln: ln.lane_id)
        collect("lane_connector", m.lane_connectors, key=lambda ln: ln.lane_id)
        collect("bike_lane", m.bike_lanes, key=lambda ln: ln.lane_id)
        collect("intersection", m.intersections, key=lambda it: it.intersection_id)
        collect("walkway", m.walkways, key=lambda w: w.walkway_id)
        collect("crosswalk", m.crosswalks, key=lambda cw: cw.crosswalk_id)
        collect("stop_line", m.stop_lines, key=lambda sl: sl.stop_line_id)
    return seen


def _draw_static_map(ax, seen: dict) -> List:
    def add_polys(layer_key: str, items):
        s = LAYER_STYLE[layer_key]
        polys = _polys([getattr(it, "polygon", None) for it in items])
        if polys:
            ax.add_collection(
                PolyCollection(polys, facecolor=s["fc"], edgecolor=s["ec"], alpha=s["alpha"], zorder=s["z"])
            )

    add_polys("drivable", list(seen.get("drivable", {}).values()))
    add_polys("road_lane", list(seen.get("road_lane", {}).values()))
    add_polys("lane_connector", list(seen.get("lane_connector", {}).values()))
    add_polys("bike_lane", list(seen.get("bike_lane", {}).values()))
    add_polys("intersection", list(seen.get("intersection", {}).values()))
    add_polys("walkway", list(seen.get("walkway", {}).values()))
    add_polys("crosswalk", list(seen.get("crosswalk", {}).values()))

    all_lanes = list(seen.get("road_lane", {}).values()) + list(seen.get("lane_connector", {}).values())
    centerlines = [list(ln.centerline) for ln in all_lanes if ln.centerline]
    if centerlines:
        s = LAYER_STYLE["centerline"]
        ax.add_collection(
            LineCollection(centerlines, colors=s["color"], linewidths=s["lw"], alpha=s["alpha"], zorder=s["z"])
        )

    _draw_lane_direction_arrows(ax, all_lanes)

    stop_segs = [
        list(sl.polyline)
        for sl in seen.get("stop_line", {}).values()
        if sl.polyline and len(sl.polyline) >= 2
    ]
    if stop_segs:
        s = LAYER_STYLE["stop_line"]
        ax.add_collection(LineCollection(stop_segs, colors=s["color"], linewidths=s["lw"], zorder=s["z"]))

    return all_lanes


def _draw_lane_direction_arrows(ax, lanes, spacing_m: float = 30.0) -> None:
    arrow_x: list[float] = []
    arrow_y: list[float] = []
    arrow_dx: list[float] = []
    arrow_dy: list[float] = []
    for ln in lanes:
        if not ln.centerline or len(ln.centerline) < 2:
            continue
        accumulated = 0.0
        next_at = spacing_m * 0.5
        for i in range(len(ln.centerline) - 1):
            a = ln.centerline[i]
            b = ln.centerline[i + 1]
            seg_dx = b[0] - a[0]
            seg_dy = b[1] - a[1]
            seg_len = math.hypot(seg_dx, seg_dy)
            if seg_len < 1e-6:
                continue
            while accumulated + seg_len >= next_at:
                t = (next_at - accumulated) / seg_len
                arrow_x.append(a[0] + seg_dx * t)
                arrow_y.append(a[1] + seg_dy * t)
                arrow_dx.append(seg_dx / seg_len * 0.55)
                arrow_dy.append(seg_dy / seg_len * 0.55)
                next_at += spacing_m
            accumulated += seg_len
    if arrow_x:
        ax.quiver(
            arrow_x, arrow_y, arrow_dx, arrow_dy,
            angles="xy", scale_units="xy", scale=1.0,
            width=0.0022, color="#4b6fa5", alpha=0.7, zorder=5,
        )


# ----------------------------------------------------------------------
# Per-frame contextual derivations
# ----------------------------------------------------------------------


def _ego_lane(snap: SceneSnapshot, all_lanes) -> Optional[object]:
    """Return the lane whose centerline minimises perpendicular distance to ego.

    Falls back to ``None`` if no centerline is within 4 m (off-road).
    """
    if not all_lanes:
        return None
    ex, ey = snap.ego.pose.x, snap.ego.pose.y
    best, best_d = None, math.inf
    for ln in all_lanes:
        if not ln.centerline or len(ln.centerline) < 2:
            continue
        d = _min_distance_point_to_polyline(ex, ey, ln.centerline)
        if d < best_d:
            best_d, best = d, ln
    return best if best_d < 4.0 else None


def _nearby_lanes(snap: SceneSnapshot, all_lanes, radius_m: float) -> List:
    ex, ey = snap.ego.pose.x, snap.ego.pose.y
    result = []
    for ln in all_lanes:
        if not ln.centerline or len(ln.centerline) < 2:
            continue
        if _min_distance_point_to_polyline(ex, ey, ln.centerline) < radius_m:
            result.append(ln)
    return result


def _nearest_traffic_light_ahead(snap: SceneSnapshot, lane_connectors) -> Tuple[Optional[str], Optional[float]]:
    """For each red/yellow TL, find its connector's centroid relative to the
    ego heading; return (state, distance_m) for the closest one forward."""
    if not snap.traffic_lights:
        return None, None
    lc_by_id = {lc.lane_id: lc for lc in lane_connectors}
    ex, ey = snap.ego.pose.x, snap.ego.pose.y
    cos_h = math.cos(snap.ego.pose.heading)
    sin_h = math.sin(snap.ego.pose.heading)
    best = (None, None, math.inf)
    for tl in snap.traffic_lights:
        state = tl.state.value if hasattr(tl.state, "value") else str(tl.state)
        lc = lc_by_id.get(tl.lane_connector_id)
        if lc is None or not lc.centerline:
            continue
        # use the centerline midpoint
        mx = sum(p[0] for p in lc.centerline) / len(lc.centerline)
        my = sum(p[1] for p in lc.centerline) / len(lc.centerline)
        dx, dy = mx - ex, my - ey
        # forward = projection onto ego heading; require > 0
        forward = dx * cos_h + dy * sin_h
        if forward < 0:
            continue
        dist = math.hypot(dx, dy)
        if dist < best[2]:
            best = (state, dist, dist)
    return best[0], best[1]


def _nearest_lead_distance(snap: SceneSnapshot) -> Optional[float]:
    """Distance to nearest agent in a forward cone (±25° of ego heading)."""
    ex, ey = snap.ego.pose.x, snap.ego.pose.y
    cos_h = math.cos(snap.ego.pose.heading)
    sin_h = math.sin(snap.ego.pose.heading)
    best = math.inf
    for a in snap.agents:
        dx, dy = a.pose.x - ex, a.pose.y - ey
        forward = dx * cos_h + dy * sin_h
        if forward <= 0.0:
            continue
        lateral = -dx * sin_h + dy * cos_h
        if abs(lateral) > forward * math.tan(math.radians(25)):
            continue
        dist = math.hypot(dx, dy)
        if dist < best:
            best = dist
    return None if best == math.inf else best


# ----------------------------------------------------------------------
# render_episode
# ----------------------------------------------------------------------


def render_episode(
    *,
    engine: RuleEngine,
    snapshots: Sequence[SceneSnapshot],
    scenario_name: str,
    output_dir: Path,
    fps: int = 10,
    map_margin_m: float = 60.0,
) -> Dict[str, Path]:
    """Drive ``engine`` over ``snapshots`` and write MP4 + summary PNG + CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ticks = run_episode(engine, snapshots)
    rule_ids = [r.id for r in engine.rules]
    rule_names = {r.id: r.name for r in engine.rules}
    vmat = violation_matrix(ticks, rule_ids)

    csv_path = output_dir / f"{scenario_name}_log.csv"
    _write_csv_log(csv_path, ticks, rule_ids, rule_names)

    mp4_path = output_dir / f"{scenario_name}.mp4"
    _render_mp4(mp4_path, snapshots, ticks, rule_ids, rule_names, vmat, fps, map_margin_m, scenario_name)

    summary_path = output_dir / f"{scenario_name}_summary.png"
    _render_summary(summary_path, ticks, rule_ids, rule_names, vmat, scenario_name, engine)

    return {"mp4": mp4_path, "summary": summary_path, "csv": csv_path}


# ----------------------------------------------------------------------
# CSV writer
# ----------------------------------------------------------------------


def _write_csv_log(path: Path, ticks: Sequence[TickEvaluation],
                   rule_ids: Sequence[str], rule_names: Dict[str, str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "rule_id", "rule_name", "applies", "is_violated", "violation_rate"])
        for tick in ticks:
            for rid in rule_ids:
                ev = tick.by_rule.get(rid)
                if ev is None:
                    continue
                w.writerow([
                    f"{tick.timestamp_s:.3f}", rid, rule_names.get(rid, ""),
                    int(ev.applies), int(ev.is_violated), f"{ev.violation_rate:.6f}",
                ])


# ----------------------------------------------------------------------
# MP4 renderer
# ----------------------------------------------------------------------


def _render_mp4(
    path: Path,
    snapshots: Sequence[SceneSnapshot],
    ticks: Sequence[TickEvaluation],
    rule_ids: Sequence[str],
    rule_names: Dict[str, str],
    vmat: np.ndarray,
    fps: int,
    map_margin_m: float,
    scenario_name: str,
) -> None:
    # Layout:
    #   row 0  HEADER (full width)
    #   row 1  MAP (left)  +  CONTEXT card + RULES panel (right column)
    #   row 2  VIOLATION STRIP (full width)
    #   row 3  LEGEND (full width)
    fig = plt.figure(figsize=(19.0, 11.5), facecolor="white")
    gs = fig.add_gridspec(
        4, 2,
        width_ratios=[4.2, 1.4],
        height_ratios=[0.45, 6.4, 1.6, 0.55],
        hspace=0.18, wspace=0.06,
        left=0.04, right=0.985, top=0.97, bottom=0.04,
    )
    ax_header = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_side = fig.add_subplot(gs[1, 1])
    ax_strip = fig.add_subplot(gs[2, :])
    ax_legend = fig.add_subplot(gs[3, :])

    for ax in (ax_header, ax_side, ax_legend):
        ax.set_axis_off()

    # Header background card
    ax_header.add_patch(
        mpatches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            transform=ax_header.transAxes,
            facecolor="#f5f3ee", edgecolor="#b3aea4", linewidth=0.8, zorder=0,
        )
    )
    header_title = ax_header.text(
        0.012, 0.55, "", transform=ax_header.transAxes,
        va="center", ha="left", fontsize=15, fontweight="bold", color="#1f1f1f",
    )
    header_clock = ax_header.text(
        0.34, 0.55, "", transform=ax_header.transAxes,
        va="center", ha="left", fontsize=12, color="#333", family="DejaVu Sans Mono",
    )
    header_ego = ax_header.text(
        0.56, 0.55, "", transform=ax_header.transAxes,
        va="center", ha="left", fontsize=12, color="#333", family="DejaVu Sans Mono",
    )
    header_status = ax_header.text(
        0.985, 0.55, "", transform=ax_header.transAxes,
        va="center", ha="right", fontsize=13, fontweight="bold",
    )

    # Map static layer
    seen = _collect_static_features(snapshots)
    all_lanes = _draw_static_map(ax_map, seen)
    lane_connectors = list(seen.get("lane_connector", {}).values())
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    # Dynamic map collections
    ego_patch = PolyCollection([], facecolors=[EGO_FC], edgecolors=EGO_EC,
                               linewidths=1.4, alpha=0.95, zorder=10)
    ax_map.add_collection(ego_patch)
    ego_heading_line = LineCollection([], colors=EGO_EC, linewidths=2.2, zorder=11)
    ax_map.add_collection(ego_heading_line)
    ego_trail, = ax_map.plot([], [], color=TRAIL_C, linewidth=1.5, alpha=0.45, zorder=8)
    agent_polys = PolyCollection([], edgecolors="#222", linewidths=0.5, alpha=0.85, zorder=9)
    ax_map.add_collection(agent_polys)
    planned_traj, = ax_map.plot(
        [], [], color=PLANNED_C, linewidth=1.8, linestyle=(0, (4, 2)), alpha=0.75, zorder=8,
    )
    tl_scatter = ax_map.scatter([], [], s=110, c=[], edgecolors="black", linewidths=0.8, zorder=6)

    # Lane-label text artists: pre-allocate a small pool and reuse, so we
    # don't churn the renderer with create/destroy each frame. Pool size 6
    # covers ego + 2 forward + 2 lateral + 1 spare.
    LANE_LABEL_POOL = 6
    lane_label_artists = [
        ax_map.text(
            0, 0, "", fontsize=9, color="#37618e", ha="center", va="center",
            alpha=0.0, zorder=5, fontweight="bold",
        )
        for _ in range(LANE_LABEL_POOL)
    ]

    # Sidebar — context card on top, rules panel below.
    # We render the card as a rounded white box with monospace key/value rows;
    # rules panel is a series of single-row text artists so each can be
    # coloured independently by severity.
    ax_side.add_patch(
        mpatches.FancyBboxPatch(
            (0.0, 0.62), 1.0, 0.38,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            transform=ax_side.transAxes,
            facecolor="#fbfaf6", edgecolor="#cfc9bb", linewidth=0.8, zorder=0,
        )
    )
    ax_side.text(
        0.06, 0.96, "CONTEXT", transform=ax_side.transAxes,
        va="top", ha="left", fontsize=11, fontweight="bold", color="#6b6258",
    )
    ctx_text = ax_side.text(
        0.06, 0.91, "", transform=ax_side.transAxes,
        va="top", ha="left", fontsize=11, family="DejaVu Sans Mono", color="#1f1f1f",
    )

    # Rules card
    ax_side.add_patch(
        mpatches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 0.58,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            transform=ax_side.transAxes,
            facecolor="#fbfaf6", edgecolor="#cfc9bb", linewidth=0.8, zorder=0,
        )
    )
    ax_side.text(
        0.06, 0.555, "ACTIVE RULES", transform=ax_side.transAxes,
        va="top", ha="left", fontsize=11, fontweight="bold", color="#6b6258",
    )
    RULE_ROWS = 16  # cap rendered rows — beyond this gets summarised
    rule_row_artists = [
        ax_side.text(
            0.04, 0.50 - i * 0.030, "", transform=ax_side.transAxes,
            va="top", ha="left", fontsize=10, family="DejaVu Sans Mono", color="#1f1f1f",
        )
        for i in range(RULE_ROWS)
    ]
    rules_overflow_text = ax_side.text(
        0.04, 0.50 - RULE_ROWS * 0.030, "", transform=ax_side.transAxes,
        va="top", ha="left", fontsize=10, color="#888", style="italic",
    )

    # Strip
    strip_im = ax_strip.imshow(
        vmat, aspect="auto", cmap=VIOLATION_CMAP,
        vmin=0.0, vmax=max(vmat.max(), 1e-3),
        interpolation="nearest",
        extent=[0, len(ticks), len(rule_ids), 0],
    )
    ax_strip.set_yticks(np.arange(len(rule_ids)) + 0.5)
    ax_strip.set_yticklabels(rule_ids, fontsize=9)
    ax_strip.tick_params(axis="x", labelsize=10)
    ax_strip.set_xlabel("tick", fontsize=11)
    ax_strip.set_title("Per-rule violation rate", fontsize=12, fontweight="bold", loc="left")
    time_cursor = ax_strip.axvline(0, color="black", linewidth=1.2)
    cb = fig.colorbar(strip_im, ax=ax_strip, fraction=0.018, pad=0.005)
    cb.set_label("violation rate", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # Legend row (no map overlay anymore)
    legend_handles = [
        mpatches.Patch(facecolor=LAYER_STYLE["drivable"]["fc"], edgecolor="#b3aea4", label="Drivable"),
        mpatches.Patch(facecolor=LAYER_STYLE["road_lane"]["fc"], edgecolor="#b3aea4", label="Travel lane"),
        mpatches.Patch(facecolor=LAYER_STYLE["lane_connector"]["fc"], edgecolor="#a59b87", label="Lane connector"),
        mpatches.Patch(facecolor=LAYER_STYLE["crosswalk"]["fc"], edgecolor="#c4a93f", label="Crosswalk"),
        mpatches.Patch(facecolor=LAYER_STYLE["walkway"]["fc"], edgecolor="#9bbf95", label="Walkway"),
        mpatches.Patch(facecolor=LAYER_STYLE["intersection"]["fc"], edgecolor="#b3aea4", label="Intersection"),
        mpatches.Patch(facecolor=EGO_FC, edgecolor=EGO_EC, label="Ego"),
        mpatches.Patch(facecolor=AGENT_COLORS[AgentType.VEHICLE.value], edgecolor="#222", label="Vehicle"),
        mpatches.Patch(facecolor=AGENT_COLORS[AgentType.PEDESTRIAN.value], edgecolor="#222", label="Pedestrian"),
        mpatches.Patch(facecolor=AGENT_COLORS[AgentType.BICYCLE.value], edgecolor="#222", label="Bicycle"),
        mpatches.Patch(facecolor="none", edgecolor=PLANNED_C, linewidth=1.8, label="Planned path"),
        mpatches.Patch(facecolor="none", edgecolor=TRAIL_C, linewidth=1.5, label="Ego trail"),
        mpatches.Patch(facecolor="#d62728", edgecolor="black", label="TL: red"),
        mpatches.Patch(facecolor="#f1c40f", edgecolor="black", label="TL: yellow"),
        mpatches.Patch(facecolor="#2ca02c", edgecolor="black", label="TL: green"),
        mpatches.Patch(facecolor=LAYER_STYLE["stop_line"]["color"], edgecolor=LAYER_STYLE["stop_line"]["color"], label="Stop line"),
    ]
    ax_legend.legend(
        handles=legend_handles, loc="center", ncol=8, frameon=False,
        fontsize=10, handlelength=1.4, handletextpad=0.5, columnspacing=1.4,
    )

    # ------------------------------------------------------------------
    def update(idx: int):
        snap = snapshots[idx]
        tick = ticks[idx]
        ego = snap.ego

        # ----- ego + agents -----
        ego_box = _oriented_box_xy(ego.pose.x, ego.pose.y, ego.length, ego.width, ego.pose.heading)
        ego_patch.set_verts([ego_box])
        cos_h, sin_h = math.cos(ego.pose.heading), math.sin(ego.pose.heading)
        ego_heading_line.set_segments([[(ego.pose.x, ego.pose.y),
                                        (ego.pose.x + 4.0 * cos_h, ego.pose.y + 4.0 * sin_h)]])
        trail = [(s.ego.pose.x, s.ego.pose.y) for s in snapshots[: idx + 1]]
        ego_trail.set_data([p[0] for p in trail], [p[1] for p in trail])

        boxes, colors = [], []
        for a in snap.agents:
            ot = a.object_type.value if hasattr(a.object_type, "value") else str(a.object_type)
            boxes.append(_oriented_box_xy(a.pose.x, a.pose.y, a.length, a.width, a.pose.heading))
            colors.append(AGENT_COLORS.get(ot, "#999"))
        agent_polys.set_verts(boxes)
        agent_polys.set_facecolors(colors)

        if snap.planned_trajectory:
            planned_traj.set_data(
                [p.pose.x for p in snap.planned_trajectory],
                [p.pose.y for p in snap.planned_trajectory],
            )
        else:
            planned_traj.set_data([], [])

        # ----- traffic-light markers (dynamic per tick) -----
        tl_x, tl_y, tl_c = [], [], []
        lc_by_id = {lc.lane_id: lc for lc in lane_connectors}
        for tl in snap.traffic_lights:
            lc = lc_by_id.get(tl.lane_connector_id)
            if lc is None or not lc.centerline:
                continue
            cx = sum(p[0] for p in lc.centerline) / len(lc.centerline)
            cy = sum(p[1] for p in lc.centerline) / len(lc.centerline)
            tl_x.append(cx)
            tl_y.append(cy)
            state_str = tl.state.value if hasattr(tl.state, "value") else str(tl.state)
            tl_c.append(TL_COLORS.get(state_str, "#888"))
        if tl_x:
            tl_scatter.set_offsets(np.column_stack([tl_x, tl_y]))
            tl_scatter.set_facecolors(tl_c)
        else:
            tl_scatter.set_offsets(np.empty((0, 2)))

        # ----- lane labels: only ego-nearby lanes, speed limit only -----
        nearby = _nearby_lanes(snap, all_lanes, LANE_LABEL_RADIUS_M)
        nearby_with_v = [ln for ln in nearby if ln.speed_limit_mps is not None]
        for art_idx, ln in enumerate(nearby_with_v[:LANE_LABEL_POOL]):
            tx, ty, lane_heading = _point_along_polyline(ln.centerline, frac=0.5)
            nx = -math.sin(lane_heading)
            ny = math.cos(lane_heading)
            art = lane_label_artists[art_idx]
            art.set_position((tx + nx * 1.4, ty + ny * 1.4))
            art.set_text(f"{ln.speed_limit_mps * 2.237:.0f}")
            art.set_alpha(0.9)
        for art_idx in range(len(nearby_with_v[:LANE_LABEL_POOL]), LANE_LABEL_POOL):
            lane_label_artists[art_idx].set_alpha(0.0)
            lane_label_artists[art_idx].set_text("")

        # ----- viewport -----
        ax_map.set_xlim(ego.pose.x - map_margin_m, ego.pose.x + map_margin_m)
        ax_map.set_ylim(ego.pose.y - map_margin_m, ego.pose.y + map_margin_m)

        # ----- header -----
        n_apply = sum(1 for e in tick.by_rule.values() if e.applies)
        n_violate = sum(1 for e in tick.by_rule.values() if e.applies and e.is_violated)
        header_title.set_text(scenario_name)
        header_clock.set_text(f"tick {idx + 1:>3d}/{len(snapshots):>3d}   t = {tick.timestamp_s:6.2f} s")
        header_ego.set_text(
            f"v={ego.speed:5.2f}m/s  ax={ego.ax:+4.1f}  ay={ego.ay:+4.1f}"
        )
        if n_violate == 0:
            header_status.set_text(f"✓ COMPLIANT  ({n_apply} appl)")
            header_status.set_color("#1d8a44")
        else:
            header_status.set_text(f"✗ {n_violate} VIOL / {n_apply} APPL")
            header_status.set_color("#c0392b")

        # ----- context card -----
        ego_ln = _ego_lane(snap, all_lanes)
        v_lim_mps = ego_ln.speed_limit_mps if ego_ln and ego_ln.speed_limit_mps is not None else None
        v_lim_mph = f"{v_lim_mps * 2.237:>5.1f} mph" if v_lim_mps is not None else "  -- mph"
        lane_id = (ego_ln.lane_id[:16] if ego_ln else "—") if ego_ln else "off-road"
        tl_state, tl_dist = _nearest_traffic_light_ahead(snap, lane_connectors)
        if tl_state is None:
            tl_line = "TL ahead    none in view"
        else:
            tl_line = f"TL ahead    {tl_state:<6}  {tl_dist:5.1f} m"
        lead_d = _nearest_lead_distance(snap)
        lead_line = f"lead agent  {lead_d:5.1f} m" if lead_d is not None else "lead agent  none"
        ctx_text.set_text(
            f"lane        {lane_id:<16}\n"
            f"v_lim       {v_lim_mph}\n"
            f"ego speed   {ego.speed:>5.2f} m/s\n"
            f"{tl_line}\n"
            f"{lead_line}\n"
            f"applicable  {n_apply} / {len(rule_ids)} rules"
        )

        # ----- rules panel (colour per row) -----
        applicable = [(rid, e) for rid, e in tick.by_rule.items() if e.applies]
        applicable.sort(key=lambda x: (-x[1].violation_rate, x[0]))
        for i in range(RULE_ROWS):
            art = rule_row_artists[i]
            if i < len(applicable):
                rid, e = applicable[i]
                tag = "✗" if e.is_violated else "·"
                # severity-based row colour
                if e.is_violated:
                    color = "#c0392b" if e.violation_rate >= 0.6 else "#e07b3a"
                    weight = "bold"
                else:
                    color = "#2d4633"
                    weight = "normal"
                art.set_text(f"{tag} {rid:<5} {e.violation_rate:6.3f}  {rule_names.get(rid, '')[:24]}")
                art.set_color(color)
                art.set_fontweight(weight)
            else:
                art.set_text("")
        if len(applicable) > RULE_ROWS:
            rules_overflow_text.set_text(f"+{len(applicable) - RULE_ROWS} more applicable")
        else:
            rules_overflow_text.set_text("")

        time_cursor.set_xdata([idx + 0.5, idx + 0.5])

        return (
            ego_patch, ego_heading_line, ego_trail, agent_polys, planned_traj,
            tl_scatter, header_title, header_clock, header_ego, header_status,
            ctx_text, *rule_row_artists, rules_overflow_text, time_cursor, *lane_label_artists,
        )

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2600)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(path), writer=writer, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Episode summary
# ----------------------------------------------------------------------


def _render_summary(
    path: Path,
    ticks: Sequence[TickEvaluation],
    rule_ids: Sequence[str],
    rule_names: Dict[str, str],
    vmat: np.ndarray,
    scenario_name: str,
    engine: RuleEngine,
) -> None:
    summary = engine.summary()
    integ = np.array([summary.rule_summaries[rid].integrated_violation for rid in rule_ids])
    applic = np.array([summary.rule_summaries[rid].n_steps_applicable for rid in rule_ids])
    violated = np.array([summary.rule_summaries[rid].n_steps_violated for rid in rule_ids])

    fig = plt.figure(figsize=(16.0, 9.5), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.65], height_ratios=[1.0, 1.0],
        hspace=0.55, wspace=0.28,
        left=0.06, right=0.985, top=0.92, bottom=0.08,
    )

    # --- Sorted integrated-violation bar chart ---
    ax_bar = fig.add_subplot(gs[:, 0])
    sort_idx = np.argsort(-integ)
    rids_sorted = [rule_ids[i] for i in sort_idx]
    integ_sorted = integ[sort_idx]
    colors_bar = ["#c0392b" if v > 0 else "#c9c2b3" for v in integ_sorted]
    bars = ax_bar.barh(range(len(rids_sorted)), integ_sorted, color=colors_bar, alpha=0.92)
    # Annotate non-zero bars with their value
    for bar, v in zip(bars, integ_sorted):
        if v > 0:
            ax_bar.text(v, bar.get_y() + bar.get_height() / 2.0, f"  {v:.2f}",
                        va="center", ha="left", fontsize=9, color="#1f1f1f")
    ax_bar.set_yticks(range(len(rids_sorted)))
    ax_bar.set_yticklabels(
        [f"{rid}  {rule_names.get(rid, '')[:24]}" for rid in rids_sorted], fontsize=10
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(r"integrated violation  $\int \mathrm{rate}\,dt$  [unit·s]", fontsize=12)
    ax_bar.set_title("Episode totals (sorted)", fontsize=13, fontweight="bold", loc="left")
    ax_bar.tick_params(axis="x", labelsize=10)
    ax_bar.grid(True, axis="x", color="#ddd", linewidth=0.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- Per-rule violation-rate heatmap timeline ---
    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(vmat, aspect="auto", cmap=VIOLATION_CMAP,
                     vmin=0.0, vmax=max(vmat.max(), 1e-3), interpolation="nearest")
    ax_h.set_yticks(range(len(rule_ids)))
    ax_h.set_yticklabels(rule_ids, fontsize=9)
    ax_h.tick_params(axis="x", labelsize=10)
    ax_h.set_xlabel("tick", fontsize=12)
    ax_h.set_title("Per-rule violation rate over the episode", fontsize=13, fontweight="bold", loc="left")
    cb_h = fig.colorbar(im, ax=ax_h, fraction=0.022, pad=0.01)
    cb_h.set_label("violation rate", fontsize=10)
    cb_h.ax.tick_params(labelsize=9)

    # --- Applicability vs violation count per rule ---
    ax_c = fig.add_subplot(gs[1, 1])
    x = np.arange(len(rule_ids))
    ax_c.bar(x - 0.2, applic, width=0.4, color="#4472c4", label="applicable ticks")
    ax_c.bar(x + 0.2, violated, width=0.4, color="#c0392b", label="violating ticks")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(rule_ids, rotation=70, fontsize=9)
    ax_c.tick_params(axis="y", labelsize=10)
    ax_c.set_ylabel("# ticks", fontsize=12)
    ax_c.set_title("Applicability vs violation count per rule", fontsize=13, fontweight="bold", loc="left")
    ax_c.legend(fontsize=10, loc="upper right", frameon=False)
    ax_c.grid(True, axis="y", color="#ddd", linewidth=0.5)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    n_violating = int(sum(1 for s in summary.rule_summaries.values() if s.n_steps_violated > 0))
    top_rid = rids_sorted[0] if rids_sorted else "-"
    top_val = integ_sorted[0] if len(integ_sorted) else 0.0
    fig.suptitle(
        f"{scenario_name}   |   {summary.duration_s:.1f} s   |   "
        f"violating rules: {n_violating}   |   top: {top_rid}  ({top_val:.2f})",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
