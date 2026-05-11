#!/usr/bin/env python
"""Per-episode jitter plots of pose vs feature evidence for the primary target.

Reads JSONL files written by `hypothesis_evidence_logger`, keeps only records
whose `graph_id` is the episode's primary target object, and writes one PNG per
episode into a sibling `<input_dir>_plots/` folder. Only the `learning_module_0`
input channel is used. Each step gets two x-jittered columns: pose evidence
(left) and feature evidence (right), one dot per tested hypothesis. There is
x-jitter but no y-jitter. The MLH hypothesis is highlighted with a larger
black-edged marker. Below the main plot, up to four small pose-vs-feature
scatter subplots are drawn for randomly chosen steps; each highlights the MLH
hypothesis (ring) and the hypothesis with the highest pose+feature evidence
(star). Out-of-reference-frame hypotheses are dropped throughout. If a
`graph_nodes.npz` (written by `hypothesis_evidence_logger.dump_graphs`) is
present in the input dir, the primary target's `learning_module_0` graph nodes
are drawn as small 3D insets stacked under the legend -- one per distinct
per-node `object_id` value, all sharing the same spatial bounds, with a single
color legend above them. Style mirrors the prediction error scatter plots in
the `debug_pred_error` branch.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

INPUT_CHANNEL = "learning_module_0"

# Hypotheses whose nearest stored node is farther than this (the LM's
# `max_match_distance`) are out of reference frame: they always get pose
# evidence -1 and feature evidence 0. Such hypotheses are asserted to look
# like that and then dropped from the plot.
MAX_MATCH_DISTANCE = 0.01

POSE_COLOR = "#4C72B0"
FEATURE_COLOR = "#DD8452"

X_JITTER_OFFSET = 0.20  # column center, offset from the integer step
X_JITTER_WIDTH = 0.15  # half-width of the uniform x-jitter within a column

SCATTER_SIZE = 6
SCATTER_ALPHA = 0.35
MLH_SIZE = 45

NUM_SUBPLOT_STEPS = 4  # max number of per-step pose-vs-feature subplots

SUB_POINT_COLOR = "#888888"
SUB_SCATTER_SIZE = 9
SUB_SCATTER_ALPHA = 0.4
SUB_MLH_SIZE = 140  # hollow ring around the MLH hypothesis
SUB_BEST_COLOR = "#C44E52"
SUB_BEST_SIZE = 90  # star on the highest pose+feature hypothesis

GRAPH_NODE_CMAP = "tab10"  # color cycle for object_id classes in the node inset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input_dir",
        "-i",
        required=True,
        type=Path,
        help="Folder containing worker_*.jsonl files for one run.",
    )
    p.add_argument(
        "--num_episodes",
        "-n",
        type=int,
        default=None,
        help="Number of files to randomly sample. Default: all files.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for file sampling and jitter.",
    )
    return p.parse_args()


def load_files(files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_json(p, lines=True) for p in files], ignore_index=True)


def load_graph_nodes(
    input_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[float, str]]:
    """Load `graph_nodes.npz` if present.

    Returns:
        `(graph_nodes, code_to_name)` where `graph_nodes` maps
        `f"{lm_id}__{graph_id}__{channel}"` (and `..__object_id`) to arrays, and
        `code_to_name` maps each `object_id` code to its object name. Both are
        empty if the file is missing.
    """
    path = input_dir / "graph_nodes.npz"
    if not path.exists():
        return {}, {}
    with np.load(path) as npz:
        data = {k: npz[k] for k in npz.files}
    codes = data.pop("__object_id_codes", None)
    names = data.pop("__object_id_names", None)
    code_to_name: dict[float, str] = {}
    if codes is not None and names is not None:
        code_to_name = {float(c): str(n) for c, n in zip(codes, names)}
    return data, code_to_name


def explode_primary_target(group: pd.DataFrame, primary_target: str) -> pd.DataFrame:
    """Long-form rows for the primary target's tested hypotheses.

    One row per (step, tested hypothesis), restricted to the `INPUT_CHANNEL`
    input channel. Out-of-reference-frame hypotheses (nearest stored node
    farther than `MAX_MATCH_DISTANCE`) are asserted to have pose evidence -1
    and feature evidence 0, then dropped.

    Returns:
        DataFrame with columns `step`, `pose_evidence`, `feature_evidence`,
        `is_mlh`.
    """
    steps: list[int] = []
    pose: list[float] = []
    feature: list[float] = []
    is_mlh: list[bool] = []
    nearest: list[float] = []
    for _, row in group[group["graph_id"] == primary_target].iterrows():
        channel_values = row["channels"].get(INPUT_CHANNEL)
        if channel_values is None:
            continue
        tested = np.asarray(row["hyp_idxs_tested"], dtype=np.int_)
        mlh_pos = np.where(tested == int(row["mlh_index"]))[0]
        channel_pose = np.asarray(channel_values["pose_evidence"], dtype=np.float64)
        channel_feature = np.asarray(
            channel_values["feature_evidence"], dtype=np.float64
        )
        channel_nearest = np.asarray(
            channel_values["custom_nearest_distance"], dtype=np.float64
        )
        n = len(channel_pose)
        mlh_flags = np.zeros(n, dtype=bool)
        if mlh_pos.size:
            mlh_flags[mlh_pos[0]] = True
        steps.extend([int(row["step"])] * n)
        pose.extend(channel_pose.tolist())
        feature.extend(channel_feature.tolist())
        is_mlh.extend(mlh_flags.tolist())
        nearest.extend(channel_nearest.tolist())

    step_arr = np.asarray(steps, dtype=np.int_)
    pose_arr = np.asarray(pose, dtype=np.float64)
    feature_arr = np.asarray(feature, dtype=np.float64)
    mlh_arr = np.asarray(is_mlh, dtype=bool)
    nearest_arr = np.asarray(nearest, dtype=np.float64)

    out_of_rf = nearest_arr > MAX_MATCH_DISTANCE
    assert np.all(pose_arr[out_of_rf] == -1.0), (
        "out-of-frame hypotheses should have pose evidence -1, got "
        f"{np.unique(pose_arr[out_of_rf])}"
    )
    assert np.all(feature_arr[out_of_rf] == 0.0), (
        "out-of-frame hypotheses should have feature evidence 0, got "
        f"{np.unique(feature_arr[out_of_rf])}"
    )
    keep = ~out_of_rf
    return pd.DataFrame(
        {
            "step": step_arr[keep],
            "pose_evidence": pose_arr[keep],
            "feature_evidence": feature_arr[keep],
            "is_mlh": mlh_arr[keep],
        }
    )


def _column_x(
    steps: np.ndarray, sign: int, rng: np.random.Generator
) -> np.ndarray:
    return (
        steps.astype(np.float64)
        + sign * X_JITTER_OFFSET
        + rng.uniform(-X_JITTER_WIDTH, X_JITTER_WIDTH, size=len(steps))
    )


def _scatter_column(ax, steps, values, sign, color, rng) -> None:
    ax.scatter(
        _column_x(steps, sign, rng),
        values,
        color=color,
        marker="o",
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        zorder=2,
    )


def _scatter_mlh(ax, steps, values, sign, color, rng) -> None:
    if len(steps) == 0:
        return
    ax.scatter(
        _column_x(steps, sign, rng),
        values,
        color=color,
        marker="o",
        s=MLH_SIZE,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )


def _pick_subplot_steps(steps: np.ndarray, rng: np.random.Generator) -> list[int]:
    unique_steps = np.unique(steps)
    count = min(NUM_SUBPLOT_STEPS, len(unique_steps))
    chosen = rng.choice(unique_steps, size=count, replace=False)
    return sorted(int(s) for s in chosen)


def _plot_step_scatter(
    ax, step_df: pd.DataFrame, step: int, show_ylabel: bool
) -> None:
    pose = step_df["pose_evidence"].to_numpy()
    feature = step_df["feature_evidence"].to_numpy()
    is_mlh = step_df["is_mlh"].to_numpy()

    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":", alpha=0.6, zorder=1)
    ax.scatter(
        feature,
        pose,
        s=SUB_SCATTER_SIZE,
        color=SUB_POINT_COLOR,
        alpha=SUB_SCATTER_ALPHA,
        linewidths=0,
        zorder=2,
    )

    best = int(np.argmax(pose + feature))
    ax.scatter(
        feature[best],
        pose[best],
        marker="*",
        s=SUB_BEST_SIZE,
        color=SUB_BEST_COLOR,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )
    if is_mlh.any():
        mlh = int(np.argmax(is_mlh))
        ax.scatter(
            feature[mlh],
            pose[mlh],
            marker="o",
            s=SUB_MLH_SIZE,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

    ax.set_title(f"step {step}", fontsize=9)
    ax.set_xlabel("feature evidence", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("pose evidence", fontsize=8)
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(labelsize=7)


def _add_graph_node_insets(
    fig,
    legend,
    primary_target: str,
    graph_nodes: dict[str, np.ndarray],
    code_to_name: dict[float, str],
) -> None:
    """Draw the primary target's INPUT_CHANNEL graph nodes split by object_id.

    One small 3D subplot per distinct per-node `object_id` value, stacked
    vertically in the white area under `legend`, all sharing the same spatial
    bounds so the colored subsets are spatially comparable. A single color
    legend mapping each `object_id` to its name sits above the stack. If the
    `object_id` feature was not dumped, a single uncolored subplot is drawn.
    """
    suffix = f"__{primary_target}__{INPUT_CHANNEL}"
    pos_keys = [
        k
        for k in graph_nodes
        if k.endswith(suffix) and not k.endswith("__object_id")
    ]
    if not pos_keys:
        return
    points = np.asarray(graph_nodes[pos_keys[0]], dtype=np.float64)
    if points.size == 0:
        return
    object_ids = graph_nodes.get(f"{pos_keys[0]}__object_id")

    cmap = plt.get_cmap(GRAPH_NODE_CMAP)
    groups: list[tuple[str, object, np.ndarray]] = []
    if object_ids is None:
        groups.append(
            (f"{primary_target}\n{INPUT_CHANNEL}", SUB_POINT_COLOR,
             np.ones(len(points), dtype=bool))
        )
    else:
        object_ids = np.asarray(object_ids, dtype=np.float64)
        for i, code in enumerate(np.unique(object_ids)):
            groups.append(
                (code_to_name.get(code, f"id {code:g}"), cmap(i % cmap.N),
                 object_ids == code)
            )

    fig.canvas.draw()
    leg_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    left = leg_bbox.x0
    width = max(leg_bbox.width, 0.14)

    if object_ids is None:
        stack_top = leg_bbox.y0 - 0.03
    else:
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=color, markeredgecolor="none", markersize=7,
                label=name,
            )
            for name, color, _ in groups
        ]
        color_legend = fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(left, leg_bbox.y0 - 0.02),
            fontsize=7,
            framealpha=0.9,
            title="node object_id",
            title_fontsize=7,
        )
        fig.canvas.draw()
        colorleg_bbox = color_legend.get_window_extent().transformed(
            fig.transFigure.inverted()
        )
        stack_top = colorleg_bbox.y0 - 0.01

    bottom = 0.04
    slot_h = (stack_top - bottom) / len(groups)

    half = (points.max(axis=0) - points.min(axis=0)).max() / 2 or 1.0
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    for i, (name, color, mask) in enumerate(groups):
        slot_bottom = stack_top - (i + 1) * slot_h
        ax3d = fig.add_axes(
            [left, slot_bottom + slot_h * 0.05, width, slot_h * 0.88],
            projection="3d",
        )
        ax3d.scatter(
            points[mask, 0], points[mask, 1], points[mask, 2],
            color=color, s=5, alpha=0.7, edgecolors="none",
        )
        ax3d.set_xlim(center[0] - half, center[0] + half)
        ax3d.set_ylim(center[1] - half, center[1] + half)
        ax3d.set_zlim(center[2] - half, center[2] + half)
        ax3d.set_axis_off()
        ax3d.view_init(elev=30, azim=45)
        ax3d.text2D(
            0.0, 1.0, f"{name}  (n={int(mask.sum())})",
            transform=ax3d.transAxes, fontsize=6, va="top",
        )


def plot_episode(
    df: pd.DataFrame,
    episode: int,
    primary_target: str,
    out_path: Path,
    rng: np.random.Generator,
    graph_nodes: dict[str, np.ndarray],
    code_to_name: dict[float, str],
) -> None:
    chosen_steps = _pick_subplot_steps(df["step"].to_numpy(), rng)

    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.3)
    ax = fig.add_subplot(grid[0])
    sub_grid = grid[1].subgridspec(1, max(len(chosen_steps), 1), wspace=0.3)
    sub_axes = [fig.add_subplot(sub_grid[0, i]) for i in range(len(chosen_steps))]

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)

    steps = df["step"].to_numpy()
    pose = df["pose_evidence"].to_numpy()
    feature = df["feature_evidence"].to_numpy()
    mlh = df["is_mlh"].to_numpy()

    _scatter_column(ax, steps[~mlh], pose[~mlh], -1, POSE_COLOR, rng)
    _scatter_column(ax, steps[~mlh], feature[~mlh], +1, FEATURE_COLOR, rng)
    _scatter_mlh(ax, steps[mlh], pose[mlh], -1, POSE_COLOR, rng)
    _scatter_mlh(ax, steps[mlh], feature[mlh], +1, FEATURE_COLOR, rng)

    ax.set_xlabel("step")
    ax.set_ylabel("evidence")
    ax.set_title(f"Episode {episode} - {primary_target}: pose vs feature evidence")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=POSE_COLOR,
            markeredgecolor=POSE_COLOR,
            markersize=8,
            label="pose evidence",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=FEATURE_COLOR,
            markeredgecolor=FEATURE_COLOR,
            markersize=8,
            label="feature evidence",
        ),
    ]
    if mlh.any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=9,
                label="MLH hypothesis",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=SUB_BEST_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=13,
            label="highest pose+feature (subplots)",
        )
    )
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0.0,
    )

    for i, (step, sub_ax) in enumerate(zip(chosen_steps, sub_axes)):
        _plot_step_scatter(sub_ax, df[df["step"] == step], step, show_ylabel=(i == 0))

    if graph_nodes:
        _add_graph_node_insets(fig, legend, primary_target, graph_nodes, code_to_name)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found in {args.input_dir}")

    if args.num_episodes is not None and args.num_episodes < len(files):
        files = random.Random(args.seed).sample(files, args.num_episodes)

    output_dir = args.input_dir.parent / f"{args.input_dir.name}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_nodes, code_to_name = load_graph_nodes(args.input_dir)
    if graph_nodes:
        print(f"Loaded {len(graph_nodes)} graph-node arrays from graph_nodes.npz")
    else:
        print("No graph_nodes.npz in input dir — node inset will be skipped.")

    df = load_files(files)
    rng = np.random.default_rng(args.seed)

    for episode, group in df.groupby("episode", sort=True):
        primary_target = (
            str(group["primary_target_graph_id"].iloc[0])
            if "primary_target_graph_id" in group
            else ""
        )
        if not primary_target:
            print(f"Episode {episode}: no primary_target_graph_id, skipping.")
            continue

        ep_df = explode_primary_target(group, primary_target)
        if ep_df.empty:
            print(
                f"Episode {episode}: no {INPUT_CHANNEL} records for primary target "
                f"{primary_target}."
            )
            continue

        out = output_dir / f"episode_{episode}.png"
        plot_episode(
            ep_df, int(episode), primary_target, out, rng, graph_nodes, code_to_name
        )
        print(f"Wrote {out}  ({len(ep_df)} points)")


if __name__ == "__main__":
    main()
