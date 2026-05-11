#!/usr/bin/env python
"""Per-episode jitter plots of pose vs feature evidence for the primary target.

Reads JSONL files written by `hypothesis_evidence_logger`, keeps only records
whose `graph_id` is the episode's primary target object, and writes one PNG per
episode into a sibling `<input_dir>_plots/` folder. Only the `learning_module_0`
input channel is used. Each step gets two x-jittered columns: pose evidence
(left) and feature evidence (right), one dot per tested hypothesis. There is
x-jitter but no y-jitter. The MLH hypothesis is highlighted with a larger
black-edged marker, and sampling steps are marked along the bottom. Style
mirrors the prediction error scatter plots in the `debug_pred_error` branch.
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

POSE_COLOR = "#4C72B0"
FEATURE_COLOR = "#DD8452"

X_JITTER_OFFSET = 0.20  # column center, offset from the integer step
X_JITTER_WIDTH = 0.15  # half-width of the uniform x-jitter within a column

SCATTER_SIZE = 6
SCATTER_ALPHA = 0.35
MLH_SIZE = 45


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


def consecutive_ranges(steps_sorted: list[int]) -> list[tuple[int, int]]:
    if not steps_sorted:
        return []
    ranges = []
    start = prev = steps_sorted[0]
    for s in steps_sorted[1:]:
        if s == prev + 1:
            prev = s
        else:
            ranges.append((start, prev))
            start = prev = s
    ranges.append((start, prev))
    return ranges


def explode_primary_target(group: pd.DataFrame, primary_target: str) -> pd.DataFrame:
    """Long-form rows for the primary target's tested hypotheses.

    One row per (step, tested hypothesis), restricted to the `INPUT_CHANNEL`
    input channel.

    Returns:
        DataFrame with columns `step`, `pose_evidence`, `feature_evidence`,
        `is_mlh`.
    """
    steps: list[int] = []
    pose: list[float] = []
    feature: list[float] = []
    is_mlh: list[bool] = []
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
        n = len(channel_pose)
        mlh_flags = np.zeros(n, dtype=bool)
        if mlh_pos.size:
            mlh_flags[mlh_pos[0]] = True
        steps.extend([int(row["step"])] * n)
        pose.extend(channel_pose.tolist())
        feature.extend(channel_feature.tolist())
        is_mlh.extend(mlh_flags.tolist())
    return pd.DataFrame(
        {
            "step": np.asarray(steps, dtype=np.int_),
            "pose_evidence": np.asarray(pose, dtype=np.float64),
            "feature_evidence": np.asarray(feature, dtype=np.float64),
            "is_mlh": np.asarray(is_mlh, dtype=bool),
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


def plot_episode(
    df: pd.DataFrame,
    episode: int,
    primary_target: str,
    sampling_ranges: list[tuple[int, int]],
    out_path: Path,
    rng: np.random.Generator,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)

    steps = df["step"].to_numpy()
    pose = df["pose_evidence"].to_numpy()
    feature = df["feature_evidence"].to_numpy()
    mlh = df["is_mlh"].to_numpy()

    _scatter_column(ax, steps[~mlh], pose[~mlh], -1, POSE_COLOR, rng)
    _scatter_column(ax, steps[~mlh], feature[~mlh], +1, FEATURE_COLOR, rng)
    _scatter_mlh(ax, steps[mlh], pose[mlh], -1, POSE_COLOR, rng)
    _scatter_mlh(ax, steps[mlh], feature[mlh], +1, FEATURE_COLOR, rng)

    for start, end in sampling_ranges:
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            ymin=0.0,
            ymax=0.025,
            color="red",
            alpha=0.9,
            zorder=0.5,
        )

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
    if sampling_ranges:
        handles.append(
            Line2D([0], [0], color="red", linewidth=6, label="sampling steps")
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0.0,
    )
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

        if "is_sampling" in group:
            sampling_steps = sorted(
                {int(s) for s in group.loc[group["is_sampling"], "step"]}
            )
        else:
            sampling_steps = []
        sampling_ranges = consecutive_ranges(sampling_steps)

        ep_df = explode_primary_target(group, primary_target)
        if ep_df.empty:
            print(
                f"Episode {episode}: no {INPUT_CHANNEL} records for primary target "
                f"{primary_target}."
            )
            continue

        out = output_dir / f"episode_{episode}.png"
        plot_episode(ep_df, int(episode), primary_target, sampling_ranges, out, rng)
        print(f"Wrote {out}  ({len(ep_df)} points)")


if __name__ == "__main__":
    main()
