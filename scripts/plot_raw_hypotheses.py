#!/usr/bin/env python
"""Per-episode prediction-error scatter plots from hypothesis-evidence JSONL.

Reads JSONL files written by `hypothesis_evidence_logger`, computes prediction
error per hypothesis, and writes one PNG per episode where every dot is one
hypothesis. Dots are jittered horizontally and colored by graph_id; the MLH
for each step is rendered on top with a larger black-edged marker.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MAX_EVIDENCE = 2.0
EVIDENCE_RANGE = 3.0  # MAX_EVIDENCE - MIN_EVIDENCE  (MIN_EVIDENCE = -1)
POSE_RANGE = 2.0  # pose evidence ∈ [-1, 1]
FEATURE_RANGE = 1.0  # feature evidence ∈ [0, 1]
POSE_FILL_COLOR = "#4C72B0"
FEATURE_FILL_COLOR = "#DD8452"


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
        "--output_dir",
        "-o",
        required=True,
        type=Path,
        help="Folder to write per-episode PNGs into.",
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
    return pd.concat(
        [pd.read_json(p, lines=True) for p in files], ignore_index=True
    )


def explode(df: pd.DataFrame) -> pd.DataFrame:
    lengths = df["evidence"].apply(len).to_numpy()
    return pd.DataFrame(
        {
            "episode": np.repeat(df["episode"].to_numpy(), lengths),
            "step": np.repeat(df["step"].to_numpy(), lengths),
            "graph_id": np.repeat(df["graph_id"].to_numpy(), lengths),
            "mlh_index": np.repeat(df["mlh_index"].to_numpy(), lengths),
            "hypothesis_index": np.concatenate([np.arange(L) for L in lengths]),
            "evidence": np.concatenate(
                [np.asarray(e, dtype=np.float64) for e in df["evidence"]]
            ),
        }
    )


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


def plot_episode(
    df: pd.DataFrame,
    episode: int,
    out_path: Path,
    rng: np.random.Generator,
    primary_target: str,
    sampling_ranges: list[tuple[int, int]],
    primary_target_breakdown: pd.DataFrame | None = None,
) -> None:
    pred_err = (MAX_EVIDENCE - df["evidence"].to_numpy()) / EVIDENCE_RANGE
    is_mlh = (df["hypothesis_index"].to_numpy() == df["mlh_index"].to_numpy())
    steps = df["step"].to_numpy()
    graph_ids_arr = df["graph_id"].to_numpy()

    graph_ids = sorted(np.unique(graph_ids_arr).tolist())
    cmap = plt.get_cmap("tab20", max(len(graph_ids), 1))
    color_map = {gid: cmap(i) for i, gid in enumerate(graph_ids)}

    jitter = rng.uniform(-0.3, 0.3, size=len(steps))
    x = steps + jitter

    fig, ax = plt.subplots(figsize=(12, 6))

    for gid in graph_ids:
        mask = (graph_ids_arr == gid) & ~is_mlh
        if mask.any():
            ax.scatter(
                x[mask],
                pred_err[mask],
                color=color_map[gid],
                s=6,
                alpha=0.35,
                linewidths=0,
                zorder=2,
            )

    for gid in graph_ids:
        mask = (graph_ids_arr == gid) & is_mlh
        if mask.any():
            ax.scatter(
                x[mask],
                pred_err[mask],
                color=color_map[gid],
                s=45,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )

    mean_per_step = (
        pd.Series(pred_err).groupby(steps).mean().sort_index()
    )
    ax.plot(
        mean_per_step.index,
        mean_per_step.values,
        color="black",
        linewidth=2,
        marker="o",
        markersize=4,
        zorder=6,
    )

    pt_mlh_mask = (graph_ids_arr == primary_target) & is_mlh
    has_breakdown_fill = False
    if (
        primary_target
        and primary_target_breakdown is not None
        and not primary_target_breakdown.empty
    ):
        bd = primary_target_breakdown.sort_values("step")
        bd_steps = bd["step"].to_numpy()
        pose_pt = bd["pose_evidence_mlh"].to_numpy()
        feature_pt = bd["feature_evidence_mlh"].to_numpy()
        # Range-equalized mismatch for each component (both in [0, 1]).
        pose_mismatch = (1 - pose_pt) / POSE_RANGE
        feature_mismatch = (1 - feature_pt) / FEATURE_RANGE
        denom = pose_mismatch + feature_mismatch
        pose_share = np.where(denom > 1e-9, pose_mismatch / denom, 0.5)
        total_err_pt = (MAX_EVIDENCE - (pose_pt + feature_pt)) / EVIDENCE_RANGE
        band_pose = pose_share * total_err_pt
        ax.fill_between(
            bd_steps,
            0,
            band_pose,
            color=POSE_FILL_COLOR,
            alpha=0.12,
            linewidth=0,
            zorder=0.6,
        )
        ax.fill_between(
            bd_steps,
            band_pose,
            total_err_pt,
            color=FEATURE_FILL_COLOR,
            alpha=0.12,
            linewidth=0,
            zorder=0.6,
        )
        has_breakdown_fill = True

    if primary_target and pt_mlh_mask.any():
        order = np.argsort(steps[pt_mlh_mask])
        ax.plot(
            steps[pt_mlh_mask][order],
            pred_err[pt_mlh_mask][order],
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        )

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
    ax.set_ylabel("prediction error")
    ax.set_title(f"Episode {episode} - From Step Evidence")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[gid],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            label=gid,
        )
        for gid in graph_ids
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            marker="o",
            markersize=4,
            label="mean (all hypotheses)",
        )
    )
    if primary_target and pt_mlh_mask.any():
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.5,
                linestyle="--",
                label="MLH (primary target)",
            )
        )
    if has_breakdown_fill:
        handles.append(
            Patch(
                facecolor=POSE_FILL_COLOR,
                alpha=0.12,
                label="pose share of error (primary)",
            )
        )
        handles.append(
            Patch(
                facecolor=FEATURE_FILL_COLOR,
                alpha=0.12,
                label="feature share of error (primary)",
            )
        )
    if sampling_ranges:
        handles.append(
            Line2D(
                [0],
                [0],
                color="red",
                linewidth=6,
                label="sampling steps",
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
    if primary_target:
        for text in legend.get_texts():
            if text.get_text() == primary_target:
                text.set_bbox(
                    dict(
                        boxstyle="round,pad=0.2",
                        facecolor="none",
                        edgecolor="black",
                        linewidth=1.0,
                    )
                )
                break
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found in {args.input_dir}")

    if args.num_episodes is not None and args.num_episodes < len(files):
        files = random.Random(args.seed).sample(files, args.num_episodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_files(files)
    rng = np.random.default_rng(args.seed)

    for episode, src_group in df.groupby("episode", sort=True):
        primary_target = (
            str(src_group["primary_target_graph_id"].iloc[0])
            if "primary_target_graph_id" in src_group
            else ""
        )
        if "is_sampling" in src_group:
            sampling_steps = sorted(
                {int(s) for s in src_group.loc[src_group["is_sampling"], "step"]}
            )
        else:
            sampling_steps = []
        sampling_ranges = consecutive_ranges(sampling_steps)

        long_group = explode(src_group)

        primary_target_breakdown: pd.DataFrame | None = None
        if (
            primary_target
            and {"pose_evidence_mlh", "feature_evidence_mlh"}.issubset(
                src_group.columns
            )
        ):
            pt_rows = src_group[src_group["graph_id"] == primary_target]
            if not pt_rows.empty:
                primary_target_breakdown = pt_rows[
                    ["step", "pose_evidence_mlh", "feature_evidence_mlh"]
                ].copy()

        out = args.output_dir / f"episode_{episode}.png"
        plot_episode(
            long_group,
            int(episode),
            out,
            rng,
            primary_target,
            sampling_ranges,
            primary_target_breakdown=primary_target_breakdown,
        )
        print(f"Wrote {out}  ({len(long_group)} points)")


if __name__ == "__main__":
    main()
