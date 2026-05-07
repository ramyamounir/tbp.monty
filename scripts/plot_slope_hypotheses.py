#!/usr/bin/env python
"""Per-episode prediction-error scatter plots from hypothesis-slope JSONL.

Reads JSONL files written by `hypothesis_evidence_logger` in tracker mode,
computes prediction error per hypothesis from slope values (finite only),
and writes one PNG per episode where every dot is one hypothesis. Dots are
jittered horizontally and colored by graph_id; the per-(step, graph_id)
hypothesis with the lowest prediction error is rendered on top with a
larger black-edged marker, standing in for the missing MLH index.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

MAX_EVIDENCE = 2.0
EVIDENCE_RANGE = 3.0  # MAX_EVIDENCE - MIN_EVIDENCE  (MIN_EVIDENCE = -1)


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
    lengths = df["slopes"].apply(len).to_numpy()
    return pd.DataFrame(
        {
            "episode": np.repeat(df["episode"].to_numpy(), lengths),
            "step": np.repeat(df["step"].to_numpy(), lengths),
            "graph_id": np.repeat(df["graph_id"].to_numpy(), lengths),
            "hypothesis_index": np.concatenate([np.arange(L) for L in lengths]),
            "slope": np.concatenate(
                [np.asarray(s, dtype=np.float64) for s in df["slopes"]]
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
) -> None:
    pred_err = (MAX_EVIDENCE - df["slope"].to_numpy()) / EVIDENCE_RANGE
    steps = df["step"].to_numpy()
    graph_ids_arr = df["graph_id"].to_numpy()

    mlh_indices = (
        pd.DataFrame(
            {"pred_err": pred_err, "step": steps, "graph_id": graph_ids_arr}
        )
        .groupby(["step", "graph_id"])["pred_err"]
        .idxmin()
        .to_numpy()
    )
    is_mlh = np.zeros(len(df), dtype=bool)
    is_mlh[mlh_indices] = True

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
    ax.set_title(f"Episode {episode} - From slope")
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
        long_group = long_group[
            np.isfinite(long_group["slope"].to_numpy())
        ].reset_index(drop=True)
        if long_group.empty:
            print(f"Skipped episode {episode} (no finite slopes)")
            continue
        out = args.output_dir / f"episode_{episode}.png"
        plot_episode(
            long_group,
            int(episode),
            out,
            rng,
            primary_target,
            sampling_ranges,
        )
        print(f"Wrote {out}  ({len(long_group)} points)")


if __name__ == "__main__":
    main()
