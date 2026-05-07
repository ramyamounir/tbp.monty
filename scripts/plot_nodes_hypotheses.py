#!/usr/bin/env python
"""Per-episode 4-subplot jitter plots comparing patch_1 vs learning_module_0
input channels of the high-level LM (learning_module_1).

Reads JSONL files written by `hypothesis_evidence_logger`, groups records by
episode, and writes one PNG per episode with four subplots:

    upper-left:  n_nodes_in_radius      (2 jitter columns / step, y-jittered)
    upper-right: nearest_distance       (2 jitter columns / step)
    lower-left:  Delta in-radius        = LM - SM (1 column / step, y-jittered)
    lower-right: Delta nearest distance = SM - LM (1 column / step)

In the upper panels, patch_1 uses a circle marker and learning_module_0 uses
a triangle marker; both are colored by graph_id.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

LM_ID = "learning_module_1"
SM_CHANNEL = "patch_1"
LM_CHANNEL = "learning_module_0"
SM_DENSITY_COLOR = "#4C72B0"
LM_DENSITY_COLOR = "#DD8452"
DIFF_DENSITY_COLOR = "#55A868"
OUT_OF_FRAME_COLOR = "#C44E52"
SHOW_OORF = False
MAX_MATCH_DISTANCE = 0.01
MAX_NNEIGHBORS = 3
THRESHOLD_COLOR = "#C44E52"

X_JITTER_DOUBLE = 0.15
X_JITTER_OFFSET = 0.20
X_JITTER_SINGLE = 0.30
Y_JITTER_INT = 0.35

SCATTER_SIZE = 6
SCATTER_ALPHA = 0.35
MARGINAL_ALPHA = 0.5
MARGINAL_BINS_CONTINUOUS = 30


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


def explode_top(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """One row per (record × hypothesis) for one channel; records missing the
    channel are dropped.
    """
    keep = df["channels"].apply(lambda c: channel in c)
    sub = df.loc[keep]
    n_arrs = sub["channels"].apply(
        lambda c: np.asarray(c[channel]["custom_n_nodes_in_radius"], dtype=np.float64)
    )
    d_arrs = sub["channels"].apply(
        lambda c: np.asarray(c[channel]["custom_nearest_distance"], dtype=np.float64)
    )
    lengths = n_arrs.apply(len).to_numpy()
    return pd.DataFrame(
        {
            "step": np.repeat(sub["step"].to_numpy(), lengths),
            "graph_id": np.repeat(sub["graph_id"].to_numpy(), lengths),
            "n_in_radius": np.concatenate(n_arrs.tolist()),
            "nearest_distance": np.concatenate(d_arrs.tolist()),
        }
    )


def explode_diff(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (record × hypothesis) for records that have both channels."""
    keep = df["channels"].apply(lambda c: SM_CHANNEL in c and LM_CHANNEL in c)
    sub = df.loc[keep]
    n_sm = sub["channels"].apply(
        lambda c: np.asarray(
            c[SM_CHANNEL]["custom_n_nodes_in_radius"], dtype=np.float64
        )
    )
    n_lm = sub["channels"].apply(
        lambda c: np.asarray(
            c[LM_CHANNEL]["custom_n_nodes_in_radius"], dtype=np.float64
        )
    )
    d_sm = sub["channels"].apply(
        lambda c: np.asarray(c[SM_CHANNEL]["custom_nearest_distance"], dtype=np.float64)
    )
    d_lm = sub["channels"].apply(
        lambda c: np.asarray(c[LM_CHANNEL]["custom_nearest_distance"], dtype=np.float64)
    )
    eu_n_sm = sub["channels"].apply(
        lambda c: np.asarray(
            c[SM_CHANNEL]["euclidean_n_nodes_in_radius"], dtype=np.int_
        )
    )
    eu_n_lm = sub["channels"].apply(
        lambda c: np.asarray(
            c[LM_CHANNEL]["euclidean_n_nodes_in_radius"], dtype=np.int_
        )
    )
    eu_d_sm = sub["channels"].apply(
        lambda c: np.asarray(
            c[SM_CHANNEL]["euclidean_nearest_distance"], dtype=np.float64
        )
    )
    eu_d_lm = sub["channels"].apply(
        lambda c: np.asarray(
            c[LM_CHANNEL]["euclidean_nearest_distance"], dtype=np.float64
        )
    )
    lengths = n_sm.apply(len).to_numpy()
    eu_n_sm_flat = np.concatenate(eu_n_sm.tolist())
    eu_n_lm_flat = np.concatenate(eu_n_lm.tolist())
    eu_d_sm_flat = np.concatenate(eu_d_sm.tolist())
    eu_d_lm_flat = np.concatenate(eu_d_lm.tolist())
    return pd.DataFrame(
        {
            "step": np.repeat(sub["step"].to_numpy(), lengths),
            "graph_id": np.repeat(sub["graph_id"].to_numpy(), lengths),
            "diff_radius": np.concatenate(
                [lm - sm for lm, sm in zip(n_lm.tolist(), n_sm.tolist())]
            ),
            "diff_distance": np.concatenate(
                [sm - lm for lm, sm in zip(d_lm.tolist(), d_sm.tolist())]
            ),
            "oof_radius": (eu_n_sm_flat == 0) & (eu_n_lm_flat == 0),
            "oof_distance": (eu_d_sm_flat > MAX_MATCH_DISTANCE)
            & (eu_d_lm_flat > MAX_MATCH_DISTANCE),
        }
    )


def _scatter_top(
    ax,
    top_df: pd.DataFrame,
    value_col: str,
    color: str,
    rng: np.random.Generator,
    sign: int,
    unique_steps: np.ndarray,
    y_jitter: float = 0.0,
) -> None:
    if top_df.empty:
        return
    step_idx = np.searchsorted(unique_steps, top_df["step"].to_numpy())
    values = top_df[value_col].to_numpy()
    x = (
        step_idx
        + sign * X_JITTER_OFFSET
        + rng.uniform(-X_JITTER_DOUBLE, X_JITTER_DOUBLE, size=len(step_idx))
    )
    y = values
    if y_jitter > 0:
        y = y + rng.uniform(-y_jitter, y_jitter, size=len(values))
    ax.scatter(
        x,
        y,
        color=color,
        marker="o",
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        zorder=2,
    )


def _scatter_diff(
    ax,
    diff_df: pd.DataFrame,
    value_col: str,
    color: str,
    rng: np.random.Generator,
    unique_steps: np.ndarray,
    y_jitter: float = 0.0,
    highlight_col: str | None = None,
    highlight_color: str = OUT_OF_FRAME_COLOR,
) -> None:
    if diff_df.empty:
        return
    if not SHOW_OORF and highlight_col is not None and highlight_col in diff_df.columns:
        diff_df = diff_df.loc[~diff_df[highlight_col].astype(bool)]
        if diff_df.empty:
            return
    step_idx = np.searchsorted(unique_steps, diff_df["step"].to_numpy())
    values = diff_df[value_col].to_numpy()
    x = step_idx + rng.uniform(-X_JITTER_SINGLE, X_JITTER_SINGLE, size=len(step_idx))
    y = values
    if y_jitter > 0:
        y = y + rng.uniform(-y_jitter, y_jitter, size=len(values))
    if SHOW_OORF and highlight_col is not None and highlight_col in diff_df.columns:
        highlight = diff_df[highlight_col].to_numpy().astype(bool)
    else:
        highlight = np.zeros(len(values), dtype=bool)
    ax.scatter(
        x[~highlight],
        y[~highlight],
        color=color,
        marker="o",
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        x[highlight],
        y[highlight],
        color=highlight_color,
        marker="o",
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        zorder=3,
    )


def _add_marginal(ax):
    divider = make_axes_locatable(ax)
    ax_marg = divider.append_axes("right", size="15%", pad=0.1, sharey=ax)
    ax_marg.tick_params(axis="y", labelleft=False, length=0)
    ax_marg.tick_params(axis="x", labelbottom=False, length=0)
    for spine in ("top", "right", "bottom"):
        ax_marg.spines[spine].set_visible(False)
    return ax_marg


def _integer_bins(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([-0.5, 0.5])
    mn, mx = int(np.floor(values.min())), int(np.ceil(values.max()))
    return np.arange(mn - 0.5, mx + 1.5, 1)


def _plot_top_marginal(
    ax_marg, sm_values: np.ndarray, lm_values: np.ndarray, integer: bool
) -> None:
    if integer:
        all_vals = (
            np.concatenate([sm_values, lm_values])
            if sm_values.size or lm_values.size
            else np.array([])
        )
        bins = _integer_bins(all_vals)
    else:
        bins = MARGINAL_BINS_CONTINUOUS
    if sm_values.size:
        ax_marg.hist(
            sm_values,
            bins=bins,
            orientation="horizontal",
            histtype="stepfilled",
            color=SM_DENSITY_COLOR,
            alpha=MARGINAL_ALPHA,
            density=True,
            linewidth=0,
        )
    if lm_values.size:
        ax_marg.hist(
            lm_values,
            bins=bins,
            orientation="horizontal",
            histtype="stepfilled",
            color=LM_DENSITY_COLOR,
            alpha=MARGINAL_ALPHA,
            density=True,
            linewidth=0,
        )


def _add_density_legend(ax_marg) -> None:
    """Vertical 'denser LM ↑ / denser SM ↓' legend right of the marginal axis."""
    arrow_x = 1.6
    label_x = 1.85
    ax_marg.annotate(
        "",
        xy=(arrow_x, 0.95),
        xytext=(arrow_x, 0.55),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color=DIFF_DENSITY_COLOR,
            lw=1.8,
            mutation_scale=18,
        ),
        annotation_clip=False,
    )
    ax_marg.text(
        label_x,
        0.95,
        f"denser {LM_CHANNEL}",
        transform=ax_marg.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        color=DIFF_DENSITY_COLOR,
        fontweight="bold",
    )
    ax_marg.annotate(
        "",
        xy=(arrow_x, 0.05),
        xytext=(arrow_x, 0.45),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color=DIFF_DENSITY_COLOR,
            lw=1.8,
            mutation_scale=18,
        ),
        annotation_clip=False,
    )
    ax_marg.text(
        label_x,
        0.05,
        f"denser {SM_CHANNEL}",
        transform=ax_marg.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        color=DIFF_DENSITY_COLOR,
        fontweight="bold",
    )
    if SHOW_OORF:
        frame_legend_handles = [
            Patch(facecolor=DIFF_DENSITY_COLOR, alpha=MARGINAL_ALPHA, linewidth=0),
            Patch(facecolor=OUT_OF_FRAME_COLOR, alpha=MARGINAL_ALPHA, linewidth=0),
        ]
        frame_legend_labels = [
            "Inside Reference Frame",
            "Out of Reference Frame",
        ]
        ax_marg.legend(
            handles=frame_legend_handles,
            labels=frame_legend_labels,
            loc="center left",
            bbox_to_anchor=(1.55, 0.5),
            fontsize=8,
            framealpha=0.9,
            borderaxespad=0.0,
            handlelength=1.2,
            handleheight=1.0,
        )


def _add_pretrained_graph_inset(
    fig, primary_target: str, graph_nodes: dict[str, np.ndarray], legend
) -> None:
    sm_key = f"{LM_ID}__{primary_target}__{SM_CHANNEL}"
    lm_key = f"{LM_ID}__{primary_target}__{LM_CHANNEL}"
    available = [k for k in (sm_key, lm_key) if k in graph_nodes]
    if not available:
        return
    fig.canvas.draw()
    leg_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    fig_w, fig_h = fig.get_size_inches()
    inset_left = leg_bbox.x0
    inset_width = max(leg_bbox.width, 0.13)
    inset_height = inset_width * fig_w / fig_h
    inset_top = leg_bbox.y0 - 0.05
    inset_bottom = inset_top - inset_height
    ax3d = fig.add_axes(
        [inset_left, inset_bottom, inset_width, inset_height], projection="3d"
    )
    if sm_key in graph_nodes:
        sm_pts = graph_nodes[sm_key]
        ax3d.scatter(
            sm_pts[:, 0],
            sm_pts[:, 1],
            sm_pts[:, 2],
            c=SM_DENSITY_COLOR,
            s=4,
            alpha=0.6,
            edgecolors="none",
        )
    if lm_key in graph_nodes:
        lm_pts = graph_nodes[lm_key]
        ax3d.scatter(
            lm_pts[:, 0],
            lm_pts[:, 1],
            lm_pts[:, 2],
            c=LM_DENSITY_COLOR,
            s=4,
            alpha=0.6,
            edgecolors="none",
        )
    all_pts = np.concatenate([graph_nodes[k] for k in available])
    mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
    half = (mx - mn).max() / 2
    center = (mx + mn) / 2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_axis_off()
    ax3d.view_init(elev=30, azim=45)
    ax3d.set_title(primary_target, fontsize=8, pad=0)


def _plot_diff_marginal(
    ax_marg,
    values: np.ndarray,
    integer: bool,
    oof_mask: np.ndarray | None = None,
) -> None:
    if values.size == 0:
        return
    if not SHOW_OORF and oof_mask is not None:
        values = values[~oof_mask.astype(bool)]
        if values.size == 0:
            return
        oof_mask = None
    if oof_mask is None:
        oof_mask = np.zeros(len(values), dtype=bool)
    in_frame_vals = values[~oof_mask]
    oof_vals = values[oof_mask]
    if integer:
        bins = _integer_bins(values)
    else:
        bins = MARGINAL_BINS_CONTINUOUS
    if in_frame_vals.size:
        ax_marg.hist(
            in_frame_vals,
            bins=bins,
            orientation="horizontal",
            histtype="stepfilled",
            color=DIFF_DENSITY_COLOR,
            alpha=MARGINAL_ALPHA,
            density=True,
            linewidth=0,
        )
    if oof_vals.size:
        ax_marg.hist(
            oof_vals,
            bins=bins,
            orientation="horizontal",
            histtype="stepfilled",
            color=OUT_OF_FRAME_COLOR,
            alpha=MARGINAL_ALPHA,
            density=True,
            linewidth=0,
        )
    ax_marg.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=3)


def plot_episode(
    src_group: pd.DataFrame,
    episode: int,
    out_path: Path,
    rng: np.random.Generator,
    primary_target: str,
    graph_nodes: dict[str, np.ndarray],
) -> None:
    if not primary_target:
        return
    src_group = src_group[src_group["graph_id"] == primary_target]
    if src_group.empty:
        return

    sm_top = explode_top(src_group, SM_CHANNEL)
    lm_top = explode_top(src_group, LM_CHANNEL)
    diff = explode_diff(src_group)

    step_arrays = []
    for src in (sm_top, lm_top, diff):
        if not src.empty:
            step_arrays.append(src["step"].to_numpy())
    unique_steps = (
        np.unique(np.concatenate(step_arrays))
        if step_arrays
        else np.array([], dtype=np.int64)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "wspace": 0.22},
    )

    ax = axes[0, 0]
    _scatter_top(
        ax,
        sm_top,
        "n_in_radius",
        SM_DENSITY_COLOR,
        rng,
        sign=-1,
        unique_steps=unique_steps,
        y_jitter=Y_JITTER_INT,
    )
    _scatter_top(
        ax,
        lm_top,
        "n_in_radius",
        LM_DENSITY_COLOR,
        rng,
        sign=+1,
        unique_steps=unique_steps,
        y_jitter=Y_JITTER_INT,
    )
    ax.set_ylabel("In Radius Count")
    max_int = int(
        max(
            sm_top["n_in_radius"].max() if not sm_top.empty else 0,
            lm_top["n_in_radius"].max() if not lm_top.empty else 0,
            MAX_NNEIGHBORS,
        )
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ymax = max_int + 0.5
    ax.set_ylim(-0.5, ymax)
    ax.axhspan(
        MAX_NNEIGHBORS + 0.5,
        ymax,
        color=THRESHOLD_COLOR,
        alpha=0.18,
        linewidth=0,
        zorder=0,
    )
    ax.axhline(
        MAX_NNEIGHBORS + 0.5,
        color=THRESHOLD_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
        zorder=1,
    )
    _plot_top_marginal(
        _add_marginal(ax),
        sm_top["n_in_radius"].to_numpy() if not sm_top.empty else np.array([]),
        lm_top["n_in_radius"].to_numpy() if not lm_top.empty else np.array([]),
        integer=True,
    )

    ax = axes[0, 1]
    _scatter_top(
        ax,
        sm_top,
        "nearest_distance",
        SM_DENSITY_COLOR,
        rng,
        sign=-1,
        unique_steps=unique_steps,
    )
    _scatter_top(
        ax,
        lm_top,
        "nearest_distance",
        LM_DENSITY_COLOR,
        rng,
        sign=+1,
        unique_steps=unique_steps,
    )
    ax.set_ylabel("Nearest Distance")
    ax.set_ylim(bottom=0)
    ymax = max(ax.get_ylim()[1], MAX_MATCH_DISTANCE * 1.1)
    ax.set_ylim(0, ymax)
    ax.axhspan(
        MAX_MATCH_DISTANCE,
        ymax,
        color=THRESHOLD_COLOR,
        alpha=0.18,
        linewidth=0,
        zorder=0,
    )
    ax.axhline(
        MAX_MATCH_DISTANCE,
        color=THRESHOLD_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
        zorder=1,
    )
    ax_marg_01 = _add_marginal(ax)
    _plot_top_marginal(
        ax_marg_01,
        sm_top["nearest_distance"].to_numpy() if not sm_top.empty else np.array([]),
        lm_top["nearest_distance"].to_numpy() if not lm_top.empty else np.array([]),
        integer=False,
    )

    ax = axes[1, 0]
    _scatter_diff(
        ax,
        diff,
        "diff_radius",
        DIFF_DENSITY_COLOR,
        rng,
        unique_steps=unique_steps,
        y_jitter=Y_JITTER_INT,
        highlight_col="oof_radius",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=3)
    ax.set_ylabel(f"Δ In Radius ({LM_CHANNEL} − {SM_CHANNEL})")
    ax.set_xlabel("step")
    if not diff.empty:
        vmax = float(np.abs(diff["diff_radius"].to_numpy()).max())
        if vmax > 0:
            ax.set_ylim(-vmax * 1.05, vmax * 1.05)
    _plot_diff_marginal(
        _add_marginal(ax),
        diff["diff_radius"].to_numpy() if not diff.empty else np.array([]),
        integer=True,
        oof_mask=(
            diff["oof_radius"].to_numpy().astype(bool) if not diff.empty else None
        ),
    )

    ax = axes[1, 1]
    _scatter_diff(
        ax,
        diff,
        "diff_distance",
        DIFF_DENSITY_COLOR,
        rng,
        unique_steps=unique_steps,
        highlight_col="oof_distance",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=3)
    ax.set_ylabel(f"Δ Distance ({SM_CHANNEL} − {LM_CHANNEL})")
    ax.set_xlabel("step")
    if not diff.empty:
        vmax = float(np.abs(diff["diff_distance"].to_numpy()).max())
        if vmax > 0:
            ax.set_ylim(-vmax * 1.05, vmax * 1.05)
    ax_marg_11 = _add_marginal(ax)
    _plot_diff_marginal(
        ax_marg_11,
        diff["diff_distance"].to_numpy() if not diff.empty else np.array([]),
        integer=False,
        oof_mask=(
            diff["oof_distance"].to_numpy().astype(bool) if not diff.empty else None
        ),
    )
    _add_density_legend(ax_marg_11)

    n_steps = len(unique_steps)
    if n_steps > 0:
        rotation = 90 if n_steps > 20 else 0
        for ax in axes.flat:
            ax.set_xlim(-0.5, n_steps - 0.5)
            ax.set_xticks(range(n_steps))
        for ax in axes[1, :]:
            ax.set_xticklabels(
                [str(int(s)) for s in unique_steps],
                rotation=rotation,
                fontsize=8,
            )

    fig.suptitle(f"Episode {episode} - {primary_target}")

    def _channel_handle(color):
        return (
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=8,
            ),
            Patch(facecolor=color, alpha=MARGINAL_ALPHA, linewidth=0),
        )

    handles = [
        _channel_handle(SM_DENSITY_COLOR),
        _channel_handle(LM_DENSITY_COLOR),
        _channel_handle(DIFF_DENSITY_COLOR),
    ]
    labels = [SM_CHANNEL, LM_CHANNEL, "channel diff"]
    legend = ax_marg_01.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        fontsize=9,
        framealpha=0.9,
        borderaxespad=0.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        handlelength=2.5,
    )

    _add_pretrained_graph_inset(fig, primary_target, graph_nodes, legend)

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

    graph_nodes_path = args.input_dir / "graph_nodes.npz"
    if graph_nodes_path.exists():
        with np.load(graph_nodes_path) as npz:
            graph_nodes = {k: npz[k] for k in npz.files}
        print(
            f"Loaded {len(graph_nodes)} (graph_id, channel) entries from "
            f"{graph_nodes_path}"
        )
    else:
        graph_nodes = {}
        print(f"No graph_nodes.npz at {graph_nodes_path} — skipping 3D inset.")

    for episode, src_group in df.groupby("episode", sort=True):
        primary_target = (
            str(src_group["primary_target_graph_id"].iloc[0])
            if "primary_target_graph_id" in src_group
            else ""
        )
        out = output_dir / f"episode_{episode}.png"
        plot_episode(src_group, int(episode), out, rng, primary_target, graph_nodes)
        print(f"Wrote {out}  ({len(src_group)} records)")


if __name__ == "__main__":
    main()
