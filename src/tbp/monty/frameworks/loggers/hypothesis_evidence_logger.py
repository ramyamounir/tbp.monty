# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import IO

import numpy as np
import numpy.typing as npt

# Include-only filters. An empty set means "no filter" (include everything).
# Edit these in source to scope what gets logged.
INCLUDE_LEARNING_MODULES: set[str] = {"learning_module_1"}
INCLUDE_GRAPH_IDS: set[str] = set()
INCLUDE_EPISODES: set[int] = set()
INCLUDE_STEPS: set[int] = set()

_BASE_DIR = Path("~/tbp/tbp.monty/pe").expanduser()
_FILENAME = "hypothesis_evidence.jsonl"

_episode: int = 0
_step: int = 0
_learning_module_id: str = ""
_primary_target_graph_id: str = ""
_file: IO[str] | None = None


def set_episode(episode: int) -> None:
    global _episode  # noqa: PLW0603
    _episode = int(episode)


def set_step(step: int) -> None:
    global _step  # noqa: PLW0603
    _step = int(step)


def set_learning_module_id(learning_module_id: str) -> None:
    global _learning_module_id  # noqa: PLW0603
    _learning_module_id = str(learning_module_id)


def set_primary_target_graph_id(graph_id: str) -> None:
    global _primary_target_graph_id  # noqa: PLW0603
    _primary_target_graph_id = str(graph_id)


def _passes_filters(graph_id: str) -> bool:
    if INCLUDE_LEARNING_MODULES and _learning_module_id not in INCLUDE_LEARNING_MODULES:
        return False
    if INCLUDE_GRAPH_IDS and graph_id not in INCLUDE_GRAPH_IDS:
        return False
    if INCLUDE_EPISODES and _episode not in INCLUDE_EPISODES:
        return False
    if INCLUDE_STEPS and _step not in INCLUDE_STEPS:
        return False
    return True


def _ensure_file_open() -> IO[str]:
    global _file  # noqa: PLW0603
    if _file is None:
        override_file = os.environ.get("HYPOTHESIS_EVIDENCE_LOG")
        override_dir = os.environ.get("HYPOTHESIS_EVIDENCE_LOG_DIR")
        if override_file:
            path = Path(override_file).expanduser()
        elif override_dir:
            path = Path(override_dir).expanduser() / f"worker_{os.getpid()}.jsonl"
        else:
            run_dir = _BASE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
            path = run_dir / _FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _file = path.open("a", buffering=1)
    return _file


def dump_graphs(model) -> None:
    """Dump per-(lm_id, graph_id, channel) node positions to graph_nodes.npz.

    Walks the Monty model's learning modules and writes every loaded graph's
    node positions to a sibling `graph_nodes.npz` next to the JSONL log.
    Keys are formatted `f"{lm_id}__{graph_id}__{channel}"`. Respects
    `INCLUDE_LEARNING_MODULES`. Skips if the file already exists.

    Call once at experiment setup (after graphs are loaded into LMs, before
    any episodes run).
    """
    f = _ensure_file_open()
    out_path = Path(f.name).parent / "graph_nodes.npz"
    if out_path.exists():
        return
    arrays: dict[str, npt.NDArray[np.float64]] = {}
    for lm in model.learning_modules:
        lm_id = lm.learning_module_id
        if INCLUDE_LEARNING_MODULES and lm_id not in INCLUDE_LEARNING_MODULES:
            continue
        graph_memory = lm.graph_memory
        for graph_id in graph_memory.get_memory_ids():
            for input_channel in graph_memory.get_input_channels_in_graph(graph_id):
                locs = np.asarray(
                    graph_memory.get_locations_in_graph(graph_id, input_channel),
                    dtype=np.float64,
                )
                arrays[f"{lm_id}__{graph_id}__{input_channel}"] = locs
    if not arrays:
        return
    np.savez_compressed(out_path, **arrays)


def log(
    graph_id: str,
    mlh_index: int,
    hyp_idxs_tested: npt.NDArray[np.int_],
    per_channel: dict[str, dict[str, npt.NDArray]],
    is_sampling: bool,
) -> None:
    """Append one JSONL record describing the displacer output for one graph.

    Args:
        graph_id: Graph being displaced.
        mlh_index: Index of the most-likely hypothesis in the full hypothesis space.
        hyp_idxs_tested: Indices (into the full hypothesis space) of hypotheses that
            had evidence above the update threshold and were re-evaluated this step.
        per_channel: Mapping `channel -> {"evidence", "pose_evidence",
            "feature_evidence", "euclidean_n_nodes_in_radius",
            "euclidean_nearest_distance", "custom_n_nodes_in_radius",
            "custom_nearest_distance"}`. `evidence` has shape (H,) (full
            hypothesis space, with min-update fill on untested hyps).
            `pose_evidence` (range [-1, 1]) and `feature_evidence` (range
            [0, feature_evidence_increment]) have shape (H_tested,), aligned
            with `hyp_idxs_tested`, and sum to the tested entries of `evidence`.
            The four diagnostic stats also have shape (H_tested,).
        is_sampling: Whether new hypotheses were sampled this step for this graph.
    """
    if not _passes_filters(graph_id):
        return
    f = _ensure_file_open()
    f.write(
        json.dumps(
            {
                "episode": _episode,
                "step": _step,
                "learning_module_id": _learning_module_id,
                "graph_id": graph_id,
                "primary_target_graph_id": _primary_target_graph_id,
                "mlh_index": int(mlh_index),
                "is_sampling": bool(is_sampling),
                "hyp_idxs_tested": hyp_idxs_tested.tolist(),
                "channels": {
                    channel: {
                        "evidence": values["evidence"].tolist(),
                        "pose_evidence": values["pose_evidence"].tolist(),
                        "feature_evidence": values["feature_evidence"].tolist(),
                        "euclidean_n_nodes_in_radius": values[
                            "euclidean_n_nodes_in_radius"
                        ].tolist(),
                        "euclidean_nearest_distance": values[
                            "euclidean_nearest_distance"
                        ].tolist(),
                        "custom_n_nodes_in_radius": values[
                            "custom_n_nodes_in_radius"
                        ].tolist(),
                        "custom_nearest_distance": values[
                            "custom_nearest_distance"
                        ].tolist(),
                    }
                    for channel, values in per_channel.items()
                },
            }
        )
        + "\n"
    )
