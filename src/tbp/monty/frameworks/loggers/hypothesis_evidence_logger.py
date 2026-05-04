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
from typing import IO, Literal

import numpy as np
import numpy.typing as npt

_BASE_DIR = Path("~/tbp/tbp.monty/pe").expanduser()
_FILENAME = "hypothesis_evidence.jsonl"

LOG_MODE: Literal["displacer", "tracker"] = "displacer"

_episode: int = 0
_step: int = 0
_primary_target_graph_id: str = ""
_file: IO[str] | None = None


def set_episode(episode: int) -> None:
    global _episode  # noqa: PLW0603
    _episode = int(episode)


def set_step(step: int) -> None:
    global _step  # noqa: PLW0603
    _step = int(step)


def set_primary_target_graph_id(graph_id: str) -> None:
    global _primary_target_graph_id  # noqa: PLW0603
    _primary_target_graph_id = str(graph_id)


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


def log(
    graph_id: str,
    mlh_index: int,
    evidence: npt.NDArray[np.float64],
    pose_evidence_mlh: float,
    feature_evidence_mlh: float,
    is_sampling: bool,
) -> None:
    if LOG_MODE != "displacer":
        return
    f = _ensure_file_open()
    f.write(
        json.dumps(
            {
                "episode": _episode,
                "step": _step,
                "graph_id": graph_id,
                "mlh_index": int(mlh_index),
                "evidence": evidence.tolist(),
                "pose_evidence_mlh": float(pose_evidence_mlh),
                "feature_evidence_mlh": float(feature_evidence_mlh),
                "is_sampling": bool(is_sampling),
                "primary_target_graph_id": _primary_target_graph_id,
            }
        )
        + "\n"
    )


def log_tracker(
    graph_id: str,
    slopes: npt.NDArray[np.float64],
    ages: npt.NDArray[np.int_],
    is_sampling: bool,
) -> None:
    if LOG_MODE != "tracker":
        return
    f = _ensure_file_open()
    f.write(
        json.dumps(
            {
                "episode": _episode,
                "step": _step,
                "graph_id": graph_id,
                "slopes": slopes.tolist(),
                "ages": ages.tolist(),
                "is_sampling": bool(is_sampling),
                "primary_target_graph_id": _primary_target_graph_id,
            }
        )
        + "\n"
    )
