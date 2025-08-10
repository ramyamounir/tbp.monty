# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from vedo import Mesh

from tbp.monty.frameworks.utils.logging_utils import load_stats


class YCBMeshLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _find_glb_file(self, obj_name: str) -> str:
        """Search for the .glb.orig file of a given YCB object in a directory.

        Args:
            obj_name: The object name to search for (e.g., "potted_meat_can").

        Returns:
            Full path to the .glb.orig file.

        Raises:
            FileNotFoundError: If the .glb.orig file for the object is not found.

        """
        for path in Path(self.data_path).rglob("*"):
            if path.is_dir() and path.name.endswith(obj_name):
                glb_orig_path = path / "google_16k" / "textured.glb.orig"
                if glb_orig_path.exists():
                    return str(glb_orig_path)

        raise FileNotFoundError(
            f"Could not find .glb.orig file for '{obj_name}' in '{self.data_path}'"
        )

    def create_mesh(self, obj_name: str) -> Mesh:
        """Reads a 3D object file in glb format and returns a Vedo Mesh object.

        Args:
            obj_name: Name of the object to load.

        Returns:
            vedo.Mesh object with UV texture and transformed orientation.
        """
        file_path = self._find_glb_file(obj_name)
        with open(file_path, "rb") as f:
            mesh = trimesh.load_mesh(f, file_type="glb")

        # create mesh from vertices and faces
        obj = Mesh([mesh.vertices, mesh.faces])

        # add texture
        obj.texture(
            tname=np.array(mesh.visual.material.baseColorTexture),
            tcoords=mesh.visual.uv,
        )

        # Shift to geometry mean and rotate to the up/front of the glb
        obj.shift(-np.mean(obj.bounds().reshape(3, 2), axis=1))
        obj.rotate_x(-90)

        return obj


class DataParser:
    def __init__(
        self,
        exp_path: str,
    ):
        self.exp_path = exp_path

        _, _, self.data, _ = load_stats(exp_path, False, False, True, False)

    def extract(self, locator, **kwargs):
        # Check if all missing steps are provided in kwargs
        for step in locator.missing_steps():
            if step.name not in kwargs:
                raise ValueError(f"Missing required value for step: {step.name}")

            if (step.type == "index" and not isinstance(kwargs[step.name], int)) or (
                step.type == "key" and not isinstance(kwargs[step.name], str)
            ):
                raise ValueError(
                    f"Provided path step value does not match step type for step: {step.name}"
                )

        curr = self.data
        for step in locator.path:
            access_value = kwargs.get(step.name, step.value)
            curr = curr[access_value]
        return curr

    def query(self, locator, **kwargs) -> dict:
        curr = self.data
        for step in locator.path:
            access_value = kwargs.get(step.name, step.value)
            if access_value is None:
                if step.type == "index":
                    return list(range(len(curr)))  # For list-type steps, return indices
                elif step.type == "key":
                    return list(curr.keys())  # For dict-type steps, return keys

            curr = curr[access_value]

        raise ValueError("No missing values to query")


@dataclass
class DataLocatorItem:
    name: str  # descriptive name, e.g., "session_index"
    type: Literal["key", "index"]  # access type: dict key or list index
    value: str | int = None  # optional fixed value


@dataclass
class DataLocator:
    path: list[DataLocatorItem]  # steps required to reach it

    def missing_steps(self) -> list[DataLocatorItem]:
        """Returns a list of steps that do not have values."""
        return [step for step in self.path if step.value is None]

    def __repr__(self) -> str:
        steps = " -> ".join(
            f"[{step.name}]" if step.type == "index" else f".{step.name}"
            for step in self.path
        )
        return f"Path: root{steps}"
