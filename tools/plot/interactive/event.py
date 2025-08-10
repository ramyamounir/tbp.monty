# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable


@dataclass
class Topic:
    subscriber: Callable
    publisher: Callable
    value: Any = None


@dataclass
class MeshTopicMessage:
    object_name: str
    object_rotation: list[float, float, float]
    object_position: list[float, float, float]

    def __eq__(self, other):
        if not isinstance(other, MeshTopicMessage):
            return False

        for field in fields(self):
            if getattr(self, field.name) != getattr(other, field.name):
                return False
        return True
