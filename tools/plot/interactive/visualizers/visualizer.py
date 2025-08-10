# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Protocol

from pubsub.core import Publisher


class Visualizer(Protocol):
    def __init__(self, name: str, event_bus: Publisher, publishers: list[str]): ...

    def subscribers(self): ...

    def publishers(self): ...

    def subscribe_to_topics(self): ...

    def construct_publishers(self): ...


def validate_subscriber_requirements(visualizers: list[Visualizer]):
    # Collect all published and required subscriber topics
    all_published = set()
    all_required = set()

    for v in visualizers:
        all_published.update(v.publishers)
        if hasattr(v, "subscribers"):
            all_required.update(v.subscribers)

    # Check for missing publishers
    missing_publishers = [name for name in all_required if name not in all_published]
    if missing_publishers:
        raise RuntimeError(f"Missing publishers for topics: {missing_publishers}")

    # Warn about unused publishers
    unused_publishers = [name for name in all_published if name not in all_required]
    if unused_publishers:
        print(
            f"Warning: These publisher topics have no subscribers: {unused_publishers}"
        )

    print("All required subscriber topics are satisfied by publishers.")
