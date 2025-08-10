# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from pubsub.core import Publisher
from vedo import (
    Plotter,
    Text2D,
)

from tools.plot.interactive.data import (
    DataLocator,
    DataLocatorItem,
    DataParser,
    YCBMeshLoader,
)
from tools.plot.interactive.event import MeshTopicMessage, Topic
from tools.plot.interactive.text import create_stats_text


class PrimaryMeshVisualizer:
    def __init__(
        self,
        name: str,
        plotter: Plotter,
        parser: DataParser,
        mesh_loader: YCBMeshLoader,
        event_bus: Publisher,
        topic_controllers: list[str],
    ):
        self.name = name
        self.plotter = plotter
        self.parser = parser
        self.mesh_loader = mesh_loader
        self.event_bus = event_bus
        self._topic_controllers = topic_controllers

        # Define data locator
        self._locators = self.initialize_data_locators()

        # Define list of topics
        self._topics: dict[str, Topic] = {
            "primary_target": Topic(
                subscriber=self.on_primary_target_message,
                publisher=self.add_primary_target_widgets,
                value=None,
            )
        }
        self.initialize_topics()

        # Define visualizer widget states
        self.current_object_mesh = None
        self.current_stats_object = None

    @property
    def subscribers(self):
        return list(self._topics)

    @property
    def publishers(self):
        return self._topic_controllers

    def initialize_topics(self):
        for topic_name, topic in self._topics.items():
            self.event_bus.subscribe(topic.subscriber, topic_name)
            if topic_name in self._topic_controllers:
                topic.publisher()

    def initialize_data_locators(self):
        primary_target_locator = DataLocator(
            path=[
                DataLocatorItem(name="episode", type="key"),
                DataLocatorItem(name="system", type="key", value="LM_0"),
                DataLocatorItem(name="telemetry", type="key"),
            ],
        )
        return {"primary_target": primary_target_locator}

    # === Subscriber Callbacks === #
    def on_primary_target_message(self, message):
        topic = self._topics["primary_target"]

        if topic.value != message:
            # Add Mesh object
            if self.current_object_mesh is not None:
                self.plotter.remove(self.current_object_mesh)
                self.current_object_mesh = None

            self.current_object_mesh = self.mesh_loader.create_mesh(
                message.object_name
            ).clone(deep=True)
            self.current_object_mesh.rotate_x(message.object_rotation[0])
            self.current_object_mesh.rotate_y(message.object_rotation[1])
            self.current_object_mesh.rotate_z(message.object_rotation[2])
            self.current_object_mesh.shift(*message.object_position)
            self.plotter.add(self.current_object_mesh)

            # Add Text object
            if self.current_stats_object is not None:
                self.plotter.remove(self.current_stats_object)
                self.current_stats_object = None

            stats_text = create_stats_text(
                {
                    "Primary Target Info": [
                        {"Target Object": message.object_name},
                        {"Position": message.object_position},
                        {"Rotation": message.object_rotation},
                    ],
                }
            )
            self.current_stats_object = Text2D(stats_text, pos="top-left", s=0.7)
            self.plotter.add(self.current_stats_object)

            # Update Topic Value
            topic.value = message

    # === Publishers Constructors === #
    def add_primary_target_widgets(self):
        self.episode_number_slider = self.plotter.add_slider(
            self.callback_episode_number,
            xmin=0,
            xmax=len(self.parser.query(self._locators["primary_target"])) - 1,
            value=0,
            pos=[(0.2, 0.05), (0.8, 0.05)],
            title="Episode",
        )

    def callback_episode_number(self, widget, event):
        value = round(widget.GetRepresentation().GetValue())

        primary_target = self.parser.extract(
            self._locators["primary_target"], episode=str(value), telemetry="target"
        )
        message = MeshTopicMessage(
            object_name=primary_target["object"],
            object_rotation=primary_target["euler_rotation"],
            object_position=primary_target["position"],
        )

        self.event_bus.sendMessage("primary_target", message=message)

    # === Other Utils === #
    def axes_dict(self) -> dict[str, tuple[float, float]]:
        """Returns axis ranges.

        Note:
            Monty translates the object by 1.5 in the y-direction.
        """
        return {
            "xrange": (-0.05, 0.05),
            "yrange": (1.45, 1.55),
            "zrange": (-0.05, 0.05),
        }
