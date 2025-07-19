# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any, Dict

from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.utils.logging_utils import compute_pose_error


class TheoreticalLimitLMLoggingMixin:
    """Mixin that adds theoretical limit and pose error logging for learning modules.

    This mixin augments the learning module with methods to compute and log:
      - The maximum evidence score for each object.
      - The theoretical lower bound of pose error on the target object, assuming
        Monty had selected the best possible hypothesis (oracle performance).
      - The actual pose error of the most likely hypothesis (MLH) on the target object.

    These metrics are useful for analyzing the performance gap between the model's
    current inference and its best achievable potential given its internal hypotheses.

    Compatible with:
        - EvidenceGraphLM
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure the mixin is used only with compatible learning modules.

        Raises:
            TypeError: If the mixin is used with a non-compatible learning module.
        """
        super().__init_subclass__(**kwargs)
        if not any(issubclass(b, (EvidenceGraphLM)) for b in cls.__bases__):
            raise TypeError(
                "TheoreticalLimitLMLoggingMixin must be mixed in with a subclass of "
                f"EvidenceGraphLM, got {cls.__bases__}"
            )

    def _add_detailed_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Add detailed statistics to the logging dictionary.

        This includes metrics like the max evidence score per object, the theoretical
        limit of Monty (i.e., pose error of Monty's best potential hypothesis on the
        target object) , and the pose error of the MLH hypothesis on the target object.

        Args:
            stats: The existing statistics dictionary to augment.

        Returns:
            Updated statistics dictionary.
        """
        stats["resampling_stats"] = self._calculate_resampling_stats()
        stats["max_evidence"] = {k: max(v) for k, v in self.evidence.items()}
        stats["target_object_theoretical_limit"] = (
            self._theoretical_limit_target_object_pose_error()
        )
        stats["target_object_pose_error"] = self._mlh_target_object_pose_error()
        return stats

    def _calculate_resampling_stats(self):
        result = {}
        for graph_id, full_stats in self.hypotheses_updater.resampling_stats.items():
            result[graph_id] = {
                channel_stat.input_channel: self._channel_resampling_stats(
                    graph_id, channel_stat
                )
                for channel_stat in full_stats
            }
        return result

    def _channel_resampling_stats(self, graph_id, channel_stat):
        channel_stats = {}
        channel_stats["remove_ids"] = channel_stat.removed_hypotheses_ids
        channel_stats["add_ids"] = channel_stat.added_hypotheses_ids
        channel_stats["evidence_slopes"] = self._hypotheses_evidence_slopes(
            graph_id, channel_stat.input_channel
        )
        channel_stats["rotations"] = self._hypotheses_rotations(
            graph_id, channel_stat.input_channel
        )
        channel_stats["pose_errors"] = self._hypotheses_pose_errors(
            graph_id, channel_stat.input_channel
        )
        return channel_stats

    def _hypotheses_evidence_slopes(
        self, graph_id: str, input_channel: str
    ) -> list[float]:
        return self.hypotheses_updater.evidence_slope_trackers[
            graph_id
        ]._calculate_slopes(input_channel)

    def _hypotheses_rotations(self, graph_id: str, input_channel: str) -> list[float]:
        mapper = self.channel_hypothesis_mapping[graph_id]
        channel_rotations = mapper.extract(self.possible_poses[graph_id], input_channel)
        hyp_rotations = Rotation.from_matrix(channel_rotations).inv()
        return hyp_rotations.as_euler("xyz")

    def _hypotheses_pose_errors(self, graph_id: str, input_channel: str) -> list[float]:
        mapper = self.channel_hypothesis_mapping[graph_id]
        channel_rotations = mapper.extract(self.possible_poses[graph_id], input_channel)
        hyp_rotations = Rotation.from_matrix(channel_rotations).inv()
        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)
        return (hyp_rotations * target_rotation.inv()).magnitude()

    def _theoretical_limit_target_object_pose_error(self) -> float:
        """Compute the theoretical minimum rotation error on the target object.

        This considers all possible hypotheses rotations on the target object
        and compares them to the target's rotation. The theoretical limit conveys the
        best achievable performance if Monty selects the best hypothesis as its most
        likely hypothesis (MLH).

        Note that having a low pose error for the theoretical limit may not be
        sufficient for deciding on the quality of the hypothesis. Despite good
        hypotheses being generally correlated with good theoretical limit, it is
        possible for rotation error to be small (i.e., low geodesic distance to
        ground-truth rotation), while the hypothesis is on a different location
        of the object.

        Returns:
            The minimum achievable rotation error (in radians).
        """
        hyp_rotations = Rotation.from_matrix(
            self.possible_poses[self.primary_target]
        ).inv()
        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)
        error = compute_pose_error(hyp_rotations, target_rotation)
        return error

    def _mlh_target_object_pose_error(self) -> float:
        """Compute the actual rotation error between predicted and target pose.

        This compares the most likely hypothesis pose (based on evidence) on the target
        object with the ground truth rotation of the target object.

        Returns:
            The rotation error (in radians).
        """
        obj_rotation = self.get_mlh_for_object(self.primary_target)["rotation"].inv()
        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)
        error = compute_pose_error(obj_rotation, target_rotation)
        return error
