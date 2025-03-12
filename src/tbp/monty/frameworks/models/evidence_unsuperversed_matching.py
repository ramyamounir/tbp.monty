# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import json
import logging
import os
import time
from collections import OrderedDict

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.utils.logging_utils import maybe_rename_existing_file


class MontyForUnsupervisedEvidenceGraphMatching(MontyForEvidenceGraphMatching):
    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.init_pre_episode = False

    # def pre_episode(self, primary_target, semantic_id_to_label=None):
    #     if not self.init_pre_episode:
    #         self.init_pre_episode = True
    #         return super().pre_episode(primary_target, semantic_id_to_label)

    #     self.min_eval_steps = 100

    #     self._is_done = False
    #     self.reset_episode_steps()
    #     self.switch_to_matching_step()
    #     self.primary_target = primary_target
    #     self.semantic_id_to_label = semantic_id_to_label

    #     for lm in self.learning_modules:
    #         lm.primary_target = primary_target["object"]
    #         lm.primary_target_rotation_quat = primary_target["quat_rotation"]


class ChannelMapper:
    def __init__(self, channel_sizes=None):
        """
        Initialize the ChannelMapper with an ordered dictionary of channel sizes.
        :param channel_sizes: Dict of {channel_name: size}, maintaining order.
        """
        self.channel_sizes = (
            OrderedDict(channel_sizes) if channel_sizes else OrderedDict()
        )

    @property
    def channels(self):
        return list(self.channel_sizes.keys())

    @property
    def dict(self):
        return self.channel_sizes

    @property
    def total_size(self):
        """Returns the total number of elements across all channels."""
        return sum(self.channel_sizes.values())

    def get_channel_range(self, channel_name):
        """Returns the start and end indices of the given channel."""
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")

        start = 0
        for name, size in self.channel_sizes.items():
            if name == channel_name:
                return (start, start + size - 1)
            start += size

    def increase_channel_size(self, channel_name, value):
        """Increases the size of the specified channel."""
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if self.channel_sizes[channel_name] + value <= 0:
            raise ValueError(
                f"Channel '{channel_name}' size cannot be negative or zero."
            )
        self.channel_sizes[channel_name] += value

    def set_channel_size(self, channel_name, value):
        """Increases the size of the specified channel."""
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if self.channel_sizes[channel_name] <= 0:
            raise ValueError(
                f"Channel '{channel_name}' size cannot be negative or zero."
            )
        self.channel_sizes[channel_name] = value

    def add_channel(self, channel_name, size, position=None):
        """Adds a new channel at a specified position (default is at the end)."""
        if channel_name in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' already exists.")

        if position is None or position >= len(self.channel_sizes):
            self.channel_sizes[channel_name] = size
        else:
            items = list(self.channel_sizes.items())
            items.insert(position, (channel_name, size))
            self.channel_sizes = OrderedDict(items)

    def remove_channel(self, channel_name):
        """Removes a channel from the mapping."""
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        del self.channel_sizes[channel_name]

    def __repr__(self):
        """Returns a string representation of the current channel mapping with computed ranges."""
        ranges = {ch: self.get_channel_range(ch) for ch in self.channel_sizes}
        return f"ChannelMapper({ranges})"


class UnuspervisedEvidenceGraphLM(EvidenceGraphLM):
    def reset(self):
        super().reset()
        # return

        self.evidence = {}
        self.channel_hypothesis_mapping = {}

    def _update_evidence(self, features, displacements, graph_id):
        # super()._update_evidence(features, displacements, graph_id)
        # return

        start_time = time.time()

        # get all usable input channels
        input_channels_to_use = [
            ic
            for ic in list(features.keys())
            if ic in self.get_input_channels_in_graph(graph_id)
        ]

        if graph_id not in self.channel_hypothesis_mapping:
            self.channel_hypothesis_mapping[graph_id] = ChannelMapper()

        # Loop over the input channels.
        for input_channel in input_channels_to_use:
            # If the channel doesn't exist, initialize it and add it.
            if input_channel not in self.channel_hypothesis_mapping[graph_id].channels:
                (
                    initial_possible_channel_locations,
                    initial_possible_channel_rotations,
                    channel_evidence,
                ) = self._get_initial_hypothesis_space(
                    features, graph_id, input_channel
                )

                self._add_hypotheses_to_hpspace(
                    graph_id=graph_id,
                    input_channel=input_channel,
                    new_loc_hypotheses=initial_possible_channel_locations,
                    new_rot_hypotheses=initial_possible_channel_rotations,
                    new_evidence=channel_evidence,
                )

            # If the channel exists update its evidence
            else:
                # Get the observed displacement for this channel
                displacement = displacements[input_channel]
                # Get the IDs in hypothesis space for this channel
                channel_start, channel_end = self.channel_hypothesis_mapping[
                    graph_id
                ].get_channel_range(input_channel)

                # Have to do this for all hypotheses so we don't loose the path
                # information
                rotated_displacements = self.possible_poses[graph_id][
                    channel_start:channel_end
                ].dot(displacement)
                search_locations = (
                    self.possible_locations[graph_id][channel_start:channel_end]
                    + rotated_displacements
                )
                # Threshold hypotheses that we update by evidence for them
                current_evidence_update_threshold = self._get_evidence_update_threshold(
                    graph_id
                )
                # Get indices of hypotheses with evidence > threshold
                hyp_ids_to_test = np.where(
                    self.evidence[graph_id][channel_start:channel_end]
                    >= current_evidence_update_threshold
                )[0]
                num_hypotheses_to_test = hyp_ids_to_test.shape[0]
                if num_hypotheses_to_test > 0:
                    logging.info(
                        f"Testing {num_hypotheses_to_test} out of "
                        f"{self.evidence[graph_id].shape[0]} hypotheses for "
                        f"{graph_id} "
                        f"(evidence > {current_evidence_update_threshold})"
                    )
                    search_locations_to_test = search_locations[hyp_ids_to_test]
                    # Get evidence update for all hypotheses with evidence > current
                    # _evidence_update_threshold
                    new_evidence = self._calculate_evidence_for_new_locations(
                        graph_id,
                        input_channel,
                        search_locations_to_test,
                        features,
                        hyp_ids_to_test,
                    )
                    min_update = np.clip(np.min(new_evidence), 0, np.inf)
                    # Alternatives (no update to other Hs or adding avg) left in
                    # here in case we want to revert back to those.
                    # avg_update = np.mean(new_evidence)
                    # evidence_to_add = np.zeros_like(self.evidence[graph_id])
                    evidence_to_add = (
                        np.ones_like(self.evidence[graph_id][channel_start:channel_end])
                        * min_update
                    )
                    evidence_to_add[hyp_ids_to_test] = new_evidence
                    # If past and present weight add up to 1, equivalent to
                    # np.average and evidence will be bound to [-1, 2]. Otherwise it
                    # keeps growing.
                    self.evidence[graph_id][channel_start:channel_end] = (
                        self.evidence[graph_id][channel_start:channel_end]
                        * self.past_weight
                        + evidence_to_add * self.present_weight
                    )
                self.possible_locations[graph_id][channel_start:channel_end] = (
                    search_locations
                )
        end_time = time.time()
        assert not np.isnan(np.max(self.evidence[graph_id])), "evidence contains NaN."
        logging.debug(
            f"evidence update for {graph_id} took "
            f"{np.round(end_time - start_time,2)} seconds."
            f" New max evidence: {np.round(np.max(self.evidence[graph_id]),3)}"
        )

    def _add_hypotheses_to_hpspace(
        self,
        graph_id,
        input_channel,
        new_loc_hypotheses,
        new_rot_hypotheses,
        new_evidence,
    ):
        # super()._add_hypotheses_to_hpspace(
        #     graph_id,
        #     input_channel,
        #     new_loc_hypotheses,
        #     new_rot_hypotheses,
        #     new_evidence,
        # )
        # return

        """Add new hypotheses to hypothesis space."""
        # Add current mean evidence to give the new hypotheses a fighting
        # chance. TODO H: Test mean vs. median here.

        if graph_id not in self.evidence.keys():
            self.possible_locations[graph_id] = new_loc_hypotheses
            self.possible_poses[graph_id] = new_rot_hypotheses
            self.evidence[graph_id] = new_evidence

        else:
            current_mean_evidence = np.mean(self.evidence[graph_id])
            new_evidence = new_evidence + current_mean_evidence
            self.possible_locations[graph_id] = np.vstack(
                [
                    self.possible_locations[graph_id],
                    new_loc_hypotheses,
                ]
            )
            self.possible_poses[graph_id] = np.vstack(
                [
                    self.possible_poses[graph_id],
                    new_rot_hypotheses,
                ]
            )
            self.evidence[graph_id] = np.hstack([self.evidence[graph_id], new_evidence])

        self.channel_hypothesis_mapping[graph_id].add_channel(
            input_channel, len(new_loc_hypotheses)
        )

    def _add_detailed_stats(self, stats, get_rotations):
        stats = super()._add_detailed_stats(stats, get_rotations)
        stats["th_limit"] = self._calculate_theoretical_limit()
        stats["mlh_error"] = self._get_pose_error_mlh()
        stats["obj_error"] = self._get_pose_error_target()
        return stats

    def _calculate_theoretical_limit(self):
        poses = self.possible_poses[self.primary_target].copy()
        hyp_rotations = Rotation.from_matrix(poses).inv().as_quat().tolist()
        limit = self.get_pose_error(hyp_rotations, self.primary_target_rotation_quat)
        return limit

    def _get_pose_error_mlh(self):
        target_rot = Rotation.from_quat(self.primary_target_rotation_quat)
        difference = self.current_mlh["rotation"] * target_rot.inv()
        return difference.magnitude()

    def _get_pose_error_target(self):
        target_rot = Rotation.from_quat(self.primary_target_rotation_quat)
        obj_mlh_id = np.argmax(self.evidence[self.primary_target])
        obj_rotation = Rotation.from_matrix(
            self.possible_poses[self.primary_target][obj_mlh_id]
        )

        difference = obj_rotation * target_rot.inv()
        return difference.magnitude()

    def get_pose_error(self, detected_pose, target_pose):
        target_r = Rotation.from_quat(target_pose)
        min_error = np.pi
        for det_r in detected_pose:
            detected_r = Rotation.from_quat(det_r)
            difference = detected_r * target_r.inv()
            error = difference.magnitude()
            if error < min_error:
                min_error = error
        return min_error


class MaxEvidenceJSONHandler(DetailedJSONHandler):
    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        """Report episode data.

        Changed name to report episode since we are currently running with
        reporting and flushing exactly once per episode.
        """
        output_data = dict()
        if mode == "train":
            total = kwargs["train_episodes_to_total"][episode]
            stats = data["BASIC"]["train_stats"][episode]

        elif mode == "eval":
            total = kwargs["eval_episodes_to_total"][episode]
            stats = data["BASIC"]["eval_stats"][episode]

        # new filtered dictionary to save
        lm_data = {}

        # extract max evidence instead of all evidences
        max_evidences = self.extract_max_evidences(
            data["DETAILED"][total]["LM_0"]["evidences"]
        )
        lm_data["evidences"] = max_evidences
        lm_data["target"] = data["DETAILED"][total]["LM_0"]["target"]
        lm_data["th_limit"] = data["DETAILED"][total]["LM_0"]["th_limit"]
        lm_data["current_mlh"] = data["DETAILED"][total]["LM_0"]["current_mlh"]
        lm_data["mlh_error"] = data["DETAILED"][total]["LM_0"]["mlh_error"]
        lm_data["obj_error"] = data["DETAILED"][total]["LM_0"]["obj_error"]

        data["DETAILED"][total] = {"LM_0": {}}
        data["DETAILED"][total]["LM_0"] = lm_data

        output_data[total] = copy.deepcopy(stats)
        output_data[total].update(data["DETAILED"][total])

        save_stats_path = os.path.join(output_dir, "detailed_run_stats.json")
        maybe_rename_existing_file(save_stats_path, ".json", self.report_count)

        with open(save_stats_path, "a") as f:
            json.dump({total: output_data[total]}, f, cls=BufferEncoder)
            f.write(os.linesep)

        print("Stats appended to " + save_stats_path)
        self.report_count += 1

    def extract_max_evidences(self, evidences_lst):
        max_evidences = []
        for step in evidences_lst:
            max_evidences.append({k: max(v) for k, v in step.items()})
        return max_evidences
