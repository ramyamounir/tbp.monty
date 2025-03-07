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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_pre_episode = False

    def pre_episode(self, primary_target, semantic_id_to_label=None):
        if not self.init_pre_episode:
            self.init_pre_episode = True
            return super().pre_episode(primary_target, semantic_id_to_label)

        self.learning_modules[0].curr_ep += 1
        self.min_eval_steps = 100

        self._is_done = False
        self.reset_episode_steps()
        self.switch_to_matching_step()
        self.primary_target = primary_target
        self.semantic_id_to_label = semantic_id_to_label

        for lm in self.learning_modules:
            lm.primary_target = primary_target["object"]
            lm.primary_target_rotation_quat = primary_target["quat_rotation"]


class UnuspervisedEvidenceGraphLM(EvidenceGraphLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curr_ep = 1
        self.updated_graph_ids = []

    def _update_evidence(self, features, displacements, graph_id):
        super()._update_evidence(features, displacements, graph_id)

        if self.curr_ep == 2 and graph_id not in self.updated_graph_ids:
            self.updated_graph_ids.append(graph_id)

            (
                initial_possible_channel_locations,
                initial_possible_channel_rotations,
                channel_evidence,
            ) = self._get_initial_hypothesis_space(features, graph_id, "patch")

            self._add_hypotheses_to_hpspace(
                graph_id=graph_id,
                input_channel="patch",
                new_loc_hypotheses=initial_possible_channel_locations,
                new_rot_hypotheses=initial_possible_channel_rotations,
                new_evidence=channel_evidence,
            )

    def _add_hypotheses_to_hpspace(
        self,
        graph_id,
        input_channel,
        new_loc_hypotheses,
        new_rot_hypotheses,
        new_evidence,
    ):
        """Add new hypotheses to hypothesis space."""
        # Add current mean evidence to give the new hypotheses a fighting
        # chance. TODO H: Test mean vs. median here.
        current_mean_evidence = np.mean(self.evidence[graph_id])
        new_evidence = new_evidence + current_mean_evidence
        # Add new hypotheses to hypothesis space
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
        # Update channel hypothesis mapping
        old_num_hypotheses = self.channel_hypothesis_mapping[graph_id]["num_hypotheses"]
        new_num_hypotheses = old_num_hypotheses + len(new_loc_hypotheses)
        self.channel_hypothesis_mapping[graph_id][input_channel] = [
            0,
            new_num_hypotheses,
        ]
        self.channel_hypothesis_mapping[graph_id]["num_hypotheses"] = new_num_hypotheses

    def _add_detailed_stats(self, stats, get_rotations):
        stats = super()._add_detailed_stats(stats, get_rotations)
        stats["th_limit"] = self._calculate_theoretical_limit()
        return stats

    def _calculate_theoretical_limit(self):
        poses = self.possible_poses[self.primary_target].copy()
        hyp_rotations = Rotation.from_matrix(poses).inv().as_quat().tolist()
        limit = self.get_pose_error(hyp_rotations, self.primary_target_rotation_quat)
        return limit

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

        max_evidences = self.extract_max_evidences(
            data["DETAILED"][total]["LM_0"]["evidences"]
        )
        # data["DETAILED"][total]["LM_0"]["evidences"] = max_evidences
        data["DETAILED"][total] = {"LM_0": {}}
        data["DETAILED"][total]["LM_0"] = {"evidences": max_evidences}

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
