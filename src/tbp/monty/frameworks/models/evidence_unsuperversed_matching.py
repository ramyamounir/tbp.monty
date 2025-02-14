# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.models.evidence_matching import MontyForEvidenceGraphMatching


class MontyForUnsupervisedEvidenceGraphMatching(MontyForEvidenceGraphMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_pre_episode = False

    def pre_episode(self, primary_target, semantic_id_to_label=None):
        if not self.init_pre_episode:
            self.init_pre_episode = True
            return super().pre_episode(primary_target, semantic_id_to_label)
        else:
            self._is_done = False
            self.reset_episode_steps()
            self.switch_to_matching_step()
            self.primary_target = primary_target
            self.semantic_id_to_label = semantic_id_to_label
