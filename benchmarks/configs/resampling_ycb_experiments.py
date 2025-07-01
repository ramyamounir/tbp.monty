# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict

from benchmarks.configs.names import ResamplingYcbExperiments
from benchmarks.configs.ycb_experiments import experiments
from tbp.monty.frameworks.models.evidence_matching.resampling_hypotheses_updater import (
    ResamplingHypothesesUpdater,
)

modified_experiments = {}
for exp_name, cfg in asdict(experiments).items():
    mod_exp_name = "resampling_" + exp_name
    mod_cfg = cfg.copy()

    lm_configs = mod_cfg["monty_config"]["learning_module_configs"]
    for lm in lm_configs:
        lm_configs[lm]["learning_module_args"]["hypotheses_updater_class"] = (
            ResamplingHypothesesUpdater
        )

    modified_experiments[mod_exp_name] = mod_cfg


experiments = ResamplingYcbExperiments(
    resampling_base_config_10distinctobj_dist_agent=modified_experiments[
        "resampling_base_config_10distinctobj_dist_agent"
    ],
    resampling_base_config_10distinctobj_surf_agent=modified_experiments[
        "resampling_base_config_10distinctobj_surf_agent"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent=modified_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent"
    ],
    resampling_randrot_noise_10distinctobj_dist_on_distm=modified_experiments[
        "resampling_randrot_noise_10distinctobj_dist_on_distm"
    ],
    resampling_randrot_noise_10distinctobj_surf_agent=modified_experiments[
        "resampling_randrot_noise_10distinctobj_surf_agent"
    ],
    resampling_randrot_10distinctobj_surf_agent=modified_experiments[
        "resampling_randrot_10distinctobj_surf_agent"
    ],
    resampling_randrot_noise_10distinctobj_5lms_dist_agent=modified_experiments[
        "resampling_randrot_noise_10distinctobj_5lms_dist_agent"
    ],
    resampling_base_10simobj_surf_agent=modified_experiments[
        "resampling_base_10simobj_surf_agent"
    ],
    resampling_randrot_noise_10simobj_surf_agent=modified_experiments[
        "resampling_randrot_noise_10simobj_surf_agent"
    ],
    resampling_randrot_noise_10simobj_dist_agent=modified_experiments[
        "resampling_randrot_noise_10simobj_dist_agent"
    ],
    resampling_randomrot_rawnoise_10distinctobj_surf_agent=modified_experiments[
        "resampling_randomrot_rawnoise_10distinctobj_surf_agent"
    ],
    resampling_base_10multi_distinctobj_dist_agent=modified_experiments[
        "resampling_base_10multi_distinctobj_dist_agent"
    ],
    resampling_surf_agent_unsupervised_10distinctobj=modified_experiments[
        "resampling_surf_agent_unsupervised_10distinctobj"
    ],
    resampling_surf_agent_unsupervised_10distinctobj_noise=modified_experiments[
        "resampling_surf_agent_unsupervised_10distinctobj_noise"
    ],
    resampling_surf_agent_unsupervised_10simobj=modified_experiments[
        "resampling_surf_agent_unsupervised_10simobj"
    ],
    resampling_base_77obj_dist_agent=modified_experiments[
        "resampling_base_77obj_dist_agent"
    ],
    resampling_base_77obj_surf_agent=modified_experiments[
        "resampling_base_77obj_surf_agent"
    ],
    resampling_randrot_noise_77obj_surf_agent=modified_experiments[
        "resampling_randrot_noise_77obj_surf_agent"
    ],
    resampling_randrot_noise_77obj_dist_agent=modified_experiments[
        "resampling_randrot_noise_77obj_dist_agent"
    ],
    resampling_randrot_noise_77obj_5lms_dist_agent=modified_experiments[
        "resampling_randrot_noise_77obj_5lms_dist_agent"
    ],
)
CONFIGS = asdict(experiments)
