# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from copy import deepcopy
from dataclasses import asdict

from benchmarks.configs.names import ResamplingYcbExperiments
from benchmarks.configs.ycb_experiments import experiments
from tbp.monty.frameworks.models.evidence_matching.resampling_hypotheses_updater import (  # noqa: E501
    ResamplingHypothesesUpdater,
)

# Adding Resampling to YCB experiments
resampling_ycb_experiments = {}
for exp_name, cfg in asdict(experiments).items():
    mod_exp_name = "resampling_" + exp_name
    mod_cfg = cfg.copy()

    lm_configs = mod_cfg["monty_config"]["learning_module_configs"]
    for lm in lm_configs:
        lm_configs[lm]["learning_module_args"]["hypotheses_updater_class"] = (
            ResamplingHypothesesUpdater
        )

    resampling_ycb_experiments[mod_exp_name] = mod_cfg


# Varying reduction factor
resampling_ycb_reduced_experiments = {}
for exp_name, cfg in asdict(experiments).items():
    if exp_name == "randrot_noise_77obj_dist_agent":
        for reduction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            mod_exp_name = (
                "resampling_" + exp_name + "_" + str(int(reduction * 10)) + "reduction"
            )
            mod_cfg = deepcopy(cfg)

            lm_configs = mod_cfg["monty_config"]["learning_module_configs"]
            for lm in lm_configs:
                lm_configs[lm]["learning_module_args"]["hypotheses_updater_class"] = (
                    ResamplingHypothesesUpdater
                )
                lm_configs[lm]["learning_module_args"]["hypotheses_updater_args"][
                    "hypotheses_space_reduction_factor"
                ] = reduction

            resampling_ycb_reduced_experiments[mod_exp_name] = mod_cfg


# Varying window size
resampling_ycb_window_experiments = {}
for exp_name, cfg in asdict(experiments).items():
    if exp_name in [
        "randrot_noise_10distinctobj_dist_agent",
        "base_config_10distinctobj_dist_agent",
    ]:
        for window in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            mod_exp_name = "resampling_" + exp_name + "_" + str(window) + "window"
            mod_cfg = deepcopy(cfg)

            lm_configs = mod_cfg["monty_config"]["learning_module_configs"]
            for lm in lm_configs:
                lm_configs[lm]["learning_module_args"]["hypotheses_updater_class"] = (
                    ResamplingHypothesesUpdater
                )
                lm_configs[lm]["learning_module_args"]["hypotheses_updater_args"][
                    "slope_tracker_window_size"
                ] = window

            resampling_ycb_window_experiments[mod_exp_name] = mod_cfg


experiments = ResamplingYcbExperiments(
    resampling_base_config_10distinctobj_dist_agent=resampling_ycb_experiments[
        "resampling_base_config_10distinctobj_dist_agent"
    ],
    resampling_base_config_10distinctobj_surf_agent=resampling_ycb_experiments[
        "resampling_base_config_10distinctobj_surf_agent"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent"
    ],
    resampling_randrot_noise_10distinctobj_dist_on_distm=resampling_ycb_experiments[
        "resampling_randrot_noise_10distinctobj_dist_on_distm"
    ],
    resampling_randrot_noise_10distinctobj_surf_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_10distinctobj_surf_agent"
    ],
    resampling_randrot_10distinctobj_surf_agent=resampling_ycb_experiments[
        "resampling_randrot_10distinctobj_surf_agent"
    ],
    resampling_randrot_noise_10distinctobj_5lms_dist_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_10distinctobj_5lms_dist_agent"
    ],
    resampling_base_10simobj_surf_agent=resampling_ycb_experiments[
        "resampling_base_10simobj_surf_agent"
    ],
    resampling_randrot_noise_10simobj_surf_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_10simobj_surf_agent"
    ],
    resampling_randrot_noise_10simobj_dist_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_10simobj_dist_agent"
    ],
    resampling_randomrot_rawnoise_10distinctobj_surf_agent=resampling_ycb_experiments[
        "resampling_randomrot_rawnoise_10distinctobj_surf_agent"
    ],
    resampling_base_10multi_distinctobj_dist_agent=resampling_ycb_experiments[
        "resampling_base_10multi_distinctobj_dist_agent"
    ],
    resampling_surf_agent_unsupervised_10distinctobj=resampling_ycb_experiments[
        "resampling_surf_agent_unsupervised_10distinctobj"
    ],
    resampling_surf_agent_unsupervised_10distinctobj_noise=resampling_ycb_experiments[
        "resampling_surf_agent_unsupervised_10distinctobj_noise"
    ],
    resampling_surf_agent_unsupervised_10simobj=resampling_ycb_experiments[
        "resampling_surf_agent_unsupervised_10simobj"
    ],
    resampling_base_77obj_dist_agent=resampling_ycb_experiments[
        "resampling_base_77obj_dist_agent"
    ],
    resampling_base_77obj_surf_agent=resampling_ycb_experiments[
        "resampling_base_77obj_surf_agent"
    ],
    resampling_randrot_noise_77obj_surf_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_77obj_surf_agent"
    ],
    resampling_randrot_noise_77obj_dist_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_77obj_dist_agent"
    ],
    resampling_randrot_noise_77obj_5lms_dist_agent=resampling_ycb_experiments[
        "resampling_randrot_noise_77obj_5lms_dist_agent"
    ],
    resampling_randrot_noise_77obj_dist_agent_1reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_1reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_2reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_2reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_3reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_3reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_4reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_4reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_5reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_5reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_6reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_6reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_7reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_7reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_8reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_8reduction"
    ],
    resampling_randrot_noise_77obj_dist_agent_9reduction=resampling_ycb_reduced_experiments[
        "resampling_randrot_noise_77obj_dist_agent_9reduction"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_10window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_10window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_20window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_20window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_30window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_30window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_40window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_40window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_50window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_50window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_60window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_60window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_70window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_70window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_80window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_80window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_90window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_90window"
    ],
    resampling_randrot_noise_10distinctobj_dist_agent_100window=resampling_ycb_window_experiments[
        "resampling_randrot_noise_10distinctobj_dist_agent_100window"
    ],
    resampling_base_config_10distinctobj_dist_agent_10window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_10window"
    ],
    resampling_base_config_10distinctobj_dist_agent_20window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_20window"
    ],
    resampling_base_config_10distinctobj_dist_agent_30window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_30window"
    ],
    resampling_base_config_10distinctobj_dist_agent_40window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_40window"
    ],
    resampling_base_config_10distinctobj_dist_agent_50window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_50window"
    ],
    resampling_base_config_10distinctobj_dist_agent_60window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_60window"
    ],
    resampling_base_config_10distinctobj_dist_agent_70window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_70window"
    ],
    resampling_base_config_10distinctobj_dist_agent_80window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_80window"
    ],
    resampling_base_config_10distinctobj_dist_agent_90window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_90window"
    ],
    resampling_base_config_10distinctobj_dist_agent_100window=resampling_ycb_window_experiments[
        "resampling_base_config_10distinctobj_dist_agent_100window"
    ],
)
CONFIGS = asdict(experiments)
