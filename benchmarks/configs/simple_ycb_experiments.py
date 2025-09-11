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

from benchmarks.configs.names import SimpleYcbExperiments
from benchmarks.configs.ycb_experiments import experiments

alpha_values = {"alpha1": 0.02, "alpha2": 0.1, "alpha3": 0.3}
simple_ycb_experiments = {}

# alpha values
for exp_name, cfg in asdict(experiments).items():
    if exp_name in [
        "base_config_10distinctobj_dist_agent",
        "randrot_noise_10distinctobj_dist_agent",
        "base_config_10distinctobj_surf_agent",
        "randrot_noise_10distinctobj_surf_agent",
        "randrot_noise_77obj_surf_agent",
    ]:
        # Add Alpha Experiments
        for k, v in alpha_values.items():
            mod_exp_name = exp_name + "_" + k
            mod_cfg = deepcopy(cfg)

            mod_cfg["monty_config"]["learning_module_configs"]["learning_module_0"][
                "learning_module_args"
            ]["present_weight"] = v
            mod_cfg["monty_config"]["learning_module_configs"]["learning_module_0"][
                "learning_module_args"
            ]["past_weight"] = 1 - v

            simple_ycb_experiments[mod_exp_name] = mod_cfg

        # Add No Trans Baseline
        mod_exp_name = exp_name + "_notrans"
        mod_cfg = deepcopy(cfg)

        mod_cfg["monty_config"]["motor_system_config"]["motor_system_args"][
            "policy_args"
        ]["use_goal_state_driven_actions"] = False

        simple_ycb_experiments[mod_exp_name] = mod_cfg

        # Add no Trans alpha experiments
        for k, v in alpha_values.items():
            mod_exp_name = exp_name + "_" + "notrans" + "_" + k
            mod_cfg = deepcopy(cfg)

            mod_cfg["monty_config"]["learning_module_configs"]["learning_module_0"][
                "learning_module_args"
            ]["present_weight"] = v
            mod_cfg["monty_config"]["learning_module_configs"]["learning_module_0"][
                "learning_module_args"
            ]["past_weight"] = 1 - v

            mod_cfg["monty_config"]["motor_system_config"]["motor_system_args"][
                "policy_args"
            ]["use_goal_state_driven_actions"] = False

            simple_ycb_experiments[mod_exp_name] = mod_cfg


CONFIGS = asdict(SimpleYcbExperiments(**simple_ycb_experiments))
