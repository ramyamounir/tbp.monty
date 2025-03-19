# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import copy
import os
from dataclasses import asdict

from benchmarks.configs.defaults import (
    default_evidence_lm_config,
    min_eval_steps,
    pretrained_dir,
)
from benchmarks.configs.names import UnsupervisedExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    PatchAndViewSOTAMontyConfig,
    SurfaceAndViewSOTAMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.models.evidence_unsuperversed_matching import (
    MaxEvidenceJSONHandler,
    MontyForUnsupervisedEvidenceGraphMatching,
    UnuspervisedEvidenceGraphLM,
)
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)

# Main parameters
num_rotations = 1
num_objects = 2
objects_list = ["strawberry", "banana"]

initial_possible_poses = "informed"
# initial_possible_poses = "uniform"

use_multithreading = False
# use_multithreading = True

evidence_update_threshold = "x_percent_threshold"
# evidence_update_threshold = "all"

# define rotations
test_rotations = get_cube_face_and_corner_views_rotations()[:num_rotations]

# define data path for supervised graph models
model_path_10distinctobj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_10distinctobj/pretrained/",
)

# define lm configs for surf agent
lower_max_nneighbors_surf_lm_config = copy.deepcopy(default_evidence_lm_config)
lower_max_nneighbors_surf_lm_config["learning_module_class"] = (
    UnuspervisedEvidenceGraphLM
)
lower_max_nneighbors_surf_lm_config["learning_module_args"][
    "evidence_update_threshold"
] = evidence_update_threshold
lower_max_nneighbors_surf_lm_config["learning_module_args"]["max_nneighbors"] = 5
lower_max_nneighbors_surf_lm_config["learning_module_args"]["gsg_args"][
    "desired_object_distance"
] = 0.025
lower_max_nneighbors_surf_lm_config["learning_module_args"]["use_multithreading"] = (
    use_multithreading
)
lower_max_nneighbors_surf_lm_config["learning_module_args"][
    "initial_possible_poses"
] = initial_possible_poses

initial_possible_poses
lower_max_nneighbors_surf_1lm_config = dict(
    learning_module_0=lower_max_nneighbors_surf_lm_config
)


unsupervised_distinctobj_surf_agent = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=len(test_rotations),
        # max_total_steps=5000,
        max_eval_steps=100,
    ),
    logging_config=DetailedEvidenceLMLoggingConfig(
        monty_handlers=[
            BasicCSVStatsHandler,
            MaxEvidenceJSONHandler,
            ReproduceEpisodeHandler,
        ],
        wandb_handlers=[],
        # python_log_level="WARNING",
    ),
    monty_config=SurfaceAndViewSOTAMontyConfig(
        monty_class=MontyForUnsupervisedEvidenceGraphMatching,
        learning_module_configs=lower_max_nneighbors_surf_1lm_config,
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=10),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=objects_list,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)


# define lm configs for dist agent
lower_max_nneighbors_dist_lm_config = copy.deepcopy(default_evidence_lm_config)
lower_max_nneighbors_dist_lm_config["learning_module_class"] = (
    UnuspervisedEvidenceGraphLM
)
lower_max_nneighbors_dist_lm_config["learning_module_args"][
    "evidence_update_threshold"
] = evidence_update_threshold
lower_max_nneighbors_dist_lm_config["learning_module_args"]["use_multithreading"] = (
    use_multithreading
)
lower_max_nneighbors_dist_lm_config["learning_module_args"]["max_nneighbors"] = 5
lower_max_nneighbors_dist_lm_config["learning_module_args"][
    "initial_possible_poses"
] = initial_possible_poses
lower_max_nneighbors_dist_lm_config = dict(
    learning_module_0=lower_max_nneighbors_dist_lm_config
)


unsupervised_distinctobj_dist_agent = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=len(test_rotations),
        # max_total_steps=5000,
        # max_eval_steps=100,
    ),
    logging_config=DetailedEvidenceLMLoggingConfig(
        monty_handlers=[
            BasicCSVStatsHandler,
            MaxEvidenceJSONHandler,
            ReproduceEpisodeHandler,
        ],
        wandb_handlers=[],
        # python_log_level="WARNING",
    ),
    monty_config=PatchAndViewSOTAMontyConfig(
        monty_class=MontyForUnsupervisedEvidenceGraphMatching,
        learning_module_configs=lower_max_nneighbors_dist_lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=10),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=objects_list,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)


experiments = UnsupervisedExperiments(
    unsupervised_distinctobj_surf_agent=unsupervised_distinctobj_surf_agent,
    unsupervised_distinctobj_dist_agent=unsupervised_distinctobj_dist_agent,
)
CONFIGS = asdict(experiments)
