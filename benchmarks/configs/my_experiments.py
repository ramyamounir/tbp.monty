# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    # FiveLMStackedMontyConfig,
    LoggingConfig,
    MontyArgs,
    TwoLMStackedMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    DebugExperimentArgs,
    EnvironmentDataloaderPerObjectArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.simulators.habitat.configs import (
    TwoLMStackedDistantMountHabitatDatasetArgs,
)

"""
Basic setup
-----------
"""
# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "FiveLMStackedMonty"


"""
Training
----------------------------------------------------------------------------------------
"""
# Here we specify which objects to learn. 'mug' and 'banana' come from the YCB dataset.
# If you don't have the YCB dataset, replace with names from habitat (e.g.,
# 'capsule3DSolid', 'cubeSolid', etc.).
object_names = ["mug", "banana"]
# Get predefined object rotations that give good views of the object from 14 angles.
train_rotations = get_cube_face_and_corner_views_rotations()
test_rotations = [[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 180.0, 0.0]]


two_stacked_lms_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_0": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            feature_weights={},
            use_multithreading=False,
            path_similarity_threshold=0.001,
            x_percent_threshold=30,
            required_symmetry_evidence=20,
            # generally would want this to be smaller than second LM
            # but setting same for now to get interesting results with YCB
            max_graph_size=0.3,
            num_model_voxels_per_dim=30,
            max_nodes_per_graph=500,
            hypotheses_updater_args=dict(max_nneighbors=5),
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_1": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                },
                # object Id currently is an int representation of the strings
                # in the object label so we keep this tolerance high. This is
                # just until we have added a way to encode object ID with some
                # real similarity measure.
                "learning_module_0": {"object_id": 1},
            },
            feature_weights={"learning_module_0": {"object_id": 1}},
            use_multithreading=False,
            path_similarity_threshold=0.001,
            x_percent_threshold=30,
            required_symmetry_evidence=20,
            max_graph_size=0.3,
            num_model_voxels_per_dim=30,
            max_nodes_per_graph=500,
            hypotheses_updater_args=dict(max_nneighbors=5),
        ),
    ),
)

TwoLM = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=DebugExperimentArgs(
        do_eval=False,
        do_train=True,
        n_train_epochs=2,  # len(test_rotations),
        n_eval_epochs=len(test_rotations),
        # model_name_or_path=five_lm_10dist_obj,
        max_train_steps=200,
        max_eval_steps=200,
        min_lms_match=2,
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=10000, min_train_steps=3),
        learning_module_configs=two_stacked_lms_config,
    ),
    logging_config=LoggingConfig(
        python_log_level="INFO",
        monty_handlers=[],
        monty_log_level="BASIC",
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=TwoLMStackedDistantMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 2),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0.0, 0.0, 0.0]]),
    ),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)


experiments = MyExperiments(
    FiveLM=TwoLM,
)
CONFIGS = asdict(experiments)
