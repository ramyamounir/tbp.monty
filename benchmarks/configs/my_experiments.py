import os
from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMStackedMontyConfig,
    MontyArgs,
    PretrainLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMStackedDistantMountHabitatDatasetArgs,
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

# The config dictionary for the pretraining experiment.
FiveLM = dict(
    # Specify monty experiment and its args.
    # The MontySupervisedObjectPretrainingExperiment class will provide the model
    # with object and pose labels for supervised pretraining.
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(train_rotations),
        do_eval=False,
    ),
    # Specify logging config.
    logging_config=PretrainLoggingConfig(
        # output_dir=project_dir,
        run_name=model_name,
        wandb_handlers=[],
    ),
    # Specify the Monty config.
    monty_config=FiveLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        # motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    # Set up the environment and agent
    dataset_class=ED.EnvironmentDataset,
    dataset_args=FiveLMStackedDistantMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
    # For a complete config we need to specify an eval_dataloader but since we only
    # train here, this is unused
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
)

experiments = MyExperiments(
    FiveLM=FiveLM,
)
CONFIGS = asdict(experiments)
