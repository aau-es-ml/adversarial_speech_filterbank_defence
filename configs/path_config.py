#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from pathlib import Path

from apppath import AppPath
from warg.imports import import_warning

import_warning(Path(__file__).with_suffix("").name)
PROJECT_APP_PATH = AppPath("adversarial_speech", __author__)
LOG_PATH = PROJECT_APP_PATH.user_log


# SOURCE PATHS - is expected to available
AURORA_NOISES = Path.home() / "Data" / "Audio" / "noises" / "AuroraNoises"
DEMAND_NOISES = Path().home() / "Data" / "Audio" / "noises" / "demand_subset"
MORTEN_NOISES = Path().home() / "Data" / "Audio" / "kolbek_slt2016"

# TARGET PATHS - will be generated
GENERATED_NOISES_UNPROCESSED_ROOT_PATH = (
    PROJECT_APP_PATH.user_data / "unprocessed" / "generated_noises"
)
DATA_ROOT_PATH = Path.home() / "Data" / "Audio" / "adversarial_speech"

DATA_ROOT_A_PATH = DATA_ROOT_PATH / "adversarial_dataset-A"
DATA_ROOT_B_PATH = DATA_ROOT_PATH / "adversarial_dataset-B"

NOISES_SPLIT_UNPROCESSED_ROOT_PATH = (
    PROJECT_APP_PATH.user_data / "unprocessed" / "noises_split"
)
DATA_ROOT_NOISED_UNPROCESSED_PATH = (
    PROJECT_APP_PATH.user_data / "unprocessed" / "noised"
)
DATA_ROOT_SS_SPLITS_UNPROCESSED_PATH = (
    PROJECT_APP_PATH.user_data / "unprocessed" / "splits"
)

DATA_NOISED_PROCESSED_PATH = PROJECT_APP_PATH.user_data / "processed" / "noised"
DATA_SS_SPLITS_PROCESSED_PATH = PROJECT_APP_PATH.user_data / "processed" / "splits"
DATA_REGULAR_PROCESSED_PATH = PROJECT_APP_PATH.user_data / "processed" / "regular"

MODEL_PERSISTENCE_PATH = PROJECT_APP_PATH.user_data / "results"
EXPORT_RESULTS_PATH = PROJECT_APP_PATH.user_data / "export"

BEST_VAL_MODEL_NAME = "best_val_model_params.pt"
FINAL_MODEL_NAME = "final_model_params.pt"
PROCESSED_FILE_ENDING = ".npz"
