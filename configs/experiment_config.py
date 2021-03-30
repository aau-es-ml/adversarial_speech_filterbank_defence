#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from copy import deepcopy
from itertools import product
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from warg import import_warning, NOD

__all__ = ["EXPERIMENTS"]

from configs.misc_config import NOISES, SNR_RATIOS

import_warning(Path(__file__).with_suffix("").name)

from draugr.numpy_utilities import Split
from warg import GDKC
from configs.path_config import (
    DATA_NOISED_PROCESSED_PATH,
    DATA_REGULAR_PROCESSED_PATH,
    DATA_SS_SPLITS_PROCESSED_PATH,
)
from data.persistence_helper import extract_tuple_from_path

DEFAULT_NUM_EPOCHS = 99
DEFAULT_OPTIMISER_SPEC = GDKC(
    constructor=torch.optim.Adam, lr=6e-4, betas=(0.9, 0.999,),
)
# DEFAULT_SCHEDULER_SPEC = GDKC(constructor=ReduceLROnPlateau) # scheduler.step(val_loss) # TODO: UNUSED, not using SGD

DEFAULT_DISTRIBUTION = NOD(
    num_samples=None,
    train_percentage=0.7,
    test_percentage=0.2,
    validation_percentage=0.1,
)
VAL_TEST_DISTRIBUTION = NOD(
    num_samples=None,
    train_percentage=0,
    test_percentage=0.2 / 0.3,
    validation_percentage=0.1 / 0.3,
)
FULL_TRAIN_DISTRIBUTION = NOD(
    num_samples=None, train_percentage=1.0, test_percentage=0, validation_percentage=0
)
FULL_TEST_DISTRIBUTION = NOD(
    num_samples=None, train_percentage=0, test_percentage=1.0, validation_percentage=0
)
FULL_VAL_DISTRIBUTION = NOD(
    num_samples=None, train_percentage=0, test_percentage=0, validation_percentage=1.0
)

A = NOD(path=DATA_REGULAR_PROCESSED_PATH / "A", **DEFAULT_DISTRIBUTION)
B = NOD(path=DATA_REGULAR_PROCESSED_PATH / "B", **DEFAULT_DISTRIBUTION)

A_SPEECH = NOD(
    path=DATA_SS_SPLITS_PROCESSED_PATH / "A" / "speech", **DEFAULT_DISTRIBUTION
)
A_SILENCE = NOD(
    path=DATA_SS_SPLITS_PROCESSED_PATH / "A" / "silence", **DEFAULT_DISTRIBUTION
)

DATA_A_NOISED_TRAINING_SPEECH_PATH = (
    DATA_NOISED_PROCESSED_PATH / "A" / Split.Training.value
)
DATA_A_NOISED_TESTING_SPEECH_PATH = (
    DATA_NOISED_PROCESSED_PATH / "A" / Split.Testing.value
)
DATA_A_NOISED_VALIDATION_SPEECH_PATH = (
    DATA_NOISED_PROCESSED_PATH / "A" / Split.Validation.value
)

NOISED_SSS = {
    k: v
    for e in [
        {
            f"{noise_type}_SNR_{strength}dB_train": NOD(
                path=DATA_A_NOISED_TRAINING_SPEECH_PATH
                / f"{noise_type}_SNR_{strength}dB",
                **FULL_TRAIN_DISTRIBUTION,
            ),
            f"{noise_type}_SNR_{strength}dB_test": NOD(
                path=DATA_A_NOISED_TESTING_SPEECH_PATH
                / f"{noise_type}_SNR_{strength}dB",
                **FULL_TEST_DISTRIBUTION,
            ),
            f"{noise_type}_SNR_{strength}dB_validation": NOD(
                path=DATA_A_NOISED_TESTING_SPEECH_PATH
                / f"{noise_type}_SNR_{strength}dB",
                **FULL_VAL_DISTRIBUTION,
            ),
        }
        for noise_type, strength in product(NOISES, SNR_RATIOS)
    ]
    for k, v in e.items()
}
NOISED_SSS["no_aug_train"] = NOD(
    path=DATA_A_NOISED_TRAINING_SPEECH_PATH / "no_aug", **FULL_TRAIN_DISTRIBUTION,
)
NOISED_SSS["no_aug_test"] = NOD(
    path=DATA_A_NOISED_TESTING_SPEECH_PATH / "no_aug", **FULL_TEST_DISTRIBUTION,
)
NOISED_SSS["no_aug_validation"] = NOD(
    path=DATA_A_NOISED_VALIDATION_SPEECH_PATH / "no_aug", **FULL_VAL_DISTRIBUTION,
)

NUM_SAMPLES_TRUNC_A_B = min(
    extract_tuple_from_path(A.path / "gfcc_A.npz")[0].__len__(),
    extract_tuple_from_path(B.path / "gfcc_B.npz")[0].__len__(),
)

TRUNCATED_A = deepcopy(A)
TRUNCATED_A.num_samples = NUM_SAMPLES_TRUNC_A_B
TRUNCATED_B = deepcopy(B)
TRUNCATED_B.num_samples = NUM_SAMPLES_TRUNC_A_B

NUM_SAMPLES_TRUNC_A_SPEECH_SILENCE = min(
    extract_tuple_from_path(A_SPEECH.path / "lfcc_speech.npz")[0].__len__(),
    extract_tuple_from_path(A_SILENCE.path / "lfcc_silence.npz")[0].__len__(),
)

TRUNCATED_A_SPEECH = deepcopy(A_SPEECH)
TRUNCATED_A_SPEECH.num_samples = NUM_SAMPLES_TRUNC_A_SPEECH_SILENCE
TRUNCATED_A_SILENCE = deepcopy(A_SILENCE)
TRUNCATED_A_SILENCE.num_samples = NUM_SAMPLES_TRUNC_A_SPEECH_SILENCE

TRUNCATED_SPLITS = {
    f"TRUNCATED_A_speech": NOD(  # Model_A_speech
        Train_Sets={"speech": (TRUNCATED_A_SPEECH,)},
        Validation_Sets={"speech": (TRUNCATED_A_SPEECH,)},
        Test_Sets={"speech": (TRUNCATED_A_SPEECH,), "silence": (TRUNCATED_A_SILENCE,),},
        Num_Epochs=DEFAULT_NUM_EPOCHS,
        Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
    ),
    f"TRUNCATED_A_silence": NOD(  # Model_A_silence
        Train_Sets={"silence": (TRUNCATED_A_SILENCE,)},
        Validation_Sets={"silence": (TRUNCATED_A_SILENCE,)},
        Test_Sets={"speech": (TRUNCATED_A_SPEECH,), "silence": (TRUNCATED_A_SILENCE,),},
        Num_Epochs=DEFAULT_NUM_EPOCHS,
        Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
    ),
}
TRUNCATED_SETS = {
    f"TRUNCATED_A": NOD(  # MODEL_A
        Train_Sets={"A": (TRUNCATED_A,)},
        Validation_Sets={"A": (TRUNCATED_A,)},
        Test_Sets={"A": (TRUNCATED_A,), "B": (TRUNCATED_B,)},
        Num_Epochs=DEFAULT_NUM_EPOCHS,
        Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
    ),
    f"TRUNCATED_B": NOD(  # Model_B
        Train_Sets={"B": (TRUNCATED_B,)},
        Validation_Sets={"B": (TRUNCATED_B,)},
        Test_Sets={"A": (TRUNCATED_A,), "B": (TRUNCATED_B,)},
        Num_Epochs=DEFAULT_NUM_EPOCHS,
        Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
    ),
}

MERGED_SETS = NOD(
    MERGE_AB_TEST_AB=NOD(
        Train_Sets={"AB": (A, B)},
        Validation_Sets={"AB": (A, B)},
        Test_Sets={"AB": (A, B), "A": (A,), "B": (B,)},
        Num_Epochs=DEFAULT_NUM_EPOCHS,
        Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
    ),
)

NO_AUG_TO_NOISE = {
    f"EQUAL_MAPPING_NO_AUG_to_NOISE_ALL_SNR": NOD(
        Train_Sets={"no_aug": (NOISED_SSS["no_aug_train"],)},
        Validation_Sets={"no_aug": (NOISED_SSS["no_aug_validation"],)},
        Test_Sets={
            k: v
            for e in [
                {
                    f"{noise_type}_SNR_{strength}dB": (
                        NOISED_SSS[f"{noise_type}_SNR_{strength}dB_test"],
                    )
                    for strength in SNR_RATIOS
                }
                for noise_type in NOISES
            ]
            for k, v in e.items()
        },
        Num_Epochs=DEFAULT_NUM_EPOCHS,
        Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
    )
}

NOISED_SETS = {
    k: v
    for e in [
        {
            f"EQUAL_MAPPING_{noise_type}_SNR_{strength}dB_to_{noise_type}_SNR_{strength}dB": NOD(
                Train_Sets={
                    f"{noise_type}_SNR_{strength}dB": (
                        NOISED_SSS[f"{noise_type}_SNR_{strength}dB_train"],
                    )
                },
                Validation_Sets={
                    f"{noise_type}_SNR_{strength}dB": (
                        NOISED_SSS[f"{noise_type}_SNR_{strength}dB_validation"],
                    )
                },
                Test_Sets={
                    f"{noise_type}_SNR_{strength}dB": (
                        NOISED_SSS[f"{noise_type}_SNR_{strength}dB_test"],
                    )
                },
                # Test_Sets={
                #    f"{noise_type1}_SNR_{strength1}dB": (
                #        NOISED_SSS[f"{noise_type1}_SNR_{strength1}dB_test"],
                #    )
                #    for noise_type1, strength1 in product(NOISES, STRENGTHS) #TODO: DO not test across
                # },
                Num_Epochs=DEFAULT_NUM_EPOCHS,
                Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
            )
        }
        for noise_type, strength in product(NOISES, SNR_RATIOS)
    ]
    for k, v in e.items()
}

# NOISES_TO_CAF_OPT_SPEC = deepcopy(DEFAULT_OPTIMISER_SPEC)
# NOISES_TO_CAF_OPT_SPEC.lr =1e-4
NOISES_TO_CAF = {
    f"NOISES_ALL_SNR_to_PCAFETER": NOD(
        Train_Sets={
            "ALL_NOISE_SNR": [
                NOISED_SSS[f"{noise_type}_SNR_{strength}dB_train"]
                for strength, noise_type in product(SNR_RATIOS, NOISES)
                if noise_type != "PCAFETER"
            ]
        },
        Validation_Sets={
            "ALL_NOISE_SNR": [
                NOISED_SSS[f"{noise_type}_SNR_{strength}dB_validation"]
                for strength, noise_type in product(SNR_RATIOS, NOISES)
                if noise_type != "PCAFETER"
            ]
        },
        Test_Sets={
            **{
                k: v
                for e in [
                    {
                        f"{noise_type}_SNR_{strength}dB": (
                            NOISED_SSS[f"{noise_type}_SNR_{strength}dB_test"],
                        )
                        for strength in SNR_RATIOS
                    }
                    for noise_type in ("PCAFETER",)
                ]
                for k, v in e.items()
            },
            **{
                "PCAFETER_ALL_SNR": [
                    NOISED_SSS[f"{noise_type}_SNR_{strength}dB_test"]
                    for strength, noise_type in product(SNR_RATIOS, NOISES)
                    if noise_type == "PCAFETER"
                ]
            },
        },
        Num_Epochs=DEFAULT_NUM_EPOCHS,
        Optimiser_Spec=DEFAULT_OPTIMISER_SPEC,
    )
}

EXPERIMENTS = NOD(
    **TRUNCATED_SETS,
    **MERGED_SETS,
    **TRUNCATED_SPLITS,
    # **NO_AUG_TO_NOISE,
    # **NOISES_TO_CAF,
    # **NOISED_SETS,
)

if __name__ == "__main__":
    # print(EXPERIMENTS.keys())
    print(NOISES_TO_CAF["NOISES_ALL_SNR_to_PCAFETER"]["Test_Sets"])