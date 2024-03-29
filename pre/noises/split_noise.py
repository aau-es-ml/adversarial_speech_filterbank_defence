#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28-10-2020
           """

from apppath import ensure_existence
from draugr.numpy_utilities import SplitEnum
from draugr.visualisation import progress_bar
from scipy.io import wavfile

from configs.path_config import (
    AURORA_NOISES,
    GENERATED_NOISES_UNPROCESSED_ROOT_PATH,
    NOISES_SPLIT_UNPROCESSED_ROOT_PATH,
)


def split_noise_files():
    noise_files = (
        *list(AURORA_NOISES.rglob("*.wav")),
        *list(GENERATED_NOISES_UNPROCESSED_ROOT_PATH.rglob("*.wav")),
    )
    for noise_file in progress_bar(noise_files):
        sr_noise, noise = wavfile.read(str(noise_file))
        split_size = len(noise) // 3
        train, valid, test = (
            noise[:split_size],
            noise[split_size:-split_size],
            noise[-split_size:],
        )
        wavfile.write(
            ensure_existence(
                NOISES_SPLIT_UNPROCESSED_ROOT_PATH / SplitEnum.training.value
            )
            / noise_file.name,
            sr_noise,
            train,
        )
        wavfile.write(
            ensure_existence(
                NOISES_SPLIT_UNPROCESSED_ROOT_PATH / SplitEnum.validation.value
            )
            / noise_file.name,
            sr_noise,
            valid,
        )
        wavfile.write(
            ensure_existence(
                NOISES_SPLIT_UNPROCESSED_ROOT_PATH / SplitEnum.testing.value
            )
            / noise_file.name,
            sr_noise,
            test,
        )


if __name__ == "__main__":
    split_noise_files()
