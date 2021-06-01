#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28-10-2020
           """

from scipy.io import wavfile

from apppath import ensure_existence
from configs.path_config import (
    GENERATED_NOISES_UNPROCESSED_ROOT_PATH,
    NOISES_SPLIT_UNPROCESSED_ROOT_PATH,
    NOISES_UNPROCESSED_ROOT_PATH,
)

from draugr.numpy_utilities import Split
from draugr.tqdm_utilities import progress_bar


def split_noise_files():
    noise_files = (
        *list(NOISES_UNPROCESSED_ROOT_PATH.rglob("*.wav")),
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
            ensure_existence(NOISES_SPLIT_UNPROCESSED_ROOT_PATH / Split.Training.value)
            / noise_file.name,
            sr_noise,
            train,
        )
        wavfile.write(
            ensure_existence(
                NOISES_SPLIT_UNPROCESSED_ROOT_PATH / Split.Validation.value
            )
            / noise_file.name,
            sr_noise,
            valid,
        )
        wavfile.write(
            ensure_existence(NOISES_SPLIT_UNPROCESSED_ROOT_PATH / Split.Testing.value)
            / noise_file.name,
            sr_noise,
            test,
        )


if __name__ == "__main__":
    split_noise_files()
