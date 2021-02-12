#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-12-2020
           """

from .asc_compute_features import compute_transformations
from .asc_noise_augmentation import compute_noise_augmented_samples
from .asc_split_noise import split_noise_files
from .asc_split_speech import compute_speech_silence_splits


def run_all_precomputations():
    split_noise_files()
    compute_speech_silence_splits()
    compute_noise_augmented_samples()
    compute_transformations()


if __name__ == "__main__":
    run_all_precomputations()
