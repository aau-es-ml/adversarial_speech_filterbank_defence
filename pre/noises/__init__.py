#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-02-2021
           """

from .noise_augmentation import compute_noise_augmented_samples
from .noise_generation import generate_babble, generate_ssn
from .select_noise import select_split_noise_files
from .split_noise import split_noise_files


def compute_noised():
    select_split_noise_files()
    generate_ssn()
    generate_babble()
    split_noise_files()
    compute_noise_augmented_samples()


if __name__ == "__main__":
    compute_noised()
