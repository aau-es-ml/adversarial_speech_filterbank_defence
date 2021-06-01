#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-12-2020
           """

from .noises import compute_noised
from .compute_features import compute_transformations

from .split_speech import compute_speech_silence_splits


def run_all_precomputations():
    compute_speech_silence_splits()
    compute_noised()
    compute_transformations()


if __name__ == "__main__":
    run_all_precomputations()
