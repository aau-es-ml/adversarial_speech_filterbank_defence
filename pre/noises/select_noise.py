#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28-10-2020
           """

from scipy.io import wavfile

from apppath import ensure_existence
from configs.path_config import (
    DEMAND_NOISES,
    GENERATED_NOISES_UNPROCESSED_ROOT_PATH,
    NOISES_SPLIT_UNPROCESSED_ROOT_PATH,
    NOISES_UNPROCESSED_ROOT_PATH,
)

from draugr.numpy_utilities import Split
from draugr.tqdm_utilities import progress_bar
from shutil import copyfile


def select_split_noise_files():

    ds = {}
    for a in DEMAND_NOISES.iterdir():
        for b in a.iterdir():
            ds[b.name] = list(b.rglob("*.wav"))[0]

    for k, v in ds.items():
        copyfile(v, GENERATED_NOISES_UNPROCESSED_ROOT_PATH / f"{k}.wav")


if __name__ == "__main__":
    select_split_noise_files()
