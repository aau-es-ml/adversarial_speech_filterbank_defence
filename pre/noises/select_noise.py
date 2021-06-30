#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28-10-2020
           """

from shutil import copyfile

from configs.path_config import (
    DEMAND_NOISES,
    GENERATED_NOISES_UNPROCESSED_ROOT_PATH,
)


def select_split_noise_files():
    ds = {}
    for a in DEMAND_NOISES.iterdir():
        for b in a.iterdir():
            ds[b.name] = list(b.rglob("*.wav"))[0]

    for k, v in ds.items():
        copyfile(v, GENERATED_NOISES_UNPROCESSED_ROOT_PATH / f"{k}.wav")


if __name__ == "__main__":
    select_split_noise_files()
