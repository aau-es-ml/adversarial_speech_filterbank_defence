#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import time
from pathlib import Path

from warg import NOD, import_warning

__all__ = ["COMMON_TRAINING_CONFIG", "LOAD_TIME"]

import_warning(Path(__file__).with_suffix("").name)
LOAD_TIME = str(int(time.time()))

COMMON_TRAINING_CONFIG = NOD(
    val_interval=1,
    num_runs=5,
    batch_size=64,
)

if __name__ == "__main__":
    pass
