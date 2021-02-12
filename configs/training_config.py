#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import time
from pathlib import Path

from warg import NOD, import_warning

__all__ = ["TRAINING_CONFIG", "LOAD_TIME"]


import_warning(Path(__file__).with_suffix("").name)
LOAD_TIME = str(int(time.time()))

TRAINING_CONFIG = NOD(
    epochs=50,
    val_interval=1,  # 5
    num_runs=5,
    batch_size=64,
    learning_rate=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    trunc_size=None,
)

if __name__ == "__main__":
    pass
