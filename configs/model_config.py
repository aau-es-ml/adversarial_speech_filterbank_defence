#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import time
from pathlib import Path

from warg import NOD, import_warning

__all__ = ["MODEL_CONFIG", "LOAD_TIME"]


import_warning(Path(__file__).with_suffix("").name)
LOAD_TIME = str(int(time.time()))

MODEL_CONFIG = NOD(
    block_window_size_ms=512,  # 128
    block_window_step_size_ms=512,  # 128
    cepstral_window_length_ms=32,
    n_fcc=20,  # 40, # 20 # 13 "12
    sample_rate=16000,
)

if __name__ == "__main__":
    pass
