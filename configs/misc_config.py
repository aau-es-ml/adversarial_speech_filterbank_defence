#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from pathlib import Path

from warg import NOD, import_warning

__all__ = ["MISC_CONFIG"]


import_warning(Path(__file__).with_suffix("").name)

MISC_CONFIG = NOD(
    projection_num_samples=1000, tsne_learning_rate=1000, tnse_perplexity=50,
)

if __name__ == "__main__":
    pass
