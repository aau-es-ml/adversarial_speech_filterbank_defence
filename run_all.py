#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-12-2020
           """

from .model.run_all import run_all_training_testing
from .post.run_all import run_all_postcomputations
from .pre.run_all import run_all_precomputations

if __name__ == "__main__":

    run_all_precomputations()
    run_all_training_testing()
    run_all_postcomputations()
