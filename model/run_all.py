#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from .stest_model import run_all_experiment_test
from .train_model import run_all_experiment_train


def run_all_training_testing():
    run_all_experiment_train()
    run_all_experiment_test()


if __name__ == "__main__":
    run_all_training_testing()
