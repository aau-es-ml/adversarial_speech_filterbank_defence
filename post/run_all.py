#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from .export_agg import compute_agg_plots
from .extract_csv import extract_metrics


def run_all_postcomputations():
    extract_metrics()
    compute_agg_plots()


if __name__ == "__main__":
    run_all_postcomputations()
