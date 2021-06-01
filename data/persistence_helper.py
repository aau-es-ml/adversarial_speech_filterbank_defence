#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from pathlib import Path
from typing import Sequence, Tuple

import numpy
from sklearn.metrics import accuracy_score, confusion_matrix
from data.adversarial_speech_dataset import misclassified_names

__all__ = ["export_results_numpy", "export_to_path", "extract_tuple_from_path"]


def extract_tuple_from_path(_path, verbose: bool = False) -> Tuple:
    data = numpy.load(str(_path))
    if verbose:
        print(f"loading data from {_path}")
    return data["features"], data["category"], data["id"]


def export_results_numpy(
    *,
    export_path,
    predictions_np,
    truth_np,
    accuracy_bw,
    confusion_bw,
    wrong_names_bw,
    predictions_per_file,
    truth_per_file,
    predictions_per_file_int,
    names_per_file,
    enabled: bool = False,
):
    """
    Not used for now, can be omitted

    :param enabled:
    :param export_path:
    :param predictions_np:
    :param truth_np:
    :param accuracy_bw:
    :param confusion_bw:
    :param wrong_names_bw:
    :param predictions_per_file:
    :param truth_per_file:
    :param predictions_per_file_int:
    :param names_per_file:
    :return:"""
    if enabled:
        numpy.savez(
            export_path,
            y_bw=predictions_np,
            d_bw=truth_np,
            accuracy_bw=accuracy_bw,
            confusion_bw=confusion_bw,
            wrong_names_bw=wrong_names_bw,
            y_pf=predictions_per_file,
            d_pf=truth_per_file,
            accuracy_pf=accuracy_score(truth_per_file, predictions_per_file_int),
            confusion_pf=confusion_matrix(truth_per_file, predictions_per_file_int),
            wrongnames_pf=misclassified_names(
                predictions_per_file_int, truth_per_file, names_per_file
            ),
        )


def export_to_path(
    out_file: Path, features: Sequence, category: Sequence, block_sample_ids: Sequence
) -> None:
    numpy.savez(
        str(out_file), features=features, category=category, id=block_sample_ids
    )
