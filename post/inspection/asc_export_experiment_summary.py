#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import numpy
import torch

from matplotlib import pyplot
from torch.utils.data import DataLoader, TensorDataset

from configs.path_config import PROCESSED_FILE_ENDING
from configs.training_config import COMMON_TRAINING_CONFIG
from data import AdversarialSpeechBlockDataset
from configs.experiment_config import EXPERIMENTS
from draugr.torch_utilities import (
    auto_select_available_cuda_device,
    global_torch_device,
    to_device_iterator,
    to_tensor,
    global_pin_memory,
)
from draugr.numpy_utilities import Split
from draugr.tqdm_utilities import progress_bar
from pre.asc_transformation_spaces import CepstralSpaceEnum
import csv

if __name__ == "__main__":

    def gather_data_summary(verbose: bool = False) -> None:
        if torch.cuda.is_available():
            device = auto_select_available_cuda_device(2048)
        else:
            device = torch.device("cpu")

        global_torch_device(override=device)

        summary_path = ensure_existence(EXPORT_RESULTS_PATH / "exp_sum")

        for cepstral_name in progress_bar(CepstralSpaceEnum, description="configs #"):
            for exp_name, exp_v in progress_bar(
                COMMON_TRAINING_CONFIG.EXPERIMENTS, description=f"{cepstral_name}"
            ):

                for k, t_ in exp_v.Train_Sets.items():
                    predictors = []
                    categories = []
                    test_names = []
                    for t in t_.values():
                        (pt, ct, nt) = AdversarialSpeechBlockDataset(
                            t.path
                            / f"{cepstral_name.value}_{k}{PROCESSED_FILE_ENDING}",
                            split=Split.Train,
                            random_seed=0,
                            shuffle_data=True,
                            train_percentage=t.train_percentage,
                            test_percentage=t.test_percentage,
                            validation_percentage=t.validation_percentage,
                            num_samples=t.num_samples,
                        ).get_all_samples_in_split()

                        predictors.append(pt)
                        categories.append(ct)
                        test_names.append(nt)

                    predictors = numpy.concatenate(predictors)
                    categories = numpy.concatenate(categories)

                    num_non_adv = numpy.count_nonzero(categories == 0)
                    num_adv = numpy.count_nonzero(categories == 1)

                    num_examples = num_adv + num_non_adv
                    assert len(predictors) == num_examples

                    with open(
                        summary_path / "train_sets" / f"{exp_name}_{k}.csv",
                        "w",
                        newline="",
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(["num_examples", "num_adv", "num_non_adv"])
                        writer.writerow([num_examples, num_adv, num_non_adv])

                for k, t_ in exp_v.Validation_Sets.items():
                    predictors = []
                    categories = []
                    test_names = []
                    for t in t_.values():
                        (pt, ct, nt) = AdversarialSpeechBlockDataset(
                            t.path
                            / f"{cepstral_name.value}_{k}{PROCESSED_FILE_ENDING}",
                            split=Split.Validation,
                            random_seed=0,
                            shuffle_data=True,
                            train_percentage=t.train_percentage,
                            test_percentage=t.test_percentage,
                            validation_percentage=t.validation_percentage,
                            num_samples=t.num_samples,
                        ).get_all_samples_in_split()

                        predictors.append(pt)
                        categories.append(ct)
                        test_names.append(nt)

                    predictors = numpy.concatenate(predictors)
                    categories = numpy.concatenate(categories)

                    num_non_adv = numpy.count_nonzero(categories == 0)
                    num_adv = numpy.count_nonzero(categories == 1)

                    num_examples = num_adv + num_non_adv
                    assert len(predictors) == num_examples

                    with open(
                        summary_path / "validation_sets" / f"{exp_name}_{k}.csv",
                        "w",
                        newline="",
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(["num_examples", "num_adv", "num_non_adv"])
                        writer.writerow([num_examples, num_adv, num_non_adv])

                for k, t_ in exp_v.Test_Sets.items():
                    predictors = []
                    categories = []
                    test_names = []
                    for t in t_.values():
                        (pt, ct, nt) = AdversarialSpeechBlockDataset(
                            t.path
                            / f"{cepstral_name.value}_{k}{PROCESSED_FILE_ENDING}",
                            split=Split.Test,
                            random_seed=0,
                            shuffle_data=True,
                            train_percentage=t.train_percentage,
                            test_percentage=t.test_percentage,
                            validation_percentage=t.validation_percentage,
                            num_samples=t.num_samples,
                        ).get_all_samples_in_split()

                        predictors.append(pt)
                        categories.append(ct)
                        test_names.append(nt)

                    predictors = numpy.concatenate(predictors)
                    categories = numpy.concatenate(categories)

                    num_non_adv = numpy.count_nonzero(categories == 0)
                    num_adv = numpy.count_nonzero(categories == 1)

                    num_examples = num_adv + num_non_adv
                    assert len(predictors) == num_examples

                    with open(
                        summary_path / "test_sets" / f"{exp_name}_{k}.csv",
                        "w",
                        newline="",
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(["num_examples", "num_adv", "num_non_adv"])
                        writer.writerow([num_examples, num_adv, num_non_adv])

                break
            break

    gather_data_summary()
