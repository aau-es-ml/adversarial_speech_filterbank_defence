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

if __name__ == "__main__":

    def plot_rep(verbose: bool = False) -> None:
        if torch.cuda.is_available():
            device = auto_select_available_cuda_device(2048)
        else:
            device = torch.device("cpu")

        global_torch_device(override=device)

        for cepstral_name in progress_bar(CepstralSpaceEnum, description="configs #"):
            for exp_name, exp_v in progress_bar(
                COMMON_TRAINING_CONFIG.EXPERIMENTS, description=f"{cepstral_name}"
            ):

                predictors = []
                categories = []
                test_names = []

                for k, t in exp_v.Test_Sets.items():
                    (pt, ct, nt) = AdversarialSpeechBlockDataset(
                        t.path / f"{cepstral_name.value}_{k}{PROCESSED_FILE_ENDING}",
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
                test_names = numpy.concatenate(test_names)

                predictors = to_tensor(predictors[:, numpy.newaxis, ...])
                categories = to_tensor(categories[:, numpy.newaxis])
                indices = to_tensor(range(len(predictors)))
                print(len(predictors))

                test_loader = DataLoader(
                    TensorDataset(predictors, categories, indices),
                    batch_size=COMMON_TRAINING_CONFIG.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=global_pin_memory(),
                )

                predictions = []
                truth = []

                with torch.no_grad():
                    for (ith_batch, (predictors2, category, indices2)) in zip(
                        range(9), to_device_iterator(test_loader, device=device)
                    ):
                        predictions += predictors
                        truth += category
                        pyplot.matshow(predictors2.to("cpu").numpy()[0][0].T)
                        idx = int(indices2.cpu()[0].item())
                        pyplot.title(
                            f"{list(exp_v.Test_Sets.keys())} {cepstral_name.value} {test_names[idx]}"
                        )
                        pyplot.show()

                    # EXPORT CSV!!!!

                break
            break

    plot_rep()
