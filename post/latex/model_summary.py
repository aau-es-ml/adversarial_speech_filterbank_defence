#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26-06-2021
           """

import os
from logging import error, info
from pathlib import Path

import numpy
import torch
from draugr import WorkerSession
from draugr.numpy_utilities import Split
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    auto_select_available_cuda_device,
    global_pin_memory,
    global_torch_device,
    to_tensor,
)
from draugr.tqdm_utilities import progress_bar
from torch.utils.data import DataLoader, TensorDataset

from architectures import AdversarialSignalClassifier
from configs import (
    COMMON_TRAINING_CONFIG,
    EXPERIMENTS,
    MODEL_PERSISTENCE_PATH,
    PROCESSED_FILE_ENDING,
    PROJECT_APP_PATH,
)
from data import AdversarialSpeechBlockDataset
from pre.cepstral_spaces import CepstralSpaceEnum


def merged_test(
    *, load_time, exp_name, cepstral_name, model_path: Path, test_sets, device
):
    seed_parent = model_path.parent
    seed = seed_parent.name[-1]
    info(f"seed: {seed}")
    model_id = model_path.relative_to(seed_parent.parent)

    predictors = []
    categories = []
    test_names = []
    merged_name = []
    for t in test_sets:

        (pt, ct, nt) = AdversarialSpeechBlockDataset(
            t.path / f"{cepstral_name.value}_{t.path.name}{PROCESSED_FILE_ENDING}",
            split=Split.Testing,
            shuffle_data=True,
            train_percentage=t.train_percentage,
            test_percentage=t.test_percentage,
            validation_percentage=t.validation_percentage,
            random_seed=int(seed[-1]),
            num_samples=t.num_samples,
        ).get_all_samples_in_split()

        predictors.append(pt)
        categories.append(ct)
        test_names.append(nt)

        merged_name.append(t.path.name)

    merged_name_str = "_".join(merged_name)

    with WorkerSession(0) as num_workers:
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log
            / load_time
            / exp_name
            / f"{cepstral_name.value}"  # 'A/' or 'AB/'
            / model_id
            / merged_name_str,
        ) as writer:

            predictors = numpy.concatenate(predictors)
            categories = numpy.concatenate(categories)
            test_names = numpy.concatenate(test_names)

            predictors = to_tensor(predictors[:, numpy.newaxis, ...])
            categories = to_tensor(categories[:, numpy.newaxis])
            # test_names = to_tensor(test_names[:, numpy.newaxis])

            test_loader = DataLoader(
                TensorDataset(predictors, categories),
                batch_size=COMMON_TRAINING_CONFIG["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=global_pin_memory(num_workers),
            )

            model = AdversarialSignalClassifier(*predictors.shape[1:])
            print(model.__repr__())
            exit()


def run_all_experiment_test(
    verbose: bool = False,
    only_latest_load_time: bool = False,
    model_names: str = "best_val_model_params.pt",
):
    if torch.cuda.is_available():
        device = auto_select_available_cuda_device(2048)
    else:
        device = torch.device("cpu")

    global_torch_device(override=device)

    for cepstral_name in progress_bar(CepstralSpaceEnum, description="configs #"):
        for exp_name, exp_v in progress_bar(
            EXPERIMENTS, description=f"{cepstral_name}"
        ):

            models = list(MODEL_PERSISTENCE_PATH.rglob(model_names))
            if len(models) == 0:
                raise FileNotFoundError("No models found")

            if only_latest_load_time:
                load_times = (
                    max(models, key=os.path.getctime)
                    .relative_to(MODEL_PERSISTENCE_PATH)
                    .parts[0],
                )
            else:
                load_times = (
                    latest_model.relative_to(MODEL_PERSISTENCE_PATH).parts[0]
                    for latest_model in models
                )

            info(f"Load times being tested: {load_times} ")

            for load_time in progress_bar(load_times, "# load times"):
                for (k, v), (k2, v2) in zip(
                    exp_v.Train_Sets.items(), exp_v.Validation_Sets.items()
                ):
                    model_name = f"{exp_name}_model_{k}_{k2}"
                    model_runs = list(
                        (
                            MODEL_PERSISTENCE_PATH
                            / load_time
                            / model_name
                            / f"{cepstral_name.value}"
                        ).rglob("best_val_model_params.pt")
                    )
                    if not len(model_runs) and False:
                        error("no model runs found, run training first")
                        exit(-1)

                    for model_path in progress_bar(
                        model_runs, description=f"{model_name} seed #"
                    ):
                        if verbose:
                            info(f"Loading model in {model_path}")
                        for k, v in exp_v.Test_Sets.items():
                            if len(v) > 1:
                                merged_test(
                                    load_time=load_time,
                                    exp_name=model_name,
                                    cepstral_name=cepstral_name,
                                    model_path=model_path,
                                    test_sets=v,
                                    device=device,
                                )


if __name__ == "__main__":

    run_all_experiment_test(only_latest_load_time=False)
