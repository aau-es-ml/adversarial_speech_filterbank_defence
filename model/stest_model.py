#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import os
from logging import error, info
from pathlib import Path

import numpy
import torch
from apppath import ensure_existence
from draugr.numpy_utilities import Split
from draugr.python_utilities import WorkerSession
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    TorchEvalSession,
    auto_select_available_cuda_device,
    global_pin_memory,
    global_torch_device,
    to_device_iterator,
    to_tensor,
)
from draugr.tqdm_utilities import progress_bar
from draugr.writers import TestingCurves, TestingScalars
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.types import Device
from torch.utils.data import DataLoader, TensorDataset

from architectures.adversarial_signal_classifier import AdversarialSignalClassifier
from configs import (
    COMMON_TRAINING_CONFIG,
    EXPERIMENTS,
    MODEL_PERSISTENCE_PATH,
    PROCESSED_FILE_ENDING,
    PROJECT_APP_PATH,
)
from data import (
    AdversarialSpeechBlockDataset,
    export_results_numpy,
    misclassified_names,
)
from pre.cepstral_spaces import CepstralSpaceEnum

__all__ = ["run_all_experiment_test"]


def separate_test(
    *, load_time, exp_name, cepstral_name, model_path: Path, test_sets, device: Device
):
    seed_parent = model_path.parent
    seed = seed_parent.name[-1]
    info(f"seed: {seed}")
    model_id = model_path.relative_to(seed_parent.parent)

    with WorkerSession(0) as num_workers:
        for t in progress_bar(test_sets):
            with TensorBoardPytorchWriter(
                PROJECT_APP_PATH.user_log
                / load_time
                / exp_name
                / f"{cepstral_name.value}"
                / model_id
                / t.path.name,
            ) as writer:

                test_loader = DataLoader(
                    AdversarialSpeechBlockDataset(
                        t.path
                        / f"{cepstral_name.value}_{t.path.name}{PROCESSED_FILE_ENDING}",
                        split=Split.Testing,
                        shuffle_data=True,
                        train_percentage=t.train_percentage,
                        test_percentage=t.test_percentage,
                        validation_percentage=t.validation_percentage,
                        random_seed=int(seed),
                        num_samples=t.num_samples,
                        return_names=True,
                    ),
                    batch_size=COMMON_TRAINING_CONFIG["batch_size"],
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=global_pin_memory(num_workers),
                )

                model = AdversarialSignalClassifier(
                    *test_loader.dataset.__getitem__(0)[0].shape
                )
                model.load_state_dict(torch.load(str(model_path), map_location=device))
                model.to(device)

                predictions = []
                truth = []
                test_names = []
                with TorchEvalSession(model):
                    with torch.no_grad():
                        for (ith_batch, (predictors, category, names)) in enumerate(
                            test_loader
                        ):
                            predictors = to_tensor(
                                predictors, device=device, dtype=float
                            )
                            category = to_tensor(category, device=device, dtype=float)
                            predictions += torch.sigmoid(model(predictors))
                            truth += category
                            test_names += names

                predictions = torch.cat(predictions, 0)
                truth = torch.cat(truth, 0)
                test_names = numpy.array(test_names)

                predictions_int = (predictions > 0.5).cpu().numpy().astype(numpy.int32)
                truth = truth.cpu()
                truth_np = truth.numpy()
                predictions_np = predictions.cpu().numpy()
                accuracy_bw = accuracy_score(truth, predictions_int)

                writer.scalar(TestingScalars.test_accuracy.value, accuracy_bw)
                writer.scalar(
                    TestingScalars.test_precision.value,
                    precision_score(truth_np, predictions_int),
                )
                writer.scalar(
                    TestingScalars.test_recall.value,
                    recall_score(truth_np, predictions_int),
                )
                writer.scalar(
                    TestingScalars.test_receiver_operator_characteristic_auc.value,
                    roc_auc_score(truth_np, predictions_int),
                )
                writer.precision_recall_curve(
                    TestingCurves.test_precision_recall.value,
                    predictions,
                    truth,
                )

                (
                    predictions_per_file,
                    truth_per_file,
                    names_per_file,
                ) = AdversarialSpeechBlockDataset.combine_blocks_to_file(
                    predictions_int, truth_np, test_names
                )

                predictions_per_file_int = (predictions_per_file > 0.5).astype(
                    numpy.int32
                )

                confusion_bw = confusion_matrix(truth_np, predictions_int)
                wrong_names_bw = misclassified_names(
                    predictions_int, truth_np, test_names
                )

                export_results_numpy(
                    export_path=ensure_existence(model_path.parent / t.path.name)
                    / f"test_results",
                    predictions_np=predictions_np,
                    truth_np=truth_np,
                    accuracy_bw=accuracy_bw,
                    confusion_bw=confusion_bw,
                    wrong_names_bw=wrong_names_bw,
                    predictions_per_file=predictions_per_file,
                    truth_per_file=truth_per_file,
                    predictions_per_file_int=predictions_per_file_int,
                    names_per_file=names_per_file,
                )


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
            model.load_state_dict(torch.load(str(model_path), map_location=device))
            model.to(device)

            predictions = []
            truth = []
            with TorchEvalSession(model):
                with torch.no_grad():
                    for (ith_batch, (predictors, category)) in enumerate(
                        to_device_iterator(test_loader, device=device)
                    ):
                        predictors = to_tensor(predictors, device=device, dtype=float)
                        predictions += torch.sigmoid(model(predictors))
                        truth += category

            predictions = torch.stack(predictions)
            truth = torch.stack(truth)

            predictions_int = (predictions > 0.5).cpu().numpy().astype(numpy.int32)
            truth = truth.cpu()
            truth_np = truth.numpy()
            predictions_np = predictions.cpu().numpy()
            accuracy_bw = accuracy_score(truth, predictions_int)

            writer.scalar(TestingScalars.test_accuracy.value, accuracy_bw)
            writer.scalar(
                TestingScalars.test_precision.value,
                precision_score(truth_np, predictions_int),
            )
            writer.scalar(
                TestingScalars.test_recall.value,
                recall_score(truth_np, predictions_int),
            )
            writer.scalar(
                TestingScalars.test_receiver_operator_characteristic_auc.value,
                roc_auc_score(truth_np, predictions_int),
            )
            writer.precision_recall_curve(
                TestingCurves.test_precision_recall.value,
                predictions,
                truth,
            )

            (
                predictions_per_file,
                truth_per_file,
                names_per_file,
            ) = AdversarialSpeechBlockDataset.combine_blocks_to_file(
                predictions_int, truth_np, test_names
            )

            predictions_per_file_int = (predictions_per_file > 0.5).astype(numpy.int32)

            confusion_bw = confusion_matrix(truth_np, predictions_int)
            wrong_names_bw = misclassified_names(predictions_int, truth_np, test_names)

            export_results_numpy(
                export_path=ensure_existence(model_path.parent / merged_name_str)
                / f"test_results",
                predictions_np=predictions_np,
                truth_np=truth_np,
                accuracy_bw=accuracy_bw,
                confusion_bw=confusion_bw,
                wrong_names_bw=wrong_names_bw,
                predictions_per_file=predictions_per_file,
                truth_per_file=truth_per_file,
                predictions_per_file_int=predictions_per_file_int,
                names_per_file=names_per_file,
            )


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
                            else:
                                separate_test(
                                    load_time=load_time,
                                    exp_name=model_name,
                                    cepstral_name=cepstral_name,
                                    model_path=model_path,
                                    test_sets=v,
                                    device=device,
                                )


if __name__ == "__main__":

    run_all_experiment_test(only_latest_load_time=False)
