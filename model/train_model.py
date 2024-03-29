#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import math
import time
from logging import info
from pathlib import Path
from typing import Iterable, Tuple

import numpy
import torch
from apppath import ensure_existence
from draugr.numpy_utilities import SplitEnum
from draugr.os_utilities import WorkerSession
from draugr.random_utilities import seed_stack
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    TorchEvalSession,
    TorchTrainSession,
    auto_select_available_cuda_device,
    global_pin_memory,
    global_torch_device,
    to_device_iterator,
    to_scalar,
    to_tensor,
    torch_clean_up,
)
from draugr.visualisation import progress_bar
from draugr.writers.standard_tags import (
    StandardTrainingCurvesEnum,
    StandardTrainingScalarsEnum,
)
from torch.types import Device
from torch.utils.data import DataLoader, TensorDataset
from warg import GDKC

from architectures.adversarial_signal_classifier import AdversarialSignalClassifier
from configs.experiment_config import EXPERIMENTS
from configs.path_config import (
    BEST_VAL_MODEL_NAME,
    FINAL_MODEL_NAME,
    LOG_PATH,
    MODEL_PERSISTENCE_PATH,
    PROCESSED_FILE_ENDING,
)
from data import AdversarialSpeechBlockDataset
from pre.cepstral_spaces import CepstralSpaceEnum

__all__ = ["run_all_experiment_train"]


def train_asc(
    writer: TensorBoardPytorchWriter,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg_name: str,
    out_path: Path,
    input_size: Tuple,
    validation_interval: int,
    optimiser_specification: GDKC,
    num_epochs: int,
    device: Device = global_torch_device(),
) -> None:
    model = AdversarialSignalClassifier(*input_size).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimiser = optimiser_specification(model.parameters())
    best_val = math.inf
    for ith_epoch in progress_bar(
        range(num_epochs + 1), description=f"{cfg_name}", leave=False
    ):
        if ith_epoch % validation_interval == 0:
            writer.parameters(model, ith_epoch)
            loss_accum_val = 0.0
            accuracy_accum_val = 0.0
            ps = []
            ts = []
            with TorchEvalSession(model):
                with torch.no_grad():
                    for ith_batch, (predictors_val, category_val) in progress_bar(
                        enumerate(to_device_iterator(val_loader, device=device), 1),
                        description=f"Val Batch #",
                        leave=False,
                    ):
                        pred = model(predictors_val)
                        loss_accum_val += to_scalar(criterion(pred, category_val))
                        # accuracy_accum_val += accuracy_score(category.cpu().numpy(),(pred > 0.5).cpu().numpy().astype(numpy.int32))
                        pred_s = torch.sigmoid(pred)
                        accuracy_accum_val += to_scalar(
                            ((pred_s > 0.5) == category_val).float().sum()
                            / float(pred_s.shape[0])
                        )
                        ts.append(category_val)
                        ps.append(pred_s)

            val_loss = loss_accum_val / ith_batch
            writer.scalar(
                StandardTrainingScalarsEnum.validation_loss.value, val_loss, ith_epoch
            )
            writer.scalar(
                StandardTrainingScalarsEnum.validation_accuracy.value,
                accuracy_accum_val / ith_batch,
                ith_epoch,
            )
            writer.precision_recall_curve(
                StandardTrainingCurvesEnum.validation_precision_recall.value,
                torch.cat(ps, 0),
                torch.cat(ts, 0),
                ith_epoch,
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), str(out_path / BEST_VAL_MODEL_NAME))
                writer.scalar(
                    StandardTrainingScalarsEnum.new_best_model.value, 1, ith_epoch
                )
            else:
                writer.scalar(
                    StandardTrainingScalarsEnum.new_best_model.value, 0, ith_epoch
                )

        if ith_epoch < num_epochs:
            accum_loss = 0.0
            with TorchTrainSession(model):
                for ith_batch, (predictors, category) in progress_bar(
                    enumerate(to_device_iterator(train_loader, device=device), 1),
                    description=f"Train Batch #",
                ):
                    optimiser.zero_grad()  # Initialize the gradients to zero
                    loss = criterion(model(predictors), category)  # Forward
                    loss.backward()  # Back propagation
                    optimiser.step()  # Parameter update

                    accum_loss += to_scalar(loss)

            writer.scalar(
                StandardTrainingScalarsEnum.training_loss.value,
                accum_loss
                / ith_batch,  # ith_batch has the number of batches that was accumulated over
                ith_epoch,
            )

    torch.save(model.state_dict(), str(out_path / FINAL_MODEL_NAME))

    torch_clean_up()


def out_train_separate(
    *,
    load_time,
    exp_name,
    cepstral_name,
    ith_run,
    train_sets,
    val_sets,
    batch_size,
    val_interval,
    optimiser_spec,
    num_epochs,
):
    for tt in progress_bar(train_sets, description=f"{exp_name} train_set"):
        for (
            vt
        ) in (
            val_sets
        ):  # REDUDANT! assume to be equal length to train_sets, so will always be len(1)
            model_id_path = (
                Path(load_time)
                / f"{exp_name}"
                / f"{cepstral_name.value}"
                / f"seed_{ith_run}"
            )

            results_folder = ensure_existence(MODEL_PERSISTENCE_PATH / model_id_path)
            with WorkerSession(0) as num_workers:
                with WorkerSession(0) as num_workers_val:
                    with TensorBoardPytorchWriter(
                        LOG_PATH / model_id_path,
                        verbose=False,
                    ) as writer:
                        train_loader = DataLoader(
                            AdversarialSpeechBlockDataset(
                                tt.path
                                / f"{cepstral_name.value}_{tt.path.name}{PROCESSED_FILE_ENDING}",
                                split=SplitEnum.training,
                                shuffle_data=True,
                                train_percentage=tt.train_percentage,
                                test_percentage=tt.test_percentage,
                                validation_percentage=tt.validation_percentage,
                                random_seed=ith_run,
                                num_samples=tt.num_samples,
                            ),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                        )
                        val_loader = DataLoader(
                            AdversarialSpeechBlockDataset(
                                vt.path
                                / f"{cepstral_name.value}_{vt.path.name}{PROCESSED_FILE_ENDING}",
                                split=SplitEnum.validation,
                                train_percentage=vt.train_percentage,
                                test_percentage=vt.test_percentage,
                                validation_percentage=vt.validation_percentage,
                                random_seed=ith_run,
                                num_samples=vt.num_samples,
                            ),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers_val,
                            pin_memory=True,
                        )

                        train_asc(
                            writer,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            out_path=results_folder,
                            cfg_name=cepstral_name.value,
                            input_size=val_loader.dataset.__getitem__(0)[0].shape,
                            validation_interval=val_interval,
                            optimiser_specification=optimiser_spec,
                            num_epochs=num_epochs,
                        )


def out_train_merged(
    *,
    load_time,
    exp_name,
    cepstral_name,
    ith_run,
    train_sets: Iterable,
    val_sets: Iterable,
    batch_size,
    val_interval,
    optimiser_spec,
    num_epochs,
):
    """
    for training with merged datasets

    :param val_sets:
    :param load_time:
    :param exp_name:
    :param cepstral_name:
    :param ith_run:
    :param train_sets:
    :param batch_size:
    :param val_interval:
    :param learning_rate:
    :param adam_beta1:
    :param adam_beta2:
    :param epochs:
    :return:"""
    model_id_path = (
        Path(load_time) / exp_name / f"{cepstral_name.value}" / f"seed_{ith_run}"
    )
    results_folder = ensure_existence(MODEL_PERSISTENCE_PATH / model_id_path)

    with WorkerSession(0) as num_workers:
        with WorkerSession(0) as num_workers_val:
            with TensorBoardPytorchWriter(
                LOG_PATH / model_id_path, verbose=False
            ) as writer:
                predictors_train = []
                categories_train = []
                # train_names = []

                for t in progress_bar(train_sets, description=f"Train set #"):
                    (pt, ct, nt) = AdversarialSpeechBlockDataset(
                        t.path
                        / f"{cepstral_name.value}_{t.path.name}{PROCESSED_FILE_ENDING}",
                        split=SplitEnum.training,
                        train_percentage=t.train_percentage,
                        test_percentage=t.test_percentage,
                        validation_percentage=t.validation_percentage,
                        random_seed=ith_run,
                        num_samples=t.num_samples,
                        shuffle_data=True,
                    ).get_all_samples_in_split()
                    predictors_train.append(pt)
                    categories_train.append(ct)
                    # train_names.append(nt)

                predictors_train = numpy.concatenate(predictors_train)
                categories_train = numpy.concatenate(categories_train)
                # train_names = numpy.concatenate(train_names)

                predictors_val = []
                categories_val = []
                # val_names = []
                for t in progress_bar(val_sets, description=f"Val set #"):
                    (pv, cv, nv) = AdversarialSpeechBlockDataset(
                        t.path
                        / f"{cepstral_name.value}_{t.path.name}{PROCESSED_FILE_ENDING}",
                        split=SplitEnum.validation,
                        shuffle_data=True,
                        train_percentage=t.train_percentage,
                        test_percentage=t.test_percentage,
                        validation_percentage=t.validation_percentage,
                        random_seed=ith_run,
                        num_samples=t.num_samples,
                    ).get_all_samples_in_split()
                    predictors_val.append(pv)
                    categories_val.append(cv)
                    # val_names.append(nv)

                predictors_val = numpy.concatenate(predictors_val)
                categories_val = numpy.concatenate(categories_val)
                # val_names = numpy.concatenate(val_names)

                # assert categories_val.shape == categories_train.shape, f'Categories of train and val are not alike: {categories_val, categories_train}'
                assert min(categories_train) == min(
                    categories_val
                ), f"min not the same {min(categories_train), min(categories_val)}"
                assert max(categories_train) == max(
                    categories_val
                ), f"max not the same {max(categories_train), max(categories_val)}"

                predictors_train = to_tensor(
                    predictors_train[:, numpy.newaxis, ...], dtype=torch.float
                )  # Add channel for convolutions
                categories_train = to_tensor(
                    categories_train[:, numpy.newaxis], dtype=torch.float
                )  # Add channel for loss functions

                predictors_val = to_tensor(
                    predictors_val[:, numpy.newaxis, ...], dtype=torch.float
                )  # Add channel for convolutions
                categories_val = to_tensor(
                    categories_val[:, numpy.newaxis], dtype=torch.float
                )  # Add channel for loss functions

                train_loader = DataLoader(
                    TensorDataset(predictors_train, categories_train),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=global_pin_memory(num_workers),
                )
                val_loader = DataLoader(
                    TensorDataset(predictors_val, categories_val),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers_val,
                    pin_memory=global_pin_memory(num_workers_val),
                )

                *_, x_height, x_width = predictors_train.shape

                train_asc(
                    writer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    out_path=results_folder,
                    cfg_name=cepstral_name.value,
                    input_size=(1, x_height, x_width),
                    validation_interval=val_interval,
                    optimiser_specification=optimiser_spec,
                    num_epochs=num_epochs,
                )


def run_all_experiment_train(spaces: Iterable = CepstralSpaceEnum):
    info(f"torch.cuda.is_available{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = auto_select_available_cuda_device(2048)
    else:
        device = torch.device("cpu")

    global_torch_device(override=device)
    info(f"using device {global_torch_device()}")

    load_time = str(int(time.time()))

    from configs.training_config import COMMON_TRAINING_CONFIG

    for cepstral_name in progress_bar(spaces, description=f"# Cepstral"):
        for exp_name, exp_v in progress_bar(
            EXPERIMENTS, description=f"{cepstral_name}"
        ):
            for ith_run in progress_bar(
                range(COMMON_TRAINING_CONFIG.num_runs),
                description=f"{exp_name}, seed #",
            ):
                seed_stack(ith_run)
                for (k, v), (k2, v2) in zip(
                    exp_v.Train_Sets.items(), exp_v.Validation_Sets.items()
                ):
                    assert len(v) == len(v2)
                    model_name = f"{exp_name}_model_{k}_{k2}"
                    if len(v) > 1:
                        out_train_merged(
                            load_time=load_time,
                            exp_name=model_name,
                            cepstral_name=cepstral_name,
                            ith_run=ith_run,
                            train_sets=v,
                            val_sets=v2,
                            batch_size=COMMON_TRAINING_CONFIG.batch_size,
                            val_interval=COMMON_TRAINING_CONFIG.val_interval,
                            optimiser_spec=exp_v.Optimiser_Spec,
                            num_epochs=exp_v.Num_Epochs,
                        )
                    else:
                        out_train_separate(
                            load_time=load_time,
                            exp_name=model_name,
                            cepstral_name=cepstral_name,
                            ith_run=ith_run,
                            train_sets=v,
                            val_sets=v2,
                            batch_size=COMMON_TRAINING_CONFIG.batch_size,
                            val_interval=COMMON_TRAINING_CONFIG.val_interval,
                            optimiser_spec=exp_v.Optimiser_Spec,
                            num_epochs=exp_v.Num_Epochs,
                        )


if __name__ == "__main__":
    run_all_experiment_train(
        # spaces = (CepstralSpaceEnum.linear_fcc,)
    )
