#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import os
from pathlib import Path

from apppath import AppPath, ensure_existence, system_open_path
from configs.path_config import EXPORT_RESULTS_PATH
from draugr.tensorboard_utilities import TensorboardEventExporter
from draugr.tqdm_utilities import progress_bar
from draugr.writers import (
    TestingCurves,
    TestingScalars,
    TestingTables,
    TrainingCurves,
    TrainingScalars,
    TrainingTables,
)

__all__ = ["extract_scalars_as_csv", "extract_curve_tensors_as_csv", "extract_metrics"]


def extract_scalars_as_csv(
    train_path: Path = EXPORT_RESULTS_PATH / "csv" / "training",
    test_path: Path = EXPORT_RESULTS_PATH / "csv" / "testing",
    export_train: bool = True,
    export_test: bool = True,
    verbose: bool = False,
    only_extract_from_latest_event_file: bool = False,
) -> None:
    """
  :param train_path:
  :param test_path:
  :param export_train:
  :param export_test:
  :param verbose:
  :param only_extract_from_latest_event_file:

  """
    if only_extract_from_latest_event_file:
        max_load_time = max(
            list(
                AppPath(
                    "Adversarial Speech", "Christian Heider Nielsen"
                ).user_log.iterdir()
            ),
            key=os.path.getctime,
        )
        unique_event_files_parents = set(
            [ef.parent for ef in max_load_time.rglob("events.out.tfevents.*")]
        )
        event_files = {max_load_time: unique_event_files_parents}
    else:
        event_files = {
            a: set([ef.parent for ef in a.rglob("events.out.tfevents.*")])
            for a in list(
                AppPath(
                    "Adversarial Speech", "Christian Heider Nielsen"
                ).user_log.iterdir()
            )
        }

    for k, v in progress_bar(event_files.items()):
        for e in progress_bar(v):
            relative_path = e.relative_to(k)
            mapping_id, *rest = relative_path.parts
            with TensorboardEventExporter(e, save_to_disk=True) as tee:
                if export_test:

                    # model_id = relative_path.parent.name can be include but is always the same
                    out_tags = []
                    for tag in progress_bar(TestingScalars):
                        if tag.value in tee.available_scalars:
                            out_tags.append(tag.value)

                    if len(out_tags):
                        tee.scalar_export_csv(
                            *out_tags,
                            out_dir=ensure_existence(
                                test_path
                                / k.name
                                / mapping_id
                                / f"Test_{relative_path.name}"
                                / Path(*rest),
                                force_overwrite=True,
                                verbose=verbose,
                            ),
                        )
                        # print(e)
                    else:
                        if verbose:
                            print(
                                f"{e}, no requested tags found {TestingScalars.__members__.values()}, {tee.available_scalars}"
                            )

                if export_train:
                    out_tags = []
                    for tag in progress_bar(TrainingScalars):
                        if tag.value in tee.available_scalars:
                            out_tags.append(tag.value)

                    if len(out_tags):
                        tee.scalar_export_csv(
                            *out_tags,
                            out_dir=ensure_existence(
                                train_path / k.name / mapping_id / Path(*rest),
                                force_overwrite=True,
                                verbose=verbose,
                            ),
                        )
                    else:
                        if verbose:
                            print(
                                f"{e}, no requested tags found {TrainingScalars.__members__.values()}, {tee.available_scalars}"
                            )


def extract_table_tensors_as_csv(
    train_path: Path = EXPORT_RESULTS_PATH / "csv" / "training",
    test_path: Path = EXPORT_RESULTS_PATH / "csv" / "testing",
    export_train: bool = True,
    export_test: bool = True,
    verbose: bool = False,
    only_extract_from_latest_event_file: bool = False,
) -> None:
    """

  :param train_path:
  :param test_path:
  :param export_train:
  :param export_test:
  :param verbose:
  :param only_extract_from_latest_event_file:
  :return:
  """
    if only_extract_from_latest_event_file:
        max_load_time = max(
            list(
                AppPath(
                    "Adversarial Speech", "Christian Heider Nielsen"
                ).user_log.iterdir()
            ),
            key=os.path.getctime,
        )
        unique_event_files_parents = set(
            [ef.parent for ef in max_load_time.rglob("events.out.tfevents.*")]
        )
        event_files = {max_load_time: unique_event_files_parents}
    else:
        event_files = {
            a: set([ef.parent for ef in a.rglob("events.out.tfevents.*")])
            for a in list(
                AppPath(
                    "Adversarial Speech", "Christian Heider Nielsen"
                ).user_log.iterdir()
            )
        }

    for k, v in progress_bar(event_files.items()):
        for e in progress_bar(v):
            relative_path = e.relative_to(k)
            mapping_id, *rest = relative_path.parts

            # model_id = relative_path.parent.name can be include but is always the same

            with TensorboardEventExporter(e, save_to_disk=True) as tee:
                if export_test:
                    out_tags = []
                    for tag in progress_bar(TestingTables):
                        if tag.value in tee.available_tensors:
                            out_tags.append(tag.value)

                    if len(out_tags):
                        tee.pr_curve_export_csv(
                            *out_tags,
                            out_dir=ensure_existence(
                                test_path
                                / k.name
                                / mapping_id
                                / f"Test_{relative_path.name}"
                                / Path(*rest),
                                force_overwrite=True,
                                verbose=verbose,
                            ),
                        )
                    else:
                        if verbose:
                            print(
                                f"{e}, no requested tags found {TestingTables.__members__.values()}, {tee.available_tensors}"
                            )

                if export_train:  # TODO: OUTPUT for all epoch steps, no support yet
                    out_tags = []
                    for tag in progress_bar(TrainingTables):
                        if tag.value in tee.available_tensors:
                            out_tags.append(tag.value)

                    if len(out_tags):
                        tee.pr_curve_export_csv(
                            *out_tags,
                            out_dir=ensure_existence(
                                train_path / k.name / mapping_id / Path(*rest),
                                force_overwrite=True,
                                verbose=verbose,
                            ),
                        )
                    else:
                        if verbose:
                            print(
                                f"{e}, no requested tags found {TrainingTables.__members__.values()}, {tee.available_tensors}"
                            )


def extract_curve_tensors_as_csv(
    train_path: Path = EXPORT_RESULTS_PATH / "csv" / "training",
    test_path: Path = EXPORT_RESULTS_PATH / "csv" / "testing",
    export_train: bool = True,
    export_test: bool = True,
    verbose: bool = False,
    only_extract_from_latest_event_file: bool = False,
) -> None:
    """

  :param train_path:
  :param test_path:
  :param export_train:
  :param export_test:
  :param verbose:
  :param only_extract_from_latest_event_file:
  :return:
  """
    if only_extract_from_latest_event_file:
        max_load_time = max(
            list(
                AppPath(
                    "Adversarial Speech", "Christian Heider Nielsen"
                ).user_log.iterdir()
            ),
            key=os.path.getctime,
        )
        unique_event_files_parents = set(
            [ef.parent for ef in max_load_time.rglob("events.out.tfevents.*")]
        )
        event_files = {max_load_time: unique_event_files_parents}
    else:
        event_files = {
            a: set([ef.parent for ef in a.rglob("events.out.tfevents.*")])
            for a in list(
                AppPath(
                    "Adversarial Speech", "Christian Heider Nielsen"
                ).user_log.iterdir()
            )
        }

    for k, v in progress_bar(event_files.items()):
        for e in progress_bar(v):
            relative_path = e.relative_to(k)
            mapping_id, *rest = relative_path.parts

            # model_id = relative_path.parent.name can be include but is always the same

            with TensorboardEventExporter(e, save_to_disk=True) as tee:
                if export_test:
                    out_tags = []
                    for tag in progress_bar(TestingCurves):
                        if tag.value in tee.available_tensors:
                            out_tags.append(tag.value)

                    if len(out_tags):
                        tee.pr_curve_export_csv(
                            *out_tags,
                            out_dir=ensure_existence(
                                test_path
                                / k.name
                                / mapping_id
                                / f"Test_{relative_path.name}"
                                / Path(*rest),
                                force_overwrite=True,
                                verbose=verbose,
                            ),
                        )
                    else:
                        if verbose:
                            print(
                                f"{e}, no requested tags found {TestingCurves.__members__.values()}, {tee.available_tensors}"
                            )

                if export_train:  # TODO: OUTPUT for all epoch steps, no support yet
                    out_tags = []
                    for tag in progress_bar(TrainingCurves):
                        if tag.value in tee.available_tensors:
                            out_tags.append(tag.value)

                    if len(out_tags):
                        tee.pr_curve_export_csv(
                            *out_tags,
                            out_dir=ensure_existence(
                                train_path / k.name / mapping_id / Path(*rest),
                                force_overwrite=True,
                                verbose=verbose,
                            ),
                        )
                    else:
                        if verbose:
                            print(
                                f"{e}, no requested tags found {TrainingCurves.__members__.values()}, {tee.available_tensors}"
                            )


def extract_metrics(only_extract_latest=False):
    if True:
        extract_scalars_as_csv(only_extract_from_latest_event_file=only_extract_latest)
    if True:
        extract_curve_tensors_as_csv(
            only_extract_from_latest_event_file=only_extract_latest
        )


if __name__ == "__main__":
    extract_metrics(only_extract_latest=False)
    # extract_scalars_as_csv(verbose=False,export_train=False)
    system_open_path(EXPORT_RESULTS_PATH / "csv", verbose=True)