#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19-10-2020
           """

import collections
import pathlib
from enum import Enum
from logging import info
from random import seed, shuffle
from typing import Any, Sequence, Tuple

import numpy
import torch
from sklearn.model_selection import train_test_split

from data.adversarial_speech_dataset import AdversarialSpeechDataset
from data.persistence_helper import extract_tuple_from_path
from draugr.torch_utilities import to_tensor
from draugr.torch_utilities import CategoricalDataset
from draugr.numpy_utilities import Split
from draugr.tqdm_utilities import progress_bar
from warg import OrderedSet

"""
import torchaudio
from torchaudio import list_audio_backends
audio_be = list_audio_backends()
if 'sox-io' in audio_be:
  from torchaudio.backend.sox_io_backend import load_wav
elif 'soundfile' in audio_be:
  from torchaudio.backend.soundfile_backend import load_wav
"""

__all__ = ["AdversarialSpeechBlockDataset", "ReductionMethod"]


class ReductionMethod(Enum):
    mean = "mean"
    max = "max"


class AdversarialSpeechBlockDataset(CategoricalDataset):
    """ """

    @property
    def categories(self) -> OrderedSet[str]:
        return OrderedSet([c.value for c in AdversarialSpeechDataset.DataCategories])

    @property
    def response_shape(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        return self.__getitem__(0)[0].shape[1:]

    def __len__(self) -> int:
        return len(self._response)

    def __getitem__(self, index: int) -> Tuple:
        return self.getter(index)

    def getter_pred_resp(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            to_tensor(self._file_names[index], dtype=torch.float).unsqueeze(
                0
            ),  # Unsqueeze for convolution operation
            to_tensor(self._response[index], dtype=torch.float),
        )

    def getter_pred_resp_name(self, index) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return (
            to_tensor(self._file_names[index], dtype=torch.float).unsqueeze(
                0
            ),  # Unsqueeze for convolution operation
            to_tensor(self._response[index], dtype=torch.float),
            self._names[index],
        )

    @property
    def statistics(self) -> dict:
        return {
            "num_samples": len(self._response),
            "support": collections.Counter(numpy.array(self._response)),
        }

    def __init__(
        self,
        dataset_path: pathlib.Path,
        split: Split,
        *,
        num_samples: int = None,
        train_percentage: float = 0.7,
        test_percentage: float = 0.2,
        validation_percentage: float = 0.1,
        shuffle_data: bool = True,
        random_seed: int = 42,
        return_names: bool = False,
    ):
        """ """
        super().__init__()

        self._split = split
        self._dataset_path = dataset_path

        (
            (p_train, c_train, n_train),
            (p_test, c_test, n_test),
            (p_val, c_val, n_val),
        ) = self.get_processed_block_splits(
            dataset_path,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
            validation_percentage=validation_percentage,
            random_seed=random_seed,
            num_samples=num_samples,
            shuffle_data=shuffle_data,
        )

        if split == split.Training:
            self._file_names = p_train
            self._response = c_train
            self._names = n_train
        elif split == split.Testing:
            self._file_names = p_test
            self._response = c_test
            self._names = n_test
        elif split == split.Validation:
            self._file_names = p_val
            self._response = c_val
            self._names = n_val
        else:
            raise NotImplementedError

        info(f"{split} of {dataset_path} has {len(self._file_names)} samples")

        if return_names:
            self.getter = self.getter_pred_resp_name
        else:
            self.getter = self.getter_pred_resp

    def get_all_samples_in_split(self) -> Tuple[Sequence, Sequence, Sequence]:
        return (self._file_names, self._response, self._names)

    @staticmethod
    def get_all_samples(data_path, shuffle_data, random_seed) -> Tuple[Any, Any, Any]:
        x, category, names = extract_tuple_from_path(data_path)

        if shuffle_data:
            ind_list = [i for i in range(len(x))]
            seed(random_seed)
            shuffle(ind_list)
            return x[ind_list], category[ind_list], names[ind_list]
        else:
            return x, category, names

    @staticmethod
    def get_processed_block_splits(
        data_path: pathlib.Path,
        *,
        num_samples: int = None,
        train_percentage: float = 0.7,
        test_percentage: float = 0.2,
        validation_percentage: float = 0.1,
        shuffle_data: bool = False,
        random_seed: int = 42,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """

        :param num_samples:
        :param data_path:
        :param train_percentage:
        :param test_percentage:
        :param validation_percentage:
        :param shuffle_data:
        :param random_seed:
        :return:"""
        assert (
            train_percentage >= 0
            and test_percentage >= 0
            and validation_percentage >= 0
        )

        def get_block_indexes(file_indices: Sequence, selection: Sequence) -> Sequence:
            # Get the indices of all blocks that belong to each file in the set.
            block_indices = []
            for s in selection:
                file_start_idx = numpy.where(s == file_indices)[0][
                    0
                ]  # returns first occurrence idx
                block_indices.extend(
                    range(
                        file_indices[file_start_idx], file_indices[file_start_idx + 1]
                    )
                )
            return block_indices

        (
            predictors,
            response,
            file_names,
        ) = AdversarialSpeechBlockDataset.get_all_samples(
            data_path, shuffle_data, random_seed
        )

        num_predictors = len(predictors)
        if num_samples:
            assert (
                num_samples <= num_predictors
            ), f"{num_samples} samples were requested, but only {num_predictors} was available"
            predictors = predictors[:num_samples]
            response = response[:num_samples]
            file_names = file_names[:num_samples]
            info(f"Truncated to size {num_samples}")
        else:
            info(f"Using all {num_predictors} samples")

        if not numpy.isclose(
            train_percentage + test_percentage + validation_percentage, 1.0
        ):
            raise Exception(
                f"The split percentages must add up to 1.0 but was {train_percentage + test_percentage + validation_percentage:f}"
            )

        unique_file_indices = numpy.where(
            "block0" == numpy.array([name.split(".")[1] for name in file_names])
        )[
            0
        ]  # Return idx of occurrence, Get indices of the first block of each file. # has same length as number of files.

        response_file_indices = response[
            unique_file_indices
        ]  # Split unique array into train, test and val.

        if train_percentage == 1.0:
            train_file_indices = unique_file_indices
            test_file_indices = []
            validation_file_indices = []
        else:
            if 1.0 - train_percentage == 1.0:
                train_file_indices = []
                if 1.0 - validation_percentage == 1.0:
                    validation_file_indices = []
                    test_file_indices = unique_file_indices
                elif 1.0 - test_percentage == 1.0:
                    validation_file_indices = unique_file_indices
                    test_file_indices = []
                else:
                    validation_file_indices, test_file_indices, _, _ = train_test_split(
                        unique_file_indices,
                        response_file_indices,
                        test_size=test_percentage
                        / (test_percentage + validation_percentage),
                        random_state=random_seed,
                        stratify=response_file_indices,
                    )  # Each file is still in their own set
            else:
                (
                    train_file_indices,
                    test_file_indices,
                    _,
                    residual_response_unique,
                ) = train_test_split(
                    unique_file_indices,
                    response_file_indices,
                    test_size=1.0 - train_percentage,
                    random_state=random_seed,
                    stratify=response_file_indices,
                )  # Each file is in their own set
                validation_file_indices, test_file_indices, _, _ = train_test_split(
                    test_file_indices,
                    residual_response_unique,
                    test_size=test_percentage
                    / (test_percentage + validation_percentage),
                    random_state=random_seed,
                    stratify=residual_response_unique,
                )  # Each file is still in their own set

        exclusion_filter = [
            i for i, s in enumerate(file_names) if "remove" in s
        ]  # Remove the 'remove' labelled files from each subset.   # Specifically for speech-nonspeech test.
        file_indices = numpy.append(unique_file_indices, len(file_names))
        train_block_indices = get_block_indexes(
            file_indices, [i for i in train_file_indices if i not in exclusion_filter]
        )
        test_block_indices = get_block_indexes(
            file_indices, [i for i in test_file_indices if i not in exclusion_filter]
        )
        validation_block_indices = get_block_indexes(
            file_indices,
            [i for i in validation_file_indices if i not in exclusion_filter],
        )

        return (
            (
                predictors[train_block_indices],
                response[train_block_indices],
                file_names[train_block_indices],
            ),
            (
                predictors[test_block_indices],
                response[test_block_indices],
                file_names[test_block_indices],
            ),
            (
                predictors[validation_block_indices],
                response[validation_block_indices],
                file_names[validation_block_indices],
            ),
        )

    @staticmethod
    def combine_blocks_to_file(
        predictors: Sequence,
        response: Sequence,
        names: Sequence,
        *,
        reduction: ReductionMethod = ReductionMethod.mean,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        sort_idxs = numpy.argsort(names)
        predictors = predictors[sort_idxs]
        response = response[sort_idxs]
        names = names[sort_idxs].tolist()
        names_uniq, unique = numpy.unique(
            [name.split(".")[0] for name in names], return_index=True
        )

        unique = numpy.append(unique, len(names))

        predictors_per_file = []  # output 'y' per file
        response_per_file = []
        for i in progress_bar(range(len(unique) - 1), "block #"):
            if reduction == ReductionMethod.mean:
                predictors_per_file.append(
                    numpy.mean(predictors[unique[i] : unique[i + 1]])
                )
            elif reduction == ReductionMethod.max:
                predictors_per_file.append(
                    numpy.max(predictors[unique[i] : unique[i + 1]])
                )
            else:
                raise Exception("reduction method must be set to 'mean' or 'max'")
            response_per_file.append(response[unique[i]])

        return (
            numpy.array(predictors_per_file),
            numpy.array(response_per_file, dtype=numpy.int32),
            numpy.array(names_uniq),
        )


if __name__ == "__main__":

    def main():
        num_samples = 10000
        path = pathlib.Path(
            r"C:\Users\deter\AppData\Local\christian_heider_nielsen\adversarial_speech\processed\regular\A\gfcc_A.npz"
        )
        ds_validation = AdversarialSpeechBlockDataset(
            path, Split.Validation, shuffle_data=True, num_samples=num_samples
        )
        print(ds_validation.response_shape)
        print(ds_validation.predictor_shape)

        print(ds_validation.__getitem__(0))
        print(ds_validation.statistics)

        ds_testing = AdversarialSpeechBlockDataset(
            path, Split.Testing, shuffle_data=True, num_samples=num_samples
        )
        ds_training = AdversarialSpeechBlockDataset(
            path, Split.Training, shuffle_data=True, num_samples=num_samples
        )
        print(len(ds_validation))
        print(len(ds_testing))
        print(len(ds_training))

        a = next(iter(ds_testing))
        print(a[0].size(), a[1].shape)

        ds_trainsad = AdversarialSpeechBlockDataset(
            path, Split.Training, shuffle_data=True, num_samples=-1
        )

        test = False
        try:
            ds_trainsad = AdversarialSpeechBlockDataset(
                path, Split.Training, shuffle_data=True, num_samples=20000
            )
        except AssertionError:
            test = True
        assert test

    main()
