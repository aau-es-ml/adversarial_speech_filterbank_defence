#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19-10-2020
           """

import collections
import os
import pathlib
from enum import Enum
from typing import List, Sequence, Tuple

__all__ = [
    "AdversarialSpeechDataset",
    "misclassified_names",
]

import numpy
import torch

from draugr.numpy_utilities import SplitEnum, SplitIndexer
from draugr.torch_utilities import to_tensor
from draugr.random_utilities import seed_stack
from scipy.io import wavfile

from draugr.torch_utilities.datasets.supervised.categorical_dataset import (
    CategoricalDataset,
)
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


def misclassified_names(pred, truth, names, threshold: float = 0.5):
    return names[
        (1 * (pred > threshold) != truth).reshape(-1)
    ]  # Array shape (x,) instead of (x,1)


class AdversarialSpeechDataset(CategoricalDataset):
    """ """

    class AttackTypeEnum(Enum):
        WhiteBox = "A"
        BlackBox = "B"

    class DataCategories(Enum):
        Benign = "normal"
        Adversarial = "adversarial"

    _categories = OrderedSet([c.value for c in DataCategories])

    @property
    def categories(self) -> OrderedSet[str]:
        return AdversarialSpeechDataset._categories

    @property
    def response_shape(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        return self.__getitem__(0)[0].shape[1:]

    def __len__(self) -> int:
        return len(self._response)

    def get_raw(self, index):
        sampling_rate, wav_data = wavfile.read(self._file_names[index])
        return to_tensor((wav_data / 2.0**15).astype(float)), sampling_rate

    def get_tensor(self, index):
        raise NotImplementedError
        # wav_data, sampling_rate = load_wav(self._file_names[index])
        # return (wav_data / 2.0 ** 15).to(torch.float), sampling_rate

    def transforms(self):
        raise NotImplementedError
        # torchaudio.transforms.InverseMelScale()
        # torchaudio.transforms.MFCC()
        # torchaudio.transforms.Spectrogram()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self._getter(index)[0], self._response[index]

    @property
    def statistics(self) -> dict:
        return {
            "num_samples": len(self._response),
            "support": collections.Counter(numpy.array(self._response)),
        }

    @property
    def split(self) -> SplitEnum:
        return self._split

    def __init__(
        self,
        dataset_path: pathlib.Path,
        split: SplitEnum = SplitEnum.training,  # use None, for all data. However for the article results this was the
        # parameterisation
        # attack_type: AttackTypeEnum = AttackTypeEnum.WhiteBox,
        random_seed: int = 0,
        training: float = 0.7,
        validation: float = 0.2,
        testing: float = 0.1,
    ):
        """ """
        super().__init__()

        self._split = split
        self._dataset_path = dataset_path

        file_names, response = self.get_dataset_files_and_categories(self._dataset_path)
        if split:
            seed_stack(random_seed)
            split_by_p = SplitIndexer(len(file_names), training, validation, testing)
            ind = split_by_p.select_shuffled_split_indices(split)
            self._file_names = numpy.array(file_names)[ind]
            self._response = numpy.array(response)[ind]
        else:
            self._file_names = numpy.array(file_names)
            self._response = numpy.array(response)

        self._getter = self.get_raw
        # self._getter = self.get_tensor

    def get_all_samples_in_split(self) -> Tuple:
        return self._file_names, self._response

    @staticmethod
    def get_dataset_files_and_categories(
        root_dir: pathlib.Path,
    ) -> Tuple[Sequence, Sequence]:
        """

        wrapper that return a flattened file name list and a binary response vector [ 0 for non-adv, 1 for adv]

        :param root_dir:
        :return:"""
        (
            normal_files,
            adv_files,
        ) = AdversarialSpeechDataset.get_wav_file_paths(root_dir)
        file_names = normal_files + adv_files
        response = [0] * len(normal_files) + [1] * len(adv_files)

        return file_names, response

    @staticmethod
    def get_wav_file_paths(
        path: pathlib.Path, verbose: bool = False
    ) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
        """

        Specific to Zheng Hua Tan dataset

        :param verbose:
        :param path:
        :return:"""
        benign = []
        adv_files = []
        for wav_p in path.rglob("*.wav"):
            if (
                "Original-examples" in str(wav_p.parent)
                or "Original-Examples" in str(wav_p.parent)
                or "Normal-Examples" in str(wav_p.parent)
                or "normal" in str(wav_p.parent)
            ):
                benign.append(wav_p)
            else:
                if (
                    "adv-" in wav_p.name
                    or "adv" == wav_p.parent.name
                    or os.path.join("Adversarial-Examples", "Adversarial-Examples")
                    in str(wav_p)
                ):
                    adv_files.append(wav_p)
                else:
                    print(f"UNEXPECTED! {wav_p}, excluding")

        assert len(benign) > 0, f"no benign examples found"
        assert len(adv_files) > 0, f"no adversarial examples found"
        if verbose:
            print(path)
            print(f"num normal samples: {len(benign)}")
            print(f"num adversarial samples: {len(adv_files)}")
        return benign, adv_files


if __name__ == "__main__":
    from configs.experiment_config import DATA_ROOT_A_PATH, DATA_ROOT_PATH

    def main():
        ds = AdversarialSpeechDataset(DATA_ROOT_A_PATH)
        print(ds.response_shape)
        print(ds.predictor_shape)
        print(len(ds))
        print(ds.__getitem__(0))
        print(ds.statistics)

    def main2():
        ds = AdversarialSpeechDataset(DATA_ROOT_A_PATH)
        _ = ds.get_wav_file_paths(
            DATA_ROOT_PATH / "adversarial_dataset-A", verbose=True
        )
        _ = ds.get_wav_file_paths(
            DATA_ROOT_PATH / "adversarial_dataset-B", verbose=True
        )
        print(
            len(
                ds.get_dataset_files_and_categories(
                    DATA_ROOT_PATH / "adversarial_dataset-A"
                )
            )
        )

    main2()

    main()
