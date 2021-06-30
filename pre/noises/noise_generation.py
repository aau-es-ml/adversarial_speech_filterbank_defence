#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19-01-2021
           """

from pathlib import Path

import torchaudio
from apppath import ensure_existence
from draugr.torch_utilities import to_tensor
from neodroidaudition.data.recognition.libri_speech import LibriSpeech
from neodroidaudition.noise_generation import (
    generate_babble_noise,
    generate_speech_shaped_noise,
)

from configs.path_config import (
    GENERATED_NOISES_UNPROCESSED_ROOT_PATH,
)


def generate_ssn(num_samples: int = 20):
    d_male = iter(
        LibriSpeech(
            path=Path.home() / "Data" / "Audio" / "Speech" / "LibriSpeech",
            custom_subset=LibriSpeech.CustomSubsets.male,
        )
    )
    d_female = iter(
        LibriSpeech(
            path=Path.home() / "Data" / "Audio" / "Speech" / "LibriSpeech",
            custom_subset=LibriSpeech.CustomSubsets.female,
        )
    )
    male_unique = {}
    while len(male_unique) < num_samples // 2:
        s = next(d_male)
        speaker_id = s[-3]
        if speaker_id not in male_unique:
            male_unique[speaker_id] = s

    female_unique = {}
    while len(female_unique) < num_samples // 2:
        s = next(d_female)
        speaker_id = s[-3]
        if speaker_id not in female_unique:
            female_unique[speaker_id] = s

    unique = (*male_unique.values(), *female_unique.values())

    files, sr = zip(*[(v[0].numpy(), v[1]) for _, v in zip(range(num_samples), unique)])
    assert all([sr[0] == s for s in sr[1:]])
    sr = sr[0]

    noise = generate_speech_shaped_noise(files, sr, long_term_avg=True)[0]

    torchaudio.save(
        str(
            ensure_existence(GENERATED_NOISES_UNPROCESSED_ROOT_PATH)
            / f"{num_samples // 2}m_{num_samples // 2}f_ssn.wav"
        ),
        to_tensor(noise),
        sr,
    )


def generate_babble(num_samples=10):
    d_male = iter(
        LibriSpeech(
            path=Path.home() / "Data" / "Audio" / "Speech" / "LibriSpeech",
            custom_subset=LibriSpeech.CustomSubsets.male,
        )
    )
    d_female = iter(
        LibriSpeech(
            path=Path.home() / "Data" / "Audio" / "Speech" / "LibriSpeech",
            custom_subset=LibriSpeech.CustomSubsets.female,
        )
    )
    male_unique = {}
    while len(male_unique) < num_samples // 2:
        s = next(d_male)
        speaker_id = s[-3]
        if speaker_id not in male_unique:
            male_unique[speaker_id] = s

    female_unique = {}
    while len(female_unique) < num_samples // 2:
        s = next(d_female)
        speaker_id = s[-3]
        if speaker_id not in female_unique:
            female_unique[speaker_id] = s

    unique = (*male_unique.values(), *female_unique.values())

    files, sr = zip(*[(v[0].numpy(), v[1]) for _, v in zip(range(num_samples), unique)])
    assert all([sr[0] == s for s in sr[1:]])
    babble = generate_babble_noise(
        files,
        sr[0],
        export_path=ensure_existence(GENERATED_NOISES_UNPROCESSED_ROOT_PATH)
        / f"{num_samples // 2}m_{num_samples // 2}f_babble.wav",
    )


if __name__ == "__main__":
    generate_ssn()
    generate_babble()
