#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from pathlib import Path

import librosa
import numpy
from apppath import ensure_existence, system_open_path
from draugr.scipy_utilities import read_normalised_wave, write_normalised_wave
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    StyleSession,
    fix_edge_gridlines,
    monochrome_line_no_marker_cycler,
    save_embed_fig,
)
from librosa.display import specshow
from matplotlib import pyplot
from warg import ContextWrapper, GDKC

from configs import DATA_ROOT_NOISED_UNPROCESSED_PATH, EXPORT_RESULTS_PATH

if __name__ == "__main__":

    def plot_signal(
        *,
        n_fft: int = 512,
        embedding_path: Path = ensure_existence(EXPORT_RESULTS_PATH / "rep"),
        use_mono_chrome_style: bool = False,
    ) -> None:
        r"""

        :param max_files:
        :param n_fcc:
        :param embedding_path:
        :param root_path:
        :type root_path:
        :param block_window_size_ms:
        :type block_window_size_ms:
        :param block_window_step_size_ms:
        :type block_window_step_size_ms:
        :param n_fft:
        :type n_fft:
        :param datasets:
        :type datasets:
        :param out_part_id:
        :type out_part_id:
        :return:
        :rtype:
        """

        # use_monochrome_style()
        adv_s = (
            "10db_adv",
            DATA_ROOT_NOISED_UNPROCESSED_PATH
            / "A"
            / "training"
            / "bbl_morten_SNR_10dB"
            / "adv"
            / "adv-short2short-000303.wav",
        )
        ben_s = (
            "10db_benign",
            DATA_ROOT_NOISED_UNPROCESSED_PATH
            / "A"
            / "training"
            / "bbl_morten_SNR_10dB"
            / "normal"
            / "sample-000303.wav",
        )

        postfix = "mono_chrome" if use_mono_chrome_style else "color"

        with ContextWrapper(
            GDKC(
                MonoChromeStyleSession,
                prop_cycler=monochrome_line_no_marker_cycler,
            ),
            True,
        ) if use_mono_chrome_style else StyleSession():
            n_fft_filters = n_fft  # fft length in the matlab mfcc function
            for id, file_ in (adv_s, ben_s):
                print(file_)
                try:
                    sampling_rate, wav_data = read_normalised_wave(file_)
                except Exception as e:
                    print(file_)
                    raise e

                parts = file_.stem.split("-")
                a = ensure_existence(
                    embedding_path
                    / f"{parts[-1]}"
                    / "raw"
                    / "noised"
                    / id
                    / f'{"_".join(parts[:-1])}'
                )
                a_path = a / f"{file_.stem}_all"
                with FigureSession():
                    pyplot.plot(wav_data)
                    fix_edge_gridlines()
                    pyplot.title(f"{str(file_.stem).replace('_', ' ')}.All")

                    save_embed_fig(f"{a_path}.pdf")
                    write_normalised_wave(f"{a_path}.wav", sampling_rate, wav_data)
                with FigureSession():
                    lin_spec_0 = librosa.stft(
                        wav_data,
                        n_fft=n_fft_filters,
                        hop_length=n_fft_filters // 2,
                        win_length=n_fft_filters,
                        window="hanning",
                    )

                    specshow(
                        # librosa.amplitude_to_db( numpy.abs(lin_spec_0,   ref=numpy.max  )),
                        librosa.power_to_db(numpy.abs(lin_spec_0) ** 2, ref=numpy.max),
                        y_axis="linear",
                        x_axis="time",
                        sr=sampling_rate,
                        hop_length=n_fft_filters // 2,
                        cmap=pyplot.rcParams["image.cmap"]
                        if use_mono_chrome_style
                        else "inferno",
                    )
                    pyplot.colorbar(format="%+2.0f dB")
                    pyplot.xlabel("Time (seconds)")
                    pyplot.ylabel("Frequency (Hz)")
                    pyplot.tight_layout()
                    save_embed_fig(f"{a_path}_librosa_spectrogram_{postfix}.pdf")
                    if False:
                        save_embed_fig(
                            f"{a_path}_librosa_spectrogram_{postfix}.svg", suffix=".svg"
                        )
            system_open_path(embedding_path, verbose=True)

    plot_signal()
