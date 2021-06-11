#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from pathlib import Path
from typing import Sequence

import librosa
import numpy
from apppath import ensure_existence, system_open_path
from draugr.scipy_utilities import read_normalised_wave, write_normalised_wave
from draugr.tqdm_utilities import progress_bar
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    SubplotSession,
    fix_edge_gridlines,
    monochrome_line_no_marker_cycler,
    save_embed_fig,
)
from librosa.display import specshow
from matplotlib import pyplot
from warg import ContextWrapper, GDKC

from configs import DATA_ROOT_PATH, EXPORT_RESULTS_PATH
from data import AdversarialSpeechDataset

if __name__ == "__main__":

    def plot_signal(
        root_path: Path,
        datasets: Sequence = (
            "adversarial_dataset-A",
            # "adversarial_dataset-B"
        ),
        out_part_id: Sequence = (
            "A",
            # "B"
        ),
        *,
        n_fcc: int = 20,
        block_window_size_ms: int = 512,  # 128
        block_window_step_size_ms: int = 512,  # 128
        n_fft: int = 512,
        export_path: Path = ensure_existence(EXPORT_RESULTS_PATH / "rep"),
        max_files: int = 9,  # <0 = Inf samples
    ) -> None:
        r"""

        :param max_files:
        :param n_fcc:
        :param export_path:
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

        with ContextWrapper(
            GDKC(
                MonoChromeStyleSession,
                prop_cycler=monochrome_line_no_marker_cycler,
            ),
            True,
        ):

            block_window_size = block_window_size_ms
            block_window_step_size = block_window_step_size_ms
            n_fft_filters = n_fft  # fft length in the matlab mfcc function

            for data_s, part_id in progress_bar(zip(datasets, out_part_id)):
                # print(data_s, part_id)
                normals, advs = AdversarialSpeechDataset.get_normal_adv_wav_file_paths(
                    root_path / data_s
                )

                file_paths = normals

                for (ith_file_idx, (file_, file_label)) in zip(
                    range(len(file_paths)),
                    progress_bar(zip(file_paths, categories), total=len(file_paths)),
                ):

                    try:
                        sampling_rate, wav_data = read_normalised_wave(file_)
                    except Exception as e:
                        print(file_)
                        raise e
                    data_len = len(wav_data)

                    parts = file_.stem.split("-")
                    a = ensure_existence(
                        export_path
                        / f"{parts[-1]}"
                        / "raw"
                        / f"{AdversarialSpeechDataset._categories[file_label]}"
                        / f'{"_".join(parts[:-1])}'
                    )
                    a_path = a / f"{file_.stem}_all"
                    with FigureSession():
                        pyplot.plot(wav_data)
                        fix_edge_gridlines()
                        pyplot.title(
                            f"{AdversarialSpeechDataset._categories[file_label]} {file_.stem}.All"
                        )

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
                            librosa.power_to_db(
                                numpy.abs(lin_spec_0) ** 2, ref=numpy.max
                            ),
                            y_axis="linear",
                            x_axis="time",
                            sr=sampling_rate,
                            hop_length=n_fft_filters // 2,
                            cmap="gray_r",
                        )
                        pyplot.colorbar(format="%+2.0f dB")
                        pyplot.xlabel("Time (seconds)")
                        pyplot.ylabel("Frequency (Hz)")
                        pyplot.tight_layout()
                        save_embed_fig(f"{a_path}_librosa_spectrogram.pdf")
                        save_embed_fig(
                            f"{a_path}_librosa_spectrogram.svg", suffix=".svg"
                        )
                    with SubplotSession(return_self=True) as sps:
                        img = specshow(
                            librosa.feature.mfcc(
                                y=wav_data,
                                sr=sampling_rate,
                                n_mfcc=n_fcc,
                                n_fft=n_fft_filters,
                                hop_length=n_fft_filters // 2,
                                win_length=n_fft_filters,
                            ),
                            sr=sampling_rate,
                            hop_length=n_fft_filters // 2,
                            x_axis="time",
                            ax=sps.axs[0],
                            cmap="gray_r",
                        )
                        sps.fig.colorbar(img, ax=sps.axs[0])
                        # sps.axs[0].set(title="MFCC")
                        save_embed_fig(f"{a_path}_librosa_mfcc.pdf")

                        # if ith_block > max_files - 1:
                        #  break
            system_open_path(export_path, verbose=True)

    plot_signal(
        DATA_ROOT_PATH,
        # ensure_existence(DATA_REGULAR_PROCESSED_PATH)
    )
