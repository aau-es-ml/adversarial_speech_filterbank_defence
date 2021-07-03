#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-06-2021
           """

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from functools import reduce
from pathlib import Path
from typing import Sequence

import numpy
from apppath import ensure_existence, system_open_path
from draugr.python_utilities import next_pow_2
from draugr.scipy_utilities import read_normalised_wave
from draugr.tqdm_utilities import progress_bar
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    StyleSession,
    fix_edge_gridlines,
    ltas_plot,
    monochrome_line_no_marker_cycler,
    save_embed_fig,
)
from matplotlib import pyplot
from scipy.signal import spectrogram
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
        block_window_size_ms: int = 512,  # 128
        block_window_step_size_ms: int = 512,  # 128
        n_fft: int = 512,
        embedding_path: Path = ensure_existence(EXPORT_RESULTS_PATH / "rep"),
        max_files: int = 20,  # <0 = Inf samples
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

        with ContextWrapper(
            GDKC(
                MonoChromeStyleSession,
                prop_cycler=monochrome_line_no_marker_cycler,
            ),
            True,
        ) if use_mono_chrome_style else StyleSession():

            block_window_size = block_window_size_ms
            block_window_step_size = block_window_step_size_ms
            n_fft_filters = n_fft  # fft length in the matlab mfcc function

            plots = {}

            for data_s, part_id in progress_bar(zip(datasets, out_part_id)):
                # print(data_s, part_id)
                normals, advs = AdversarialSpeechDataset.get_wav_file_paths(
                    root_path / data_s
                )

                num_samples_each = max_files // 2
                normals = [item for item in normals if "short-signals" in str(item)]
                advs = [
                    item
                    for item in advs
                    if "short-signals" in str(item) and "adv-short-target" in str(item)
                ]

                file_paths = []
                categories = []

                file_paths += advs[:num_samples_each]
                categories += [1] * num_samples_each

                file_paths += normals[:num_samples_each]
                categories += [0] * num_samples_each

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

                    block_window_size_ms = (
                        block_window_size * sampling_rate
                    ) // 1000  # window size in mS
                    block_step_size_ms = (
                        block_window_step_size * sampling_rate
                    ) // 1000

                    if (
                        block_window_size_ms >= data_len
                        or block_step_size_ms >= data_len
                        or n_fft_filters >= data_len
                    ):
                        if block_window_size_ms >= data_len:
                            print("full size is reached for window")
                        if block_step_size_ms >= data_len:
                            print("full size is reached for step")
                        if n_fft_filters >= data_len:
                            print("to bad...")
                    else:
                        parts = file_.stem.split("-")
                        a = ensure_existence(embedding_path / f"{parts[-1]}" / "psd")
                        a_path = a / f"{file_.stem}_all"

                        if parts[-1] not in plots:
                            plots[parts[-1]] = {}

                        plots[parts[-1]][
                            AdversarialSpeechDataset._categories[file_label]
                        ] = [wav_data, sampling_rate, a_path, {"_".join(parts[:-1])}]

            for ko, a in plots.items():
                with FigureSession():
                    p = None
                    # id_o = None
                    asd = []
                    for k, (b, sr, a_p, id_a) in a.items():
                        if k == "normal":
                            k = "benign"
                            z = 1
                        else:
                            z = 2
                        ltas_plot(b, sr, label=k, zorder=z)
                        p = a_p
                        # id_o=id_a

                        f, t, frames = spectrogram(
                            b,
                            sampling_rate,
                            nperseg=next_pow_2(sampling_rate * (20 / 1000)),
                            scaling="spectrum",
                        )
                        # f, spectrum = welch(                b, sampling_rate, window="hanning", nperseg=next_pow_2(
                        # sampling_rate * (20 / 1000)                    )  # 20 ms, next_pow_2 per seg == n_fft
                        # , scaling="spectrum"                )
                        asd.append(numpy.sqrt(frames))

                    asdsafas = numpy.abs(reduce(numpy.subtract, asd))

                    ax1 = pyplot.gca()
                    ax2 = ax1.twinx()
                    ax1.set_zorder(ax2.get_zorder() + 1)
                    ax2.plot(
                        f,
                        numpy.mean(asdsafas, -1),
                        color="0.6",
                        label="diff",
                        # zorder=1
                    )
                    fix_edge_gridlines()
                    ax1.legend(loc="upper center")
                    ax2.set_ylabel("Average absolute difference", color="0.6")

                    pyplot.title(f"{file_.stem}")
                    save_embed_fig(f"{p}_diff_average.pdf")
                    # with SubplotSession(return_self=True) as sps:
                    #  ltas_plot(wav_data,sampling_rate)
                    #  save_embed_fig(f"{a_path}_ltass.pdf")

            system_open_path(embedding_path, verbose=True)

    plot_signal(DATA_ROOT_PATH)
