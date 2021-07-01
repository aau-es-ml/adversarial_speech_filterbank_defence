#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

from itertools import chain
from pathlib import Path

from apppath import ensure_existence, system_open_path
from draugr.scipy_utilities import read_normalised_wave
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    StyleSession,
    ltas_plot,
    monochrome_line_no_marker_cycler,
    save_embed_fig,
)
from matplotlib import pyplot
from warg import ContextWrapper, GDKC

from configs import (
    EXPORT_RESULTS_PATH,
    GENERATED_NOISES_UNPROCESSED_ROOT_PATH,
    MORTEN_NOISES,
)

if __name__ == "__main__":

    def plot_noise_psd(
        *,
        export_path: Path = ensure_existence(EXPORT_RESULTS_PATH / "rep"),
        use_mono_chrome_style: bool = False,
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
        ) if use_mono_chrome_style else StyleSession():

            file_paths = chain(
                GENERATED_NOISES_UNPROCESSED_ROOT_PATH.iterdir(),
                (
                    MORTEN_NOISES / "bbl" / "bbl_train.wav",
                    MORTEN_NOISES / "ssn" / "ssn_train.wav",
                ),
            )
            with FigureSession():
                for file_ in file_paths:
                    print(file_)
                    try:
                        sampling_rate, wav_data = read_normalised_wave(file_)
                    except Exception as e:
                        print(file_)
                        raise e

                    ltas_plot(
                        wav_data, sampling_rate, label=str(file_.name).replace("_", " ")
                    )

                a = ensure_existence(export_path / "raw" / "noise")
                a_path = a / f"noise_types_all"

                # pyplot.ylabel("Frequency (Hz)")
                # pyplot.xlabel("Amplitude")
                pyplot.legend()
                pyplot.tight_layout()
                save_embed_fig(f"{a_path}_psd.pdf")

                system_open_path(a_path, verbose=True)

    plot_noise_psd()
