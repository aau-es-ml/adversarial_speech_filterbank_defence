#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27-05-2021
           """

from pathlib import Path

from apppath import ensure_existence, system_open_path
from draugr.visualisation import StyleSession
from matplotlib import pyplot
from spafe.fbanks import bark_fbanks, gammatone_fbanks, linear_fbanks, mel_fbanks
from warg import ContextWrapper, GDKC
from configs import EXPORT_RESULTS_PATH
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    monochrome_line_no_marker_cycler,
)


def plot_filter_banks(path: Path, use_mono_chrome_style: bool = False):
    # init vars
    channels = 20
    n_fft = 512
    sampling_frequency = 16000
    low_freq = 0
    high_freq = None
    scale = "constant"

    mfcc_b = mel_fbanks.mel_filter_banks(
        nfilts=channels,
        nfft=n_fft,
        fs=sampling_frequency,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
    )
    gfcc_b = gammatone_fbanks.gammatone_filter_banks(
        nfilts=channels,
        nfft=n_fft,
        fs=sampling_frequency,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
    )
    lfcc_b = linear_fbanks.linear_filter_banks(
        nfilts=channels,
        nfft=n_fft,
        fs=sampling_frequency,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
    )
    bfcc_b = bark_fbanks.bark_filter_banks(
        nfilts=channels,
        nfft=n_fft,
        fs=sampling_frequency,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
    )
    _fbs = {
        "lfcc": lfcc_b,
        "mfcc": mfcc_b,
        "imfcc": mfcc_b[:, ::-1],
        "gfcc": gfcc_b,
        "igfcc": gfcc_b[:, ::-1],
        # 'bfcc': bfcc_b,
        # 'ibfcc':bfcc_b[:, ::-1],
    }
    ylabel, xlabel = "Gain", "Frequency (linear index 0 Hz to 8000 Hz)"

    for k, v in _fbs.items():
        with ContextWrapper(
            GDKC(
                MonoChromeStyleSession,
                prop_cycler=monochrome_line_no_marker_cycler,
            ),
            True,
        ) if use_mono_chrome_style else StyleSession():
            for fbank in v:
                pyplot.plot(fbank)
            pyplot.title(k)
            pyplot.ylabel(ylabel)
            pyplot.xlabel(xlabel)
            if False:
                pyplot.show()
            else:
                pyplot.savefig(str(path / f"{k}.pdf"))
                pyplot.close()


if __name__ == "__main__":
    export_path = ensure_existence(EXPORT_RESULTS_PATH / "filter_banks")
    plot_filter_banks(export_path)

    system_open_path(export_path, verbose=True)
