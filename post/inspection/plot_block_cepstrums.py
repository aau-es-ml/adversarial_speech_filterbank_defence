# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import sys
from itertools import chain
from pathlib import Path

sys.path.extend([str(Path.home() / "Projects" / "MyGithub" / "adversarial_speech")])

from pathlib import Path
from typing import Sequence

import librosa
from draugr.visualisation import save_embed_fig
from librosa.display import specshow
from matplotlib import pyplot

from torch.utils.data import DataLoader, TensorDataset
import numpy
import torch
from apppath import ensure_existence, system_open_path
from draugr.torch_utilities import (
    auto_select_available_cuda_device,
    global_pin_memory,
    global_torch_device,
    to_device_iterator,
    to_tensor,
)
from draugr.tqdm_utilities import progress_bar
from draugr.numpy_utilities import Split
from draugr.visualisation import SubplotSession
from warg import NOD

from configs.experiment_config import NO_AUG_TO_NOISE
from pre.cepstral_spaces import CepstralSpaceEnum, OtherSpacesEnum
from configs import (
    DATA_ROOT_PATH,
    PROCESSED_FILE_ENDING,
    EXPORT_RESULTS_PATH,
)
from data import AdversarialSpeechBlockDataset, AdversarialSpeechDataset
from draugr.visualisation import use_monochrome_style


__all__ = []

if __name__ == "__main__":

    def plot_rep(
        experiments,
        transformations: Sequence = (*CepstralSpaceEnum, OtherSpacesEnum.short_term_ft),
        embedding_path: Path = EXPORT_RESULTS_PATH / "rep",
    ) -> None:
        """
        always called after asc_plot_signals!

        :param experiments:
        :param transformations:
        :param embedding_path:
        :return:
        """
        use_monochrome_style()

        if torch.cuda.is_available():
            device = auto_select_available_cuda_device(2048)
        else:
            device = torch.device("cpu")
        global_torch_device(override=device)

        for transformation in progress_bar(transformations):
            for exp_name, exp_v in progress_bar(experiments):
                predictors = []
                categories = []
                test_names = []

                assert len(exp_v.Validation_Sets) > 0

                for k, t_ in progress_bar(
                    chain(
                        exp_v.Train_Sets.items(),
                        exp_v.Validation_Sets.items(),
                        exp_v.Test_Sets.items(),
                    )
                ):
                    t = t_[0]
                    asd = t.path / f"{transformation.value}_{k}{PROCESSED_FILE_ENDING}"
                    assert asd.exists(), asd
                    (pt, ct, nt,) = AdversarialSpeechBlockDataset.get_all_samples(
                        asd,
                        shuffle_data=False,
                        random_seed=42,
                    )

                    predictors.append(pt)
                    categories.append(ct)
                    test_names.append(nt)

                if len(predictors) > 0:
                    predictors = numpy.concatenate(predictors)
                    categories = numpy.concatenate(categories)
                    test_names = numpy.concatenate(test_names)

                    predictors = to_tensor(predictors[:, numpy.newaxis, ...])
                    categories = to_tensor(categories[:, numpy.newaxis])
                    indices = to_tensor(range(len(predictors)))

                    test_loader = DataLoader(
                        TensorDataset(predictors, categories, indices),
                        batch_size=1,
                        shuffle=False,  #
                        num_workers=0,
                        pin_memory=global_pin_memory(0),
                    )

                    rep_entries = [
                        str(c.name) for c in embedding_path.iterdir() if c.is_dir()
                    ]
                    assert len(rep_entries), "Did you run asc_plot_signals.py first?"

                    with torch.no_grad():
                        ds_ = AdversarialSpeechDataset(DATA_ROOT_PATH, Split.Testing)
                        sr = ds_.get_raw(0)[-1]  # 16000

                        for (predictors2, category, indices2) in progress_bar(
                            to_device_iterator(test_loader, device=device),
                            total=len(predictors),
                        ):  # Dumb way of selecting
                            idx = int(indices2.cpu()[0].item())
                            for names_id1 in progress_bar(rep_entries):
                                if names_id1 in test_names[idx]:
                                    # print(f'hit! {names_id1} {test_names[idx]}')
                                    with SubplotSession() as a:
                                        fig, (ax, *_) = a
                                        if (
                                            transformation
                                            == OtherSpacesEnum.short_term_ft
                                        ):
                                            """

                                            left = -(spec.shape[0] / sr) / 2
                                            right = (
                                                spec.shape[1] * 256 / sr
                                                + (spec.shape[0] / sr) / 2
                                            )
                                            lower = -sr / spec.shape[0]
                                            upper = sr / 2 + sr / spec.shape[0]
                                            # freqs = fftfreq(spec.shape[0], 1 / float(sr))[:(spec.shape[0] // 2)]
                                            # ax.yaxis.set_ticks(freqs[::16]) # TODO: FIX
                                            # ax.yaxis.set_ticklabels([f'{m:.0f}' for m in freqs[::16]])
                                            spec = (
                                                numpy.abs(predictors2.to("cpu").numpy()[0][0])
                                                ** 2
                                            )
                                            spec = spec[: (spec.shape[0] // 2)]
                                            stft_db = 10 * numpy.log10()

                                            pyplot.imshow(stft_db,
                                                          # fignum=0, #use current axis
                                                          origin="lower",
                                                          cmap="gray_r",
                                                          aspect="auto",
                                                          extent=[left, right, lower, upper],
                                                          )  # Spectrogram
                                            """

                                            spec = numpy.abs(
                                                predictors2.to("cpu").numpy()[0][0]
                                            )

                                            spec = spec[: (spec.shape[0] // 2)]
                                            specshow(
                                                librosa.amplitude_to_db(
                                                    spec, ref=numpy.max
                                                ),
                                                y_axis="linear",
                                                x_axis="time",
                                                sr=sr,
                                                hop_length=len(
                                                    spec
                                                ),  # half of fft length default
                                                cmap="gray_r",
                                            )

                                            ax.set_ylabel("Frequency [hz]")
                                            ax.set_xlabel("Time [Seconds]")
                                            cax = pyplot.colorbar(format="%+2.0f dB")
                                            # cax.set_label("Magnitude")
                                        elif (
                                            transformation == OtherSpacesEnum.power_spec
                                        ):
                                            """

                                            left = -(spec.shape[0] / sr) / 2
                                            right = (
                                                spec.shape[1] * 256 / sr
                                                + (spec.shape[0] / sr) / 2
                                            )
                                            lower = -sr / spec.shape[0]
                                            upper = sr / 2 + sr / spec.shape[0]
                                            # freqs = fftfreq(spec.shape[0], 1 / float(sr))[:(spec.shape[0] // 2)]
                                            # ax.yaxis.set_ticks(freqs[::16]) # TODO: FIX
                                            # ax.yaxis.set_ticklabels([f'{m:.0f}' for m in freqs[::16]])
                                            spec = (
                                                numpy.abs(predictors2.to("cpu").numpy()[0][0])
                                                ** 2
                                            )
                                            spec = spec[: (spec.shape[0] // 2)]
                                            stft_db = 10 * numpy.log10()

                                            pyplot.imshow(stft_db,
                                                          # fignum=0, #use current axis
                                                          origin="lower",
                                                          cmap="gray_r",
                                                          aspect="auto",
                                                          extent=[left, right, lower, upper],
                                                          )  # Spectrogram
                                            """

                                            spec = predictors2.to("cpu").numpy()[0][0]
                                            specshow(
                                                librosa.power_to_db(
                                                    spec, ref=numpy.max
                                                ),
                                                y_axis="linear",
                                                x_axis="time",
                                                sr=sr,
                                                hop_length=len(
                                                    spec
                                                ),  # half of fft length default
                                                cmap="gray_r",
                                            )

                                            ax.set_ylabel("Frequency [hz]")
                                            ax.set_xlabel("Time [Seconds]")
                                            cax = pyplot.colorbar(format="%+2.0f dB")
                                            # cax.set_label("Magnitude")
                                        else:
                                            img = specshow(
                                                predictors2.to("cpu").numpy()[0][0].T,
                                                x_axis="time",
                                                ax=ax,
                                                sr=sr,
                                                hop_length=256,  # half of fft length default
                                                # fignum=0,
                                                cmap="gray_r",
                                                # aspect="auto",
                                                # origin="lower",
                                            )
                                            ax.set_ylabel("Coefficients")
                                            # ax.set_xlabel("Time [ms]")
                                            fig.colorbar(img, ax=ax)

                                        pyplot.tight_layout()
                                        pyplot.title(
                                            f"{transformation.value} {test_names[idx]} {ds_.idx_to_str(categories[idx])}"
                                        )
                                        save_embed_fig(
                                            ensure_existence(
                                                embedding_path
                                                / names_id1
                                                / "trans"
                                                / test_names[idx].split(".")[0]
                                            )
                                            / f"{transformation.value}_{test_names[idx]}.pdf"
                                        )
                else:
                    raise FileNotFoundError

        system_open_path(embedding_path, verbose=True)

    plot_rep(
        NOD(
            # **TRUNCATED_SETS,
            # **MERGED_SETS,
            # **TRUNCATED_SPLITS,
            # **NOISED_SETS,
            **NO_AUG_TO_NOISE,
        ),
        # transformations=(          OtherSpacesEnum.short_term_ft,          CepstralSpaceEnum.linear_fcc,          ),
    )

    # system_open_path(EXPORT_RESULTS_PATH / "rep", verbose=True)
