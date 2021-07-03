#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import numpy
import torch
from apppath import ensure_existence
from draugr.numpy_utilities import Split
from draugr.random_utilities import seed_stack
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    auto_select_available_cuda_device,
    global_pin_memory,
    global_torch_device,
    to_device_iterator,
    to_tensor,
)
from draugr.tqdm_utilities import progress_bar
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    StyleSession,
    monochrome_line_no_marker_cycler,
    save_embed_fig,
    scatter_auto_mark,
)
from matplotlib import pyplot
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from warg import ContextWrapper, GDKC

from configs import EXPORT_RESULTS_PATH, MISC_CONFIG
from configs.experiment_config import EXPERIMENTS
from configs.path_config import (
    PROCESSED_FILE_ENDING,
    PROJECT_APP_PATH,
)
from data import AdversarialSpeechBlockDataset
from data.adversarial_speech_dataset import AdversarialSpeechDataset
from pre.cepstral_spaces import CepstralSpaceEnum

if __name__ == "__main__":

    def plot_tsne(verbose: bool = False, use_monochrome_style: bool = False):
        if torch.cuda.is_available():
            device = auto_select_available_cuda_device(2048)
        else:
            device = torch.device("cpu")

        global_torch_device(override=device)
        embedding_path = ensure_existence(EXPORT_RESULTS_PATH / "tsne")

        for cepstral_name in progress_bar(CepstralSpaceEnum, description="configs #"):
            for exp_name, exp_v in progress_bar(
                EXPERIMENTS, description=f"{cepstral_name}"
            ):
                if (
                    exp_name == "TRUNCATED_A" or exp_name == "TRUNCATED_B"
                ):  #'EQUAL_MAPPING_NO_AUG_to_NOISE_ALL_SNR':
                    # print(exp_name)
                    pass
                else:
                    # print(f'goahead {exp_name}')
                    continue

                with ContextWrapper(
                    TensorBoardPytorchWriter(
                        PROJECT_APP_PATH.user_log
                        / "NONE"
                        / exp_name
                        / f"{cepstral_name.value}",
                        verbose=True,
                    ),
                    False,
                ) as writer:
                    for k, t_ in exp_v.Train_Sets.items():
                        # if k != 'no_aug':
                        #  continue
                        # else:
                        #  pass
                        # print(k)
                        for t in t_:
                            seed_stack(0)

                            (
                                predictors,
                                categories,
                                nt,
                            ) = AdversarialSpeechBlockDataset(
                                t.path
                                / f"{cepstral_name.value}_{t.path.name}{PROCESSED_FILE_ENDING}",
                                split=Split.Training,
                                random_seed=0,
                                train_percentage=1.0,
                                test_percentage=0,
                                validation_percentage=0,
                                shuffle_data=True,
                                num_samples=None,
                            ).get_all_samples_in_split()

                            num_samples = (
                                MISC_CONFIG.projection_num_samples
                                if MISC_CONFIG.projection_num_samples
                                else len(predictors)
                            )
                            test_loader = DataLoader(
                                TensorDataset(
                                    to_tensor(predictors[:, numpy.newaxis, ...]),
                                    to_tensor(categories[:, numpy.newaxis]),
                                ),
                                batch_size=num_samples,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=global_pin_memory(0),
                            )

                            with torch.no_grad():
                                (predictor_, category_) = next(
                                    iter(to_device_iterator(test_loader, device=device))
                                )
                                with FigureSession():
                                    with ContextWrapper(
                                        GDKC(
                                            MonoChromeStyleSession,
                                            prop_cycler=monochrome_line_no_marker_cycler,
                                        ),
                                        True,
                                    ) if use_monochrome_style else StyleSession():
                                        x, y = zip(
                                            *TSNE(
                                                n_components=2,
                                                random_state=0,
                                                perplexity=MISC_CONFIG.tnse_perplexity,
                                                learning_rate=MISC_CONFIG.tsne_learning_rate,
                                            ).fit_transform(
                                                torch.flatten(predictor_, start_dim=1)
                                                .cpu()
                                                .numpy()
                                            )
                                        )
                                        """
                                        mask = category_.cpu().numpy() > 0
                                        scattr = scatter_auto_mark(
                                            x[mask],
                                            y[mask],
                                            c=category_.cpu().numpy()[mask],
                                            alpha=0.6,
                                            s=2.0,
                                            rasterized=True,
                                            cmap=pyplot.cm.coolwarm,
                                        )
                                        """

                                        scattr = scatter_auto_mark(
                                            x,
                                            y,
                                            c=category_.cpu().numpy(),
                                            alpha=0.6,
                                            s=40.0,
                                            linewidth=1,
                                            facecolor="none",
                                            markerfacecolor="none",
                                            rasterized=True,
                                            cmap=pyplot.cm.coolwarm,
                                        )
                                        """ # scatter_auto_mark broke legend code
                                        pyplot.legend(
                                            handles=scattr.legend_elements()[0],
                                            labels=[
                                                c.value
                                                if c.value != "normal"
                                                else "benign"
                                                for c in AdversarialSpeechDataset.DataCategories
                                            ],
                                        )
                                        """
                                        """
                    pyplot.title(f"{str(cepstral_name.value).replace('_',' ')} "
                                 f"{str(k).replace('_',' ')} {num_samples} samples")
                                 """
                                        save_embed_fig(
                                            embedding_path
                                            / f"{cepstral_name.value}_{k}.pdf"
                                        )
                                # writer.embed('TSNE', X_embedded, label_img=predictors2.to("cpu").numpy()) # BORKED!

            # break

    plot_tsne()
