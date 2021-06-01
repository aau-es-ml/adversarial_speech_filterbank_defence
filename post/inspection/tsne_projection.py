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
    monochrome_line_no_marker_cycler,
    save_pdf_embed_fig,
)
from matplotlib import cm, pyplot
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

    def plot_tsne(verbose: bool = False):
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
                        for t in t_:
                            seed_stack(0)

                            (
                                predictors,
                                categories,
                                nt,
                            ) = AdversarialSpeechBlockDataset(
                                t.path
                                / f"{cepstral_name.value}_{k}{PROCESSED_FILE_ENDING}",
                                split=Split.Training,
                                random_seed=0,
                                train_percentage=1.0,
                                test_percentage=0,
                                validation_percentage=0,
                                shuffle_data=True,
                                num_samples=None,
                            ).get_all_samples_in_split()

                            test_loader = DataLoader(
                                TensorDataset(
                                    to_tensor(predictors[:, numpy.newaxis, ...]),
                                    to_tensor(categories[:, numpy.newaxis]),
                                ),
                                batch_size=MISC_CONFIG.projection_num_samples,
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
                                    ):
                                        scattr = pyplot.scatter(
                                            *zip(
                                                *TSNE(
                                                    n_components=2,
                                                    random_state=0,
                                                    perplexity=MISC_CONFIG.tnse_perplexity,
                                                    learning_rate=MISC_CONFIG.tsne_learning_rate,
                                                ).fit_transform(
                                                    torch.flatten(
                                                        predictor_, start_dim=1
                                                    )
                                                    .cpu()
                                                    .numpy()
                                                )
                                            ),
                                            c=category_.cpu().numpy(),
                                            alpha=0.6,
                                            s=2.0,
                                            cmap=cm.PuOr,
                                        )
                                        pyplot.legend(
                                            handles=scattr.legend_elements()[0],
                                            labels=[
                                                c.value
                                                for c in AdversarialSpeechDataset.DataCategories
                                            ],
                                        )
                                        pyplot.title(f"{cepstral_name.value} {k}")
                                        save_pdf_embed_fig(
                                            embedding_path
                                            / f"{cepstral_name.value}_{k}.pdf"
                                        )
                                # writer.embed('TSNE', X_embedded, label_img=predictors2.to("cpu").numpy()) # BORKED!

            # break

    plot_tsne()
