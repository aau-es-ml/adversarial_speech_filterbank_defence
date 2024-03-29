from pathlib import Path
from shutil import copyfile
from typing import Iterable, Tuple

import numpy
from apppath import ensure_existence
from draugr.numpy_utilities import SplitEnum, SplitIndexer
from draugr.random_utilities import seed_stack
from draugr.visualisation import progress_bar, parallel_umap

from neodroidaudition.noise_generation.additive_noise import (
    compute_additive_noise_samples,
)
from scipy.io import wavfile

from configs.misc_config import SNR_RATIOS
from configs.path_config import (
    DATA_ROOT_NOISED_UNPROCESSED_PATH,
    DATA_ROOT_PATH,
    NOISES_SPLIT_UNPROCESSED_ROOT_PATH,
)
from data import AdversarialSpeechDataset
from external.rVADfast.rVADfast.rVAD_fast import get_rvad

RUN_PARALLEL = False  # REQUIRES LOTS OF RAM!!!!


def aug_func(
    benign_wav_file, packed: Tuple[Iterable[Path], Iterable[Path], Path, bool, bool]
):
    adv_files, noise_files, out_dir, parallel, skip_if_exists = packed

    clean_rvad_mask = get_rvad(*wavfile.read(benign_wav_file))
    if True:
        target = ensure_existence(out_dir / "no_aug" / "normal") / benign_wav_file.name
        if skip_if_exists and target.exists():
            pass
        else:
            copyfile(
                benign_wav_file,
                target,
            )

    if True:  # ENABLE!
        for noise_file in progress_bar(noise_files, disable=RUN_PARALLEL):
            compute_additive_noise_samples(
                voice_activity_mask=clean_rvad_mask,
                signal_file=benign_wav_file,
                out_dir=out_dir,
                category="normal",
                noise_file=noise_file,
                snrs=SNR_RATIOS,
            )

    if parallel:
        parallel_umap(
            adv_noise_aug_func,
            adv_files,
            func_kws=dict(
                noise_files=noise_files,
                normal_example_file=benign_wav_file,
                clean_rvad_mask=clean_rvad_mask,
                out_dir=out_dir,
            ),
        )
    else:
        for adv_example_file in progress_bar(
            adv_files, disable=RUN_PARALLEL
        ):  # Find adversarial examples matching original and use the clean rvad mask
            adv_noise_aug_func(
                adv_example_file,
                packed=(noise_files, benign_wav_file, clean_rvad_mask, out_dir),
            )


def adv_noise_aug_func(
    adv_example_file, packed: Tuple[Iterable[Path], Path, numpy.ndarray, Path]
) -> None:
    """
    SLOW AND INEFFICIENT!

    :param adv_example_file:
    :param packed:
    :return:"""
    noise_files, normal_example_file, clean_rvad_mask, out_dir = packed
    if (
        normal_example_file.name.split("-")[-1] in adv_example_file.name.split("-")[-1]
    ):  # Same id in name
        if True:
            copyfile(
                adv_example_file,
                ensure_existence(out_dir / "no_aug" / "adv") / adv_example_file.name,
            )

        if True:  # ENABLE!
            for noise_file in progress_bar(noise_files, disable=RUN_PARALLEL):
                compute_additive_noise_samples(
                    voice_activity_mask=clean_rvad_mask,
                    signal_file=adv_example_file,
                    out_dir=out_dir,
                    category="adv",
                    noise_file=noise_file,
                    snrs=SNR_RATIOS,
                )


def compute_noise_augmented_samples(
    parallel: bool = True, skip_if_exists: bool = True, subsets: SplitEnum = SplitEnum
):
    if True:
        for ss in progress_bar(("A",), disable=RUN_PARALLEL):
            data_path = DATA_ROOT_PATH / f"adversarial_dataset-{ss}"

            (
                benign_files,
                adv_files,
            ) = AdversarialSpeechDataset.get_wav_file_paths(data_path)
            out_dir = DATA_ROOT_NOISED_UNPROCESSED_PATH / ss

            seed_stack(0)

            benign_file_split_indexer = SplitIndexer(
                len(benign_files),
            )
            adv_file_split_indexer = SplitIndexer(len(adv_files))

            for split, benign_indices, adv_indices in progress_bar(
                zip(
                    SplitEnum,
                    benign_file_split_indexer.shuffled_indices().values(),
                    adv_file_split_indexer.shuffled_indices().values(),
                ),
                disable=RUN_PARALLEL,
            ):
                if split in subsets:
                    noise_files = list(
                        (NOISES_SPLIT_UNPROCESSED_ROOT_PATH / split.value).rglob(
                            "*.wav"
                        )
                    )
                    # noise_files = [nf for nf in noise_files if 'm_' in nf.name]  # TODO: Remove
                    # assert len(noise_files) == 2  # TODO: Remove
                    benign_files_split = numpy.array(benign_files)[benign_indices]
                    adv_files_split = numpy.array(adv_files)[adv_indices]
                    out_dir_split = out_dir / split.value

                    if parallel:
                        parallel_umap(
                            aug_func,
                            benign_files_split,
                            func_kws=dict(
                                adv_files=adv_files_split,
                                noise_files=noise_files,
                                out_dir=out_dir_split,
                                parallel=False,  # deamonic process cannot have children
                                skip_if_exists=skip_if_exists,
                            ),
                        )
                    else:
                        for benign_wav_file in progress_bar(
                            benign_files_split, disable=RUN_PARALLEL
                        ):
                            aug_func(
                                benign_wav_file,
                                packed=(
                                    adv_files_split,
                                    noise_files,
                                    out_dir_split,
                                    parallel,
                                    skip_if_exists,
                                ),
                            )
                else:
                    print(f"Skipping subset {split}")


if __name__ == "__main__":
    compute_noise_augmented_samples(
        parallel=RUN_PARALLEL,
        # subsets=(
        # SplitEnum.Training,
        # SplitEnum.Validation,
        # SplitEnum.Testing
    )
    # )
