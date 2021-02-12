from pathlib import Path
from shutil import copyfile
from typing import Iterable, Sequence, Tuple

import numpy
from neodroidaudition.noise_generation.additive_noise import (
    compute_additive_noise_samples,
)
from scipy.io import wavfile

from apppath import ensure_existence
from configs.path_config import (
    DATA_ROOT_NOISED_UNPROCESSED_PATH,
    DATA_ROOT_PATH,
    NOISES_SPLIT_UNPROCESSED_ROOT_PATH,
)
from data import AdversarialSpeechDataset
from draugr.numpy_utilities import Split, SplitIndexer
from draugr.random_utilities import seed_stack
from draugr.tqdm_utilities import progress_bar, parallel_umap

from external.rVADfast.rVADfast.rVAD_fast import get_rvad


def aug_func(
    normal_example_file, packed: Tuple[Iterable[Path], Iterable[Path], Path, bool]
):
    adv_files, noise_files, out_dir, parallel = packed

    clean_rvad_mask = get_rvad(*wavfile.read(normal_example_file))
    if True:
        copyfile(
            normal_example_file,
            ensure_existence(out_dir / "no_aug" / "normal") / normal_example_file.name,
        )

    if True:  # ENABLE!
        for noise_file in progress_bar(noise_files):
            compute_additive_noise_samples(
                voice_activity_mask=clean_rvad_mask,
                signal_file=normal_example_file,
                out_dir=out_dir,
                category="normal",
                noise_file=noise_file,
            )

    if parallel:
        parallel_umap(
            inner_noise_aug_func,
            adv_files,
            func_kws=dict(
                noise_files=noise_files,
                normal_example_file=normal_example_file,
                clean_rvad_mask=clean_rvad_mask,
                out_dir=out_dir,
            ),
        )
    else:
        for adv_example_file in progress_bar(
            adv_files
        ):  # Find adversarial examples matching original and use the clean rvad mask
            inner_noise_aug_func(
                adv_example_file,
                packed=(noise_files, normal_example_file, clean_rvad_mask, out_dir),
            )


def inner_noise_aug_func(
    adv_example_file, packed: Tuple[Iterable[Path], Path, numpy.ndarray, Path]
) -> None:
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
            for noise_file in progress_bar(noise_files):
                compute_additive_noise_samples(
                    voice_activity_mask=clean_rvad_mask,
                    signal_file=adv_example_file,
                    out_dir=out_dir,
                    category="adv",
                    noise_file=noise_file,
                )


def compute_noise_augmented_samples(parallel: bool = True):
    if True:
        for ss in ("A",):
            data_path = DATA_ROOT_PATH / f"adversarial_dataset-{ss}"

            (
                normal_files,
                adv_files,
            ) = AdversarialSpeechDataset.get_normal_adv_wav_files(data_path)
            out_dir = DATA_ROOT_NOISED_UNPROCESSED_PATH / ss

            seed_stack(0)

            normal_file_split_indexer = SplitIndexer(len(normal_files))
            adv_file_split_indexer = SplitIndexer(len(adv_files))

            for (split, nf_indices, af_indices) in zip(
                Split,
                normal_file_split_indexer.shuffled_indices().values(),
                adv_file_split_indexer.shuffled_indices().values(),
            ):
                noise_files = list(
                    (NOISES_SPLIT_UNPROCESSED_ROOT_PATH / split.value).rglob("*.wav")
                )
                # noise_files = [nf for nf in noise_files if 'm_' in nf.name]  # TODO: Remove
                # assert len(noise_files) == 2  # TODO: Remove
                normal_files_split = numpy.array(normal_files)[nf_indices]
                adv_files_split = numpy.array(adv_files)[af_indices]
                out_dir_split = out_dir / split.value

                if parallel:
                    parallel_umap(
                        aug_func,
                        normal_files_split,
                        func_kws=dict(
                            adv_files=adv_files_split,
                            noise_files=noise_files,
                            out_dir=out_dir_split,
                            parallel=False,  # deamonic process cannot have children
                        ),
                    )
                else:
                    for normal_example_file in progress_bar(normal_files_split):
                        aug_func(
                            normal_example_file,
                            packed=(
                                adv_files_split,
                                noise_files,
                                out_dir_split,
                                parallel,
                            ),
                        )


if __name__ == "__main__":

    compute_noise_augmented_samples()
