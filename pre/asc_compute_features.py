from pathlib import Path
from typing import Sequence, Tuple

import numpy
from draugr.scipy_utilities import read_normalised_wave

from apppath import ensure_existence
from configs.experiment_config import (
    DATA_NOISED_PROCESSED_PATH,
    DATA_REGULAR_PROCESSED_PATH,
    DATA_SS_SPLITS_PROCESSED_PATH,
)
from configs.path_config import (
    DATA_ROOT_NOISED_UNPROCESSED_PATH,
    DATA_ROOT_PATH,
    DATA_ROOT_SS_SPLITS_UNPROCESSED_PATH,
)
from data import AdversarialSpeechDataset
from data.persistence_helper import export_to_path
from draugr.matlab_utilities import start_engine

__all__ = []

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
IMPORTANT! matlab scripts uses 'buffer' which requires Signal Processing Toolbox of Matlab!
and 'trimf' requires Fuzzy Logic Toolbox.


VERY SLOW! 

TODO: parallelise
"""

from draugr.tqdm_utilities import progress_bar

from pre.asc_transformation_spaces import OtherSpacesEnum, CepstralSpaceEnum
from pre.feature_extraction.matlab_extractor import cepstral_extractor

MATLAB_ENGINE = start_engine()
matlab_files_path = str(
    # Path.cwd()
    Path(__file__).parent
    / "feature_extraction"
    / "matlab_implementation"
    / "matlab_code"
)


def file_wise_feature_extraction(
    function: CepstralSpaceEnum,
    source_path: Path,
    *,
    save_to_disk: bool,
    out_path: Path,
    out_id: str,
    n_fcc: int,
    cepstral_window_length_ms: int,  # window length in the matlab mfcc function
    n_fft: int = 512,  # fft length in the matlab mfcc function
    max_files: int = 0,  # <0 = Inf samples
    min_trim: bool = False,  # Else max pad
) -> Tuple:
    r"""
!!! UNUSED # NOT FINISHED!!!
:param min_trim:
:param max_files:
:param function:
:type function:
:param source_path:
:type source_path:
:param save_to_disk:
:type save_to_disk:
:param out_path:
:type out_path:
:param out_id:
:type out_id:
:param n_fcc:
:type n_fcc:
:param cepstral_window_length_ms:
:type cepstral_window_length_ms:
:param n_fft:
:type n_fft:
:return:
:rtype:"""

    if False:  # Get engine in each process, For multi process setup, not used fully yet
        matlab_files_path = str(Path.cwd() / "matlab_code")
        print(f"Cd.ing to: {matlab_files_path}")
        MATLAB_ENGINE.cd(matlab_files_path)

    file_paths, categories = get_dataset_files_and_categories(source_path)

    blocks_total = 0
    labels_split = []
    block_sample_ids = []
    all_cepstral_blocks = []

    for (ith_file_idx, (file_, file_label)) in progress_bar(
        zip(range(len(file_paths)), zip(file_paths, categories)),
        total=len(file_paths),
        auto_describe_iterator=False,
    ):
        if max_files and ith_file_idx >= max_files:
            break

        sampling_rate, wav_data = read_normalised_wave(file_)
        data_len = len(wav_data)

        if n_fft >= data_len:
            if n_fft >= data_len:
                print(f"to short... skipping {file_}")
        else:

            labels_split.append(file_label)
            block_sample_ids.append(f"{file_.stem}.block{0}")

            blocks_total += 1  # +1 to get the last part
            all_cepstral_blocks.append(
                cepstral_extractor(
                    function,
                    wav_data,
                    matlab_engine_=MATLAB_ENGINE,
                    sample_rate=sampling_rate,
                    cepstral_window_length_ms=cepstral_window_length_ms,
                    num_fft=n_fft,
                    num_fcc=n_fcc,
                )
            )  # gather all files in one list

            if ith_file_idx < 1:
                print(
                    f"data_len: {data_len}, num blocks for file_idx#{ith_file_idx}: {1}, cepstral shape: {all_cepstral_blocks[0].shape}"
                )

    sample_blocks = [numpy.asarray(sample) for sample in all_cepstral_blocks]

    if min_trim:
        trim = min([s.shape[-1] for s in sample_blocks])  # Trim to shortest sample
        features = numpy.array([s[..., :trim] for s in sample_blocks])
    else:  # Then max pad
        max_pad = max([s.shape[-1] for s in sample_blocks])
        features = numpy.array(
            [
                numpy.pad(
                    s,
                    ((0, 0), (0, max_pad - s.shape[-1])),
                    mode="constant",
                    constant_values=(0, 0),
                )
                for s in sample_blocks
            ]
        )

    features = features.transpose((0, 2, 1))
    category = numpy.asarray(labels_split)
    assert len(block_sample_ids) == len(sample_blocks)

    if save_to_disk:
        ensure_existence(out_path)
        out_file = out_path / f"{function.value}_{out_id}"
        print(f"Saving {out_file}")
        export_to_path(out_file, features, category, block_sample_ids)

    return features, category


def block_wise_feature_extraction(
    function: CepstralSpaceEnum,
    source_path: Path,
    *,
    save_to_disk: bool,
    out_path: Path,
    out_id: str,
    block_window_size: int,
    block_window_step_size: int,
    n_fcc: int,
    cepstral_window_length_ms: int,  # window length in the matlab mfcc function
    n_fft: int = 512,  # fft length in the matlab mfcc function
    max_files: int = 0,  # <0 = Inf samples
    verbose: bool = False,
) -> Tuple:
    r"""

:param function:
:param source_path:
:param save_to_disk:
:param block_window_size:
:param block_window_step_size:
:param out_path:
:param out_id:
:param n_fcc:
:param cepstral_window_length_ms:
:param n_fft:
:param max_files:
:param verbose:
:return:"""
    if True:

        print(f"Cd.ing to: {matlab_files_path}")
        MATLAB_ENGINE.cd(matlab_files_path)

    file_paths, categories = AdversarialSpeechDataset(
        source_path
    ).get_all_samples_in_split()

    blocks_total = 0
    responses = []
    block_sample_ids = []
    all_cepstral_blocks = []

    for (ith_file_idx, (file_, file_label)) in progress_bar(
        zip(range(len(file_paths)), zip(file_paths, categories)),
        total=len(file_paths),
        auto_describe_iterator=False,
    ):
        if max_files and ith_file_idx >= max_files:
            break

        try:
            sampling_rate, wav_data = read_normalised_wave(file_)
        except Exception as e:
            print(file_)
            raise e
        data_len = len(wav_data)

        block_window_size_ms = (
            block_window_size * sampling_rate
        ) // 1000  # window size in mS
        block_step_size_ms = (block_window_step_size * sampling_rate) // 1000

        if (
            block_window_size_ms >= data_len
            or block_step_size_ms >= data_len
            or n_fft >= data_len
        ):
            if block_window_size_ms >= data_len:
                print("full size is reached for window")
            if block_step_size_ms >= data_len:
                print("full size is reached for step")
            if n_fft >= data_len:
                print("to bad...")
        else:
            file_cepstral_blocks = []
            num_blocks: int = (data_len - block_window_size_ms) // block_step_size_ms

            for ith_block in progress_bar(
                range(num_blocks), auto_describe_iterator=False
            ):
                file_cepstral_blocks.append(
                    cepstral_extractor(
                        function,
                        wav_data[
                            ith_block
                            * block_step_size_ms : ith_block
                            * block_step_size_ms
                            + block_window_size_ms
                        ],  # split data into blocks of window size
                        matlab_engine_=MATLAB_ENGINE,
                        sample_rate=sampling_rate,
                        cepstral_window_length_ms=cepstral_window_length_ms,
                        num_fft=n_fft,
                        num_fcc=n_fcc,
                    )
                )
                responses.append(file_label)
                block_sample_ids.append(f"{file_.stem}.block{ith_block}")

            file_cepstral_blocks.append(
                cepstral_extractor(
                    function,
                    wav_data[data_len - block_window_size_ms :],
                    matlab_engine_=MATLAB_ENGINE,
                    sample_rate=sampling_rate,
                    cepstral_window_length_ms=cepstral_window_length_ms,
                    num_fft=n_fft,
                    num_fcc=n_fcc,
                )  # The last part of the file
            )  # NOTE: May overrepresent the last part
            responses.append(file_label)
            block_sample_ids.append(f"{file_.stem}.block{num_blocks}")

            blocks_total += num_blocks + 1  # +1 to get the last part
            all_cepstral_blocks.append(
                file_cepstral_blocks
            )  # gather all files in one list

            if ith_file_idx < 1 and verbose:
                print(
                    f"data_len: {data_len}, num blocks for file_idx#{ith_file_idx}: {len(file_cepstral_blocks)}, cepstral shape: {file_cepstral_blocks[0].shape}"
                )

    features = numpy.asarray(
        [numpy.asarray(block) for sample in all_cepstral_blocks for block in sample]
    )
    category = numpy.asarray(responses)

    # print(features.shape)
    assert (
        features.shape
        == numpy.empty((blocks_total, *numpy.shape(all_cepstral_blocks[0][0]))).shape
    )

    # from (block_sample_ith, mfcc_filt_ith, window_ith) to (block_sample_ith, window_ith, mfcc_filt_ith)
    features = features.transpose((0, 2, 1))  # transpose features
    # print(features.shape)

    assert len(block_sample_ids) == features.shape[0]

    if save_to_disk:
        out_file = ensure_existence(out_path) / f"{function.value}_{out_id}"
        print(f"Saving {out_file}")
        export_to_path(out_file, features, category, block_sample_ids)

    return features, category


def compute_dataset_features(
    root_path: Path,
    processed_path: Path,
    datasets=("adversarial_dataset-A", "adversarial_dataset-B"),
    out_part_id=("A", "B"),
    *,
    block_window_size_ms=512,  # 128
    block_window_step_size_ms=512,  # 128
    n_mfcc=20,  # 40
    n_fft=512,
    cepstral_window_length_ms=32,
    transformations: Sequence = CepstralSpaceEnum,
    block_wise: bool = True,
    skip_if_existing_dir: bool = False,
    skip_if_existing_file: bool = True,
) -> None:
    r"""

:param skip_if_existing_dir:
:param skip_if_existing_file:
:param processed_path:
:param root_path:
:type root_path:
:param block_window_size_ms:
:type block_window_size_ms:
:param block_window_step_size_ms:
:type block_window_step_size_ms:
:param n_mfcc:
:type n_mfcc:
:param n_fft:
:type n_fft:
:param cepstral_window_length_ms:
:type cepstral_window_length_ms:
:param transformations:
:type transformations:
:param datasets:
:type datasets:
:param out_part_id:
:type out_part_id:
:param block_wise:
:type block_wise:
:return:
:rtype:"""

    for data_s, part_id in progress_bar(zip(datasets, out_part_id)):
        data_path = root_path / data_s
        out_path = processed_path / part_id
        if skip_if_existing_dir and out_path.exists():
            continue
        ensure_existence(out_path)

        for fe in progress_bar(transformations):
            if (
                skip_if_existing_file and (out_path / f"{fe.value}_{part_id}").exists()
            ):  # LEAKY name but works for now
                continue
            if block_wise:
                block_wise_feature_extraction(
                    fe,
                    data_path,
                    save_to_disk=True,
                    out_path=out_path,
                    out_id=part_id,
                    block_window_size=block_window_size_ms,
                    block_window_step_size=block_window_step_size_ms,
                    n_fcc=n_mfcc,
                    n_fft=n_fft,
                    cepstral_window_length_ms=cepstral_window_length_ms,
                )
            else:
                file_wise_feature_extraction(
                    fe,
                    data_path,
                    save_to_disk=True,
                    out_path=out_path,
                    out_id=part_id,
                    n_fcc=n_mfcc,
                    n_fft=n_fft,
                    cepstral_window_length_ms=cepstral_window_length_ms,
                )


def compute_speech_silence_features(
    root_path: Path,
    processed_path: Path,
    *,
    block_window_size_ms=512,  # 128
    block_window_step_size_ms=512,  # 128
    n_mfcc=20,  # 40
    n_fft=512,
    cepstral_window_length_ms=32,
    transformations: Sequence = CepstralSpaceEnum,
    datasets=(
        "A",
        # 'B'
    ),
    partitions=("silence", "speech"),
    block_wise: bool = True,
    skip_if_existing_dir: bool = False,
    skip_if_existing_file: bool = True,
) -> None:
    r"""
Remains only block wise for now

:param skip_if_existing_dir:
:param skip_if_existing_file:
:param block_wise:
:param processed_path:
:param root_path:
:type root_path:
:param block_window_size_ms:
:type block_window_size_ms:
:param block_window_step_size_ms:
:type block_window_step_size_ms:
:param n_mfcc:
:type n_mfcc:
:param n_fft:
:type n_fft:
:param cepstral_window_length_ms:
:type cepstral_window_length_ms:
:param transformations:
:type transformations:
:param datasets:
:type datasets:
:param partitions:
:type partitions:
:return:
:rtype:"""

    for data_s in progress_bar(datasets):
        for partition in progress_bar(partitions):
            post = (data_s, partition)
            source_path = root_path / Path(*post)
            out_path = processed_path / Path(*post)
            if skip_if_existing_dir and out_path.exists():
                continue
            ensure_existence(out_path)
            for fe in progress_bar(transformations):
                out_id = partition
                if (
                    skip_if_existing_file
                    and (out_path / f"{fe.value}_{out_id}").exists()
                ):
                    continue
                if block_wise:
                    block_wise_feature_extraction(
                        fe,
                        source_path,
                        save_to_disk=True,
                        out_path=out_path,
                        out_id=out_id,
                        block_window_size=block_window_size_ms,
                        block_window_step_size=block_window_step_size_ms,
                        n_fcc=n_mfcc,
                        n_fft=n_fft,
                        cepstral_window_length_ms=cepstral_window_length_ms,
                    )
                else:
                    raise NotImplementedError


def compute_noised_dataset_features(
    root_path: Path,
    processed_path: Path,
    *,
    block_window_size_ms=512,  # 128
    block_window_step_size_ms=512,  # 128
    n_fcc=20,  # 40
    n_fft=512,
    cepstral_window_length_ms=32,
    transformations: Sequence = CepstralSpaceEnum,
    block_wise: bool = True,
    skip_if_existing_dir: bool = False,
    skip_if_existing_file: bool = True,
) -> None:
    r"""

:param skip_if_existing_dir:
:param skip_if_existing_file:
:param processed_path:
:param root_path:
:type root_path:
:param block_window_size_ms:
:type block_window_size_ms:
:param block_window_step_size_ms:
:type block_window_step_size_ms:
:param n_fcc:
:type n_fcc:
:param n_fft:
:type n_fft:
:param cepstral_window_length_ms:
:type cepstral_window_length_ms:
:param transformations:
:type transformations:
:param block_wise:
:type block_wise:
:return:
:rtype:"""
    for data_set in progress_bar(root_path.iterdir()):
        if data_set.is_dir():
            for data_split in progress_bar(data_set.iterdir()):
                if data_split.is_dir():
                    for snr in progress_bar(data_split.iterdir()):
                        """ # NARROW SELLECTION
            if True:
              if data_split.name != 'training' or snr.name != 'no_aug': #validation
                print('skip')
                continue
              else:
                print('csiuah')
                print(snr)
            """

                        if True:
                            if data_split.is_dir():
                                out_path = processed_path / Path(
                                    *(data_set.name, data_split.name, snr.name)
                                )
                                if skip_if_existing_dir and out_path.exists():
                                    continue
                                ensure_existence(out_path)
                                for fe in progress_bar(transformations):
                                    out_id = snr.name
                                    if (
                                        skip_if_existing_file
                                        and (out_path / f"{fe.value}_{out_id}").exists()
                                    ):
                                        continue
                                    if block_wise:
                                        block_wise_feature_extraction(
                                            fe,
                                            snr,
                                            save_to_disk=True,
                                            out_path=out_path,
                                            out_id=out_id,
                                            block_window_size=block_window_size_ms,
                                            block_window_step_size=block_window_step_size_ms,
                                            n_fcc=n_fcc,
                                            n_fft=n_fft,
                                            cepstral_window_length_ms=cepstral_window_length_ms,
                                        )
                                    else:
                                        file_wise_feature_extraction(
                                            fe,
                                            snr,
                                            save_to_disk=True,
                                            out_path=out_path,
                                            out_id=out_id,
                                            n_fcc=n_fcc,
                                            n_fft=n_fft,
                                            cepstral_window_length_ms=cepstral_window_length_ms,
                                        )


def compute_transformations(
    compu_reg=True,
    compu_ss=True,
    compu_noised=True,
    transformations: Sequence = (*CepstralSpaceEnum, OtherSpacesEnum.stft),
    # transformations=[FuncEnum.mel_fcc]
    block_window_size_ms=512,  # 512,  # 128
    block_window_step_size_ms=512,  # 512,  # 128
    n_fcc=20,  # 40, # 20
    n_fft=512,
    cepstral_window_length_ms=32,
    block_wise=True,
    skip_if_existing_dir: bool = False,
    skip_if_existing_file: bool = True,
):
    if compu_reg:
        print("computing regular example transformations")
        compute_dataset_features(
            DATA_ROOT_PATH,
            ensure_existence(DATA_REGULAR_PROCESSED_PATH),
            block_window_size_ms=block_window_size_ms,
            block_window_step_size_ms=block_window_step_size_ms,
            n_mfcc=n_fcc,
            n_fft=n_fft,
            cepstral_window_length_ms=cepstral_window_length_ms,
            transformations=transformations,
            block_wise=block_wise,
            skip_if_existing_dir=skip_if_existing_dir,
            skip_if_existing_file=skip_if_existing_file,
        )

    if compu_ss:
        print("computing ss_split example transformations")
        compute_speech_silence_features(
            DATA_ROOT_SS_SPLITS_UNPROCESSED_PATH,
            ensure_existence(DATA_SS_SPLITS_PROCESSED_PATH),
            block_window_size_ms=block_window_size_ms,
            block_window_step_size_ms=block_window_step_size_ms,
            n_mfcc=n_fcc,
            n_fft=n_fft,
            cepstral_window_length_ms=cepstral_window_length_ms,
            transformations=transformations,
            block_wise=block_wise,
            skip_if_existing_dir=skip_if_existing_dir,
            skip_if_existing_file=skip_if_existing_file,
        )

    if compu_noised:
        print("computing noised example transformations")
        compute_noised_dataset_features(
            DATA_ROOT_NOISED_UNPROCESSED_PATH,
            ensure_existence(DATA_NOISED_PROCESSED_PATH),
            block_window_size_ms=block_window_size_ms,
            block_window_step_size_ms=block_window_step_size_ms,
            n_fcc=n_fcc,
            n_fft=n_fft,
            cepstral_window_length_ms=cepstral_window_length_ms,
            transformations=transformations,
            block_wise=block_wise,
            skip_if_existing_dir=skip_if_existing_dir,
            skip_if_existing_file=skip_if_existing_file,
        )


if __name__ == "__main__":

    compute_transformations(
        compu_reg=False,
        compu_ss=False,
        compu_noised=True,
        skip_if_existing_file=False,
        transformations=[
            OtherSpacesEnum.stft,
            # CepstralSpaceEnum.linear_fcc
        ],
    )
