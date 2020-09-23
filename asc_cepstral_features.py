from pathlib import Path
from typing import Sequence, Tuple

import numpy
import tqdm
from apppath import ensure_existence
from draugr.matlab_utilities import matlab_to_ndarray, ndarray_to_matlab, start_engine
from scipy.io import wavfile
from scipy.signal import stft

from asc_utilities.zht_dataset_utilities import get_dataset_files_and_categories
from experiment_config import DATA_NOISED_PATH, DATA_REGULAR_PATH, DATA_ROOT_NOISED_PATH, DATA_ROOT_PATH, DATA_ROOT_SPLITS_PATH, DATA_SPLITS_PATH

__all__ = []

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
IMPORTANT! matlab scripts uses 'buffer' which requires Signal Processing Toolbox of Matlab!
and 'trimf' requires Fuzzy Logic Toolbox.


VERY SLOW! 

TODO: parallelise
'''

from MyGithub.adversarial_speech.asc_transformations import TransformationEnum


def processing_func(function: TransformationEnum,
                    data: numpy.ndarray,
                    *,
                    matlab_engine_,
                    sample_rate: int,
                    mfcc_window_length_ms: int,
                    n_fft: int,  # fft length in the matlab mfcc function
                    n_mfcc_filters: int,
                    mfcc_window_step_size_ms: int = 16,
                    article_check: bool = False) -> numpy.ndarray:  # chose what kind of processing to do
  if article_check:
    assert mfcc_window_length_ms == 32
    assert mfcc_window_step_size_ms == 16
    assert n_mfcc_filters == 20
    assert sample_rate == 16000
    assert n_fft == 512

  data_list_mat = ndarray_to_matlab(data)  # make data suited for matlab
  if function == TransformationEnum.mel_fcc:
    # future = matlab_engine_.extract_mfcc_edited(data_list_mat,background=True) #TODO: PARALLEL
    # ret = future.result()

    return matlab_to_ndarray(
        matlab_engine_.extract_mfcc_edited(data_list_mat,
                                           float(sample_rate),
                                           float(mfcc_window_length_ms),
                                           float(n_fft),
                                           float(n_mfcc_filters),
                                           nargout=3)[0]).T  # (speech,Fs,Window_Length,NFFT,No_Filter)


  elif function == TransformationEnum.inverse_mfcc:
    return numpy.asarray(
        matlab_engine_.extract_imfcc_edited(data_list_mat,
                                            float(sample_rate),
                                            float(mfcc_window_length_ms),
                                            float(n_fft),
                                            float(n_mfcc_filters),
                                            nargout=3)[0]).T  # (speech,Fs,Window_Length,NFFT,No_Filter)

  elif function == TransformationEnum.gammatone_fcc or function == TransformationEnum.inverse_gfcc:
    filterbank = numpy.matrix(matlab_engine_.fft2gammatonemx(float(n_fft),
                                                             float(sample_rate),
                                                             float(n_mfcc_filters),
                                                             float(1),
                                                             float(50),
                                                             float(8000),
                                                             float(n_fft / 2 + 1),
                                                             nargout=2)[0])  # (nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    filterbank = filterbank.getH()

    if function == TransformationEnum.inverse_gfcc:
      filterbank = filterbank[::-1]  # reverse the filterbank

    filterbank = ndarray_to_matlab(filterbank)

    return numpy.asarray(
        matlab_engine_.extract_fbcc_edited(data_list_mat,
                                           float(sample_rate),
                                           float(mfcc_window_length_ms),
                                           float(n_fft),
                                           float(n_mfcc_filters),
                                           filterbank, nargout=3)[0]).T  # (speech,Fs,Window_Length,NFFT,No_Filter,filterbank)
  elif function == TransformationEnum.linear_frequency_cepstral_coefficients:
    return numpy.asarray(matlab_engine_.extract_lfcc(data_list_mat,
                                                     float(sample_rate),
                                                     float(mfcc_window_length_ms),
                                                     float(n_fft),
                                                     float(n_mfcc_filters),
                                                     nargout=3)[0]).T  # (speech,Fs,Window_Length,NFFT,No_Filter)
  elif function == TransformationEnum.short_time_frequency_transform:  # UNUSED!
    return numpy.abs(
        stft(data,
             nfft=n_fft,
             nperseg=n_fft,
             noverlap=(mfcc_window_step_size_ms * sample_rate // 1000) // 2,
             # hop_length=stft_window_step_size_ms * sample_rate // 1000
             # Number of samples to jump for next fft frame
             ))

  raise Exception(f'{function} is not supported')


def file_wise_feature_extraction(function: TransformationEnum,
                                 path: Path,
                                 *,
                                 save_to_disk: bool,
                                 out_path: Path,
                                 out_id: str,
                                 n_mfcc_filters: int,
                                 cepstral_window_length_ms: int,  # window length in the matlab mfcc function
                                 n_fft_filters: int = 512,  # fft length in the matlab mfcc function
                                 max_files: int = 0,  # <0 = Inf samples
                                 min_trim: bool = False  # Else max pad
                                 ) -> Tuple:
  r'''
  UNUSED
  :param min_trim:
  :param max_files:
  :param function:
  :type function:
  :param path:
  :type path:
  :param save_to_disk:
  :type save_to_disk:
  :param out_path:
  :type out_path:
  :param out_id:
  :type out_id:
  :param n_mfcc_filters:
  :type n_mfcc_filters:
  :param cepstral_window_length_ms:
  :type cepstral_window_length_ms:
  :param n_fft_filters:
  :type n_fft_filters:
  :return:
  :rtype:
  '''

  if True: # For multi process setup, not used fully yet
    matlab_files_path = str(Path.cwd() / 'matlab_code')
    print(f'Cd.ing to: {matlab_files_path}')
    MATLAB_ENGINE.cd(matlab_files_path)

  file_paths, categories = get_dataset_files_and_categories(path)

  blocks_total = 0
  labels_split = []
  block_sample_ids = []
  all_cepstral_blocks = []

  for (ith_file_idx, (file_, file_label)) in zip(range(len(file_paths)),
                                                 tqdm.tqdm(zip(file_paths, categories),
                                                           total=len(file_paths),
                                                           desc=f'{function.value}')):
    if max_files and ith_file_idx >= max_files:
      break

    sampling_rate, wav_data = wavfile.read(file_)
    data_len = len(wav_data)

    if n_fft_filters >= data_len:
      if n_fft_filters >= data_len:
        print('to bad...')
    else:
      wav_data = (wav_data / 2.0 ** 15).astype(float)

      labels_split.append(file_label)
      block_sample_ids.append(f'{file_.stem}.block{0}')

      blocks_total += 1  # +1 to get the last part
      all_cepstral_blocks.append(processing_func(function,
                                                 wav_data,
                                                 matlab_engine_=MATLAB_ENGINE,
                                                 sample_rate=sampling_rate,
                                                 mfcc_window_length_ms=cepstral_window_length_ms,
                                                 n_fft=n_fft_filters,
                                                 n_mfcc_filters=n_mfcc_filters))  # gather all files in one list

      if ith_file_idx < 1:
        print(f'data_len: {data_len}, num blocks for file_idx#{ith_file_idx}: {1}, cepstral shape: {all_cepstral_blocks[0].shape}')

  sample_blocks = [numpy.asarray(sample) for sample in all_cepstral_blocks]

  if min_trim:
    trim = min([s.shape[-1] for s in sample_blocks])  # Trim to shortest sample, like in article
    features = numpy.array([s[..., :trim] for s in sample_blocks])
  else:  # Then max pad
    max_pad = max([s.shape[-1] for s in sample_blocks])
    features = numpy.array([numpy.pad(s, ((0, 0), (0, max_pad - s.shape[-1])), mode='constant', constant_values=(0, 0)) for s in sample_blocks])

  features = features.transpose((0, 2, 1))  # transpose features like in article
  print(features.shape)

  category = numpy.asarray(labels_split)

  assert len(block_sample_ids) == len(sample_blocks)

  if save_to_disk:
    ensure_existence(out_path)
    out_file = out_path / f'{function.value}_{out_id}'
    print(f'Saving {out_file}')
    numpy.savez(str(out_file), features=features, category=category, id=block_sample_ids)

  return features, category


def block_wise_feature_extraction(function: TransformationEnum,
                                  path: Path,
                                  *,
                                  save_to_disk: bool,
                                  out_path: Path,
                                  out_id: str,
                                  block_window_size: int,
                                  block_window_step_size: int,
                                  n_mfcc_filters: int,
                                  cepstral_window_length_ms: int,  # window length in the matlab mfcc function
                                  n_fft_filters: int = 512,  # fft length in the matlab mfcc function
                                  max_files: int = 0  # <0 = Inf samples
                                  ) -> Tuple:
  r'''

  :param function:
  :param path:
  :param save_to_disk:
  :param block_window_size:
  :param block_window_step_size:
  :param out_path:
  :param out_id:
  :param n_mfcc_filters:
  :param cepstral_window_length_ms:
  :param n_fft_filters:
  :param max_files:
  :return:
  '''
  if True:

    print(f'Cd.ing to: {matlab_files_path}')
    MATLAB_ENGINE.cd(matlab_files_path)

  file_paths, categories = get_dataset_files_and_categories(path)

  blocks_total = 0
  responses = []
  block_sample_ids = []
  all_cepstral_blocks = []

  for (ith_file_idx, (file_, file_label)) in zip(range(len(file_paths)),
                                                 tqdm.tqdm(zip(file_paths, categories),
                                                           total=len(file_paths),
                                                           desc=f'{function.value}')):
    if max_files and ith_file_idx >= max_files:
      break

    try:
      sampling_rate, wav_data = wavfile.read(file_)
    except Exception as e:
      print(file_)
      raise e
    data_len = len(wav_data)

    block_window_size_ms = (block_window_size * sampling_rate) // 1000  # window size in mS
    block_step_size_ms = (block_window_step_size * sampling_rate) // 1000

    if block_window_size_ms >= data_len or block_step_size_ms >= data_len or n_fft_filters >= data_len:
      if block_window_size_ms >= data_len:
        print("full size is reached for window")
      if block_step_size_ms >= data_len:
        print("full size is reached for step")
      if n_fft_filters >= data_len:
        print('to bad...')
    else:
      wav_data = (wav_data / 2.0 ** 15).astype(float)
      file_cepstral_blocks = []
      num_blocks: int = (data_len - block_window_size_ms) // block_step_size_ms

      for ith_block in tqdm.tqdm(range(num_blocks)):  # if negative
        file_cepstral_blocks.append(processing_func(function,
                                                    wav_data[ith_block * block_step_size_ms:ith_block * block_step_size_ms + block_window_size_ms],  # split data into blocks of window size
                                                    matlab_engine_=MATLAB_ENGINE,
                                                    sample_rate=sampling_rate,
                                                    mfcc_window_length_ms=cepstral_window_length_ms,
                                                    n_fft=n_fft_filters,
                                                    n_mfcc_filters=n_mfcc_filters))
        responses.append(file_label)
        block_sample_ids.append(f'{file_.stem}.block{ith_block}')

      file_cepstral_blocks.append(processing_func(function,
                                                  wav_data[data_len - block_window_size_ms:],
                                                  matlab_engine_=MATLAB_ENGINE,
                                                  sample_rate=sampling_rate,
                                                  mfcc_window_length_ms=cepstral_window_length_ms,
                                                  n_fft=n_fft_filters,
                                                  n_mfcc_filters=n_mfcc_filters)  # The last part of the file
                                  )  # NOTE: May overrepresent the last part
      responses.append(file_label)
      block_sample_ids.append(f'{file_.stem}.block{num_blocks}')

      blocks_total += num_blocks + 1  # +1 to get the last part
      all_cepstral_blocks.append(file_cepstral_blocks)  # gather all files in one list

      if ith_file_idx < 1:
        print(f'data_len: {data_len}, num blocks for file_idx#{ith_file_idx}: {len(file_cepstral_blocks)}, cepstral shape: {file_cepstral_blocks[0].shape}')

  features = numpy.asarray([numpy.asarray(block) for sample in all_cepstral_blocks for block in sample])
  category = numpy.asarray(responses)

  # print(features.shape)
  assert features.shape == numpy.empty((blocks_total, *numpy.shape(all_cepstral_blocks[0][0]))).shape

  # from (block_sample_ith, mfcc_filt_ith, window_ith) to (block_sample_ith, window_ith, mfcc_filt_ith)
  features = features.transpose((0, 2, 1))  # transpose features like in article
  # print(features.shape)

  assert len(block_sample_ids) == features.shape[0]

  if save_to_disk:
    out_file = ensure_existence(out_path) / f'{function.value}_{out_id}'
    print(f'Saving {out_file}')
    numpy.savez(str(out_file), features=features, category=category, id=block_sample_ids)

  return features, category


def compute_dataset_features(root_path: Path,
                             processed_path: Path,
                             datasets=('adversarial_dataset-A', 'adversarial_dataset-B'),
                             out_part_id=('A', 'B'),
                             *,
                             block_window_size_ms=512,  # 128
                             block_window_step_size_ms=512,  # 128
                             n_mfcc=20,  # 40
                             n_fft=512,
                             cepstral_window_length_ms=32,
                             transformations: Sequence = TransformationEnum,
                             block_wise: bool = True) -> None:
  r'''

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
  :rtype:
  '''

  for data_s, part_id in tqdm.tqdm(zip(datasets, out_part_id)):
    data_path = root_path / data_s
    out_path = ensure_existence(processed_path / part_id)
    for fe in tqdm.tqdm(transformations):
      if block_wise:
        block_wise_feature_extraction(fe,
                                      data_path,
                                      save_to_disk=True,
                                      out_path=out_path,
                                      out_id=part_id,
                                      block_window_size=block_window_size_ms,
                                      block_window_step_size=block_window_step_size_ms,
                                      n_mfcc_filters=n_mfcc,
                                      n_fft_filters=n_fft,
                                      cepstral_window_length_ms=cepstral_window_length_ms
                                      )
      else:
        file_wise_feature_extraction(fe,
                                     data_path,
                                     save_to_disk=True,
                                     out_path=out_path,
                                     out_id=part_id,
                                     n_mfcc_filters=n_mfcc,
                                     n_fft_filters=n_fft,
                                     cepstral_window_length_ms=cepstral_window_length_ms)


def compute_speech_silence_features(root_path: Path,
                                    processed_path: Path,
                                    *,
                                    block_window_size_ms=512,  # 128
                                    block_window_step_size_ms=512,  # 128
                                    n_mfcc=20,  # 40
                                    n_fft=512,
                                    cepstral_window_length_ms=32,
                                    transformations: Sequence = TransformationEnum,
                                    datasets=('A',
                                        # 'B'
                                              ),
                                    partitions=('silence', 'speech'),
                                    block_wise: bool = True) -> None:
  r'''
  Remains only block wise for now

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
  :rtype:
  '''

  for data_s in tqdm.tqdm(datasets):
    for partition in tqdm.tqdm(partitions):
      path = root_path / data_s / partition
      out_path = ensure_existence(processed_path / data_s / partition)
      for fe in tqdm.tqdm(transformations):
        if block_wise:
          block_wise_feature_extraction(fe,
                                        path,
                                        save_to_disk=True,
                                        out_path=out_path,
                                        out_id=f'{data_s}_{partition}',
                                        block_window_size=block_window_size_ms,
                                        block_window_step_size=block_window_step_size_ms,
                                        n_mfcc_filters=n_mfcc,
                                        n_fft_filters=n_fft,
                                        cepstral_window_length_ms=cepstral_window_length_ms
                                        )


def compute_noised_dataset_features(root_path: Path,
                                    processed_path: Path,
                                    *,
                                    block_window_size_ms=512,  # 128
                                    block_window_step_size_ms=512,  # 128
                                    n_mfcc=20,  # 40
                                    n_fft=512,
                                    cepstral_window_length_ms=32,
                                    transformations: Sequence = TransformationEnum,
                                    block_wise: bool = True,
                                    skip_if_existing:bool=True) -> None:
  r'''

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
  :rtype:
  '''
  for data_s in tqdm.tqdm(root_path.iterdir()):
    if data_s.is_dir():
      for snr in tqdm.tqdm(data_s.iterdir(), desc=f'{data_s}'):
        if data_s.is_dir():
          out_path = ensure_existence(processed_path / data_s.name / snr.name)
          for fe in tqdm.tqdm(transformations, desc=f'{snr}'):
            if skip_if_existing:
              if  (out_path / f'{fe.value}_A_noised_{snr.name}.npz').exists():
                print(f'WARNING SKIPPING {out_path,fe.value}')
                continue
            if block_wise:
              block_wise_feature_extraction(fe,
                                            snr,
                                            save_to_disk=True,
                                            out_path=out_path,
                                            out_id=f'A_noised_{snr.name}',
                                            block_window_size=block_window_size_ms,
                                            block_window_step_size=block_window_step_size_ms,
                                            n_mfcc_filters=n_mfcc,
                                            n_fft_filters=n_fft,
                                            cepstral_window_length_ms=cepstral_window_length_ms
                                            )
            else:
              file_wise_feature_extraction(fe,
                                           snr,
                                           save_to_disk=True,
                                           out_path=out_path,
                                           out_id=f'A_noised_{snr.name}',
                                           n_mfcc_filters=n_mfcc,
                                           n_fft_filters=n_fft,
                                           cepstral_window_length_ms=cepstral_window_length_ms)


if __name__ == "__main__":
  MATLAB_ENGINE = start_engine()
  matlab_files_path = str(Path.cwd() / 'asc_utilities' / 'feature_extraction' / 'matlab_implementation' / 'matlab_code')

  if False:
    print('computing regular example transformations')
    compute_dataset_features(
        DATA_ROOT_PATH,
        ensure_existence(DATA_REGULAR_PATH),
        block_window_size_ms=512,  # 512,  # 128
        block_window_step_size_ms=512,  # 512,  # 128
        n_mfcc=20,  # 40, # 20
        n_fft=512,
        cepstral_window_length_ms=32,
        # transformations=[FuncEnum.mel_fcc],
        block_wise=True)

  if False:
    print('computing split example transformations')
    compute_speech_silence_features(
        DATA_ROOT_SPLITS_PATH,
        ensure_existence(DATA_SPLITS_PATH),
        block_window_size_ms=512,  # 512,  # 128
        block_window_step_size_ms=512,  # 512,  # 128
        n_mfcc=20,  # 40, # 20
        n_fft=512,
        cepstral_window_length_ms=32,
        # transformations=[FuncEnum.mel_fcc],
        block_wise=True
        )

  if True:
    print('computing noised example transformations')
    compute_noised_dataset_features(
        DATA_ROOT_NOISED_PATH,
        ensure_existence(DATA_NOISED_PATH),
        block_window_size_ms=512,  # 512,  # 128
        block_window_step_size_ms=512,  # 512,  # 128
        n_mfcc=20,  # 40, # 20
        n_fft=512,
        cepstral_window_length_ms=32,
        # transformations=[FuncEnum.mel_fcc],
        block_wise=True
        )
