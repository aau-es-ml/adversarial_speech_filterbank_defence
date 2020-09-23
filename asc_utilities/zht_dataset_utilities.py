import os
import pathlib
from typing import List, Sequence, Tuple

__all__ = ['get_dataset_files_and_categories', 'get_normal_adv_wav_files']

from experiment_config import DATA_ROOT_PATH


def get_dataset_files_and_categories(root_dir: pathlib.Path) -> Tuple[Sequence, Sequence]:
  '''

  wrapper that return a flattened file name list and a binary response vector [ 0 for non-adv, 1 for adv]

  :param root_dir:
  :return:
  '''
  normal_files, adv_files = get_normal_adv_wav_files(root_dir)
  file_names = normal_files + adv_files
  response = [0] * len(normal_files) + [1] * len(adv_files)

  return file_names, response


def get_normal_adv_wav_files(path: pathlib.Path, verbose: bool = False) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
  """

  Specific to Zheng Hua Tan dataset

  :param verbose:
  :param path:
  :return:
  """
  normal_files = []
  adv_files = []
  for wav_p in path.rglob("*.wav"):
    if ('Original-examples' in str(wav_p.parent) or
        'Original-Examples' in str(wav_p.parent) or
        'Normal-Examples' in str(wav_p.parent) or
        'normal' in str(wav_p.parent)
    ):
      normal_files.append(wav_p)
    else:
      if ("adv" in wav_p.name or
          "adv" == wav_p.parent.name or
          os.path.join('Adversarial-Examples', 'Adversarial-Examples') in str(wav_p)
      ):
        adv_files.append(wav_p)
      else:
        print(f'UNEXPECTED! {wav_p}, excluding')

  assert len(normal_files) > 0, f'no normal examples found'
  assert len(adv_files) > 0, f'no adversarial examples found'
  if verbose:
    print(path)
    print(f'num normal samples: {len(normal_files)}')
    print(f'num adversarial samples: {len(adv_files)}')
  return normal_files, adv_files


if __name__ == '__main__':
  _ = get_normal_adv_wav_files(DATA_ROOT_PATH / 'adversarial_dataset-A', verbose=True)
  _ = get_normal_adv_wav_files(DATA_ROOT_PATH / 'adversarial_dataset-B', verbose=True)
