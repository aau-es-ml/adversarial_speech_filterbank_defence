import os
import random
from pathlib import Path
from typing import Any

import numpy
import scipy
import scipy.signal as scipy_signal
import tqdm
from apppath import ensure_existence
from draugr.multiprocessing_utilities.pooled_tqdm import parallel_umap
from scipy.io import wavfile

from asc_utilities import get_normal_adv_wav_files
from experiment_config import DATA_ROOT_NOISED_A_PATH, DATA_ROOT_PATH
from external.my_rvadfast_py.rVAD_fast import get_rvad


def sample_noise(noise, noise_rate: int, signal_len: int, signal_rate: int, *, resample_noise: bool = True) -> Any:
  noise_len = len(noise)
  if resample_noise:
    noise = scipy_signal.resample(noise, round(noise_len * float(signal_rate) / noise_rate))
    # from sklearn.utils import resample
    # resample

  start_index = numpy.random.randint(0, noise_len)
  noise = numpy.tile(noise, ((signal_len // noise_len) + 2))  # atleast tile once (=2)
  return noise[start_index:signal_len + start_index]


def rms(data) -> numpy.ndarray:
  return numpy.sqrt(numpy.mean(data ** 2))


def get_speech(vad, data) -> numpy.ndarray:
  speech = []
  silence = []
  for n in range(min(len(data), len(vad))):
    if vad[n] == 1:
      speech.append(data[n])
    else:
      silence.append(data[n])

  return numpy.asarray(speech)  # ,numpy.asarray(silence)


def add_noise_5_types(rvad: numpy.ndarray,
                      file_name: Path,
                      noise=None):
  sr, data = scipy.io.wavfile.read(file_name)

  if not noise:
    noise_files = ("noise/bus/bus_train.wav",
                   "noise/str/str_train.wav",
                   "noise/ped/ped_train.wav",
                   "noise/caf/caf_train_cut.wav",
                   "noise/ssn/ssn_train.wav"
                   )
    noise_file = random.choice(noise_files)
    noise_type = noise_file[6:9]
    # print(noise_type)
    sr_noise, noise = scipy.io.wavfile.read(noise_file)
    noise = noise / numpy.max(noise)

  max_sample = numpy.max(data)
  data = data / max_sample
  data_speech = get_speech(rvad, data)
  start_index, index = sample_noise(len(noise), len(data))
  # data_rms = get_rms(data)
  speech_rms = rms(data_speech)
  noise_rms = rms(noise[index])

  ratio = speech_rms / noise_rms
  noise = noise * ratio
  SNR = numpy.random.randint(0, 6) * 5
  noise = noise / (10 ** (SNR / 20))
  if SNR == 25:
    noise = noise * 0
    SNR = 'clean'

  data_out = data + noise[index]
  data_out = data_out / numpy.max(data_out)
  data_out = data_out * max_sample

  file = remove_dir_in_name(file_name)
  out_name = f"{file[:-4]}_{noise_type}_{SNR}dB.wav"

  out_dir = "data/new_data/noise_combined_train/"
  scipy.io.wavfile.write(out_dir + out_name, sr, data_out.astype(numpy.int16))
  return numpy.asarray(data_out)


def add_noise_6_types(rvad,
                      file_name: Path,
                      noise_files=("noise/bus/bus_train.wav",
                                   "noise/str/str_train.wav",
                                   "noise/ped/ped_train.wav",
                                   "noise/ssn/ssn_train.wav",
                                   "noise/bbl/bbl_train.wav",
                                   "noise/caf/caf_train_cut.wav"
                                   )  # creates 5 dataset folders of .wav files that contains dataset A with 6 types of noise added; one for each SNR.The specified amounts of noise in the test sets are the SNRs: 0dB, 5dB,  10dB, 15dB and 20dB
                      ):
  sr, data = scipy.io.wavfile.read(file_name)
  max_sample = numpy.max(data)
  data = data / max_sample
  data_speech = get_speech(rvad, data)

  for snr in numpy.arange(0, 5) * 5:
    rand_noise = numpy.random.randint(len(noise_files))
    _, noise = scipy.io.wavfile.read(noise_files[rand_noise])
    noise = noise / numpy.max(noise)

    start_index, index = sample_noise(len(noise), len(data))
    speech_rms = rms(data_speech)
    noise_rms = rms(noise[index])
    ratio = speech_rms / noise_rms
    noise = noise * ratio

    noise = noise / (10 ** (snr / 20))
    data_out = data + noise[index]
    data_out = data_out / numpy.max(data_out)
    data_out = data_out * max_sample

    file = remove_dir_in_name(file_name)
    out_name = f"{file[:-4]}_{noise_files[rand_noise].split('/')[1]}_{snr}dB.wav"

    out_dir = f"data/noise_6/noise_W_Box_train{snr}/"
    os.makedirs(out_dir, exist_ok=True)
    scipy.io.wavfile.write(out_dir + out_name, sr, data_out.astype(numpy.int16))

  return numpy.asarray(data_out)


def compute_additive_noise_samples(rvad: numpy.ndarray,
                                   signal_file: Path,
                                   *,
                                   category,
                                   out_dir,
                                   noise_file) -> numpy.ndarray:
  sr_noise, noise = wavfile.read(str(noise_file))
  noise = noise / numpy.max(noise)

  sr_signal, signal = wavfile.read(str(signal_file))
  max_sample = numpy.max(signal)
  signal = signal / max_sample

  noise = sample_noise(noise, noise_rate=sr_noise, signal_len=len(signal), signal_rate=sr_signal)
  # assert len(rvad) >= len(signal), f'{len(rvad), len(signal)}'
  # speech_part = signal[rvad.astype(numpy.bool)[:min(len(signal),len(rvad))]]
  speech_part = get_speech(rvad, signal)
  noise = noise * (rms(speech_part) / rms(noise))  # scaled by ratio of speech to noise level

  for snr in (i * 5 for i in range(5)):
    noise = noise / (10 ** (snr / 20))
    noised = signal + noise
    noised = noised / numpy.max(noised)
    noised = noised * max_sample
    wavfile.write(str(ensure_existence(out_dir / f'{noise_file.with_suffix("").name}_SNR_{snr}dB' / category) / signal_file.name),
                  sr_signal,
                  noised.astype(numpy.int16))


def a(normal_example_file, packed):
  adv_files, noise_files, out_dir, parallel = packed

  clean_rvad_mask = get_rvad(*wavfile.read(normal_example_file))
  for noise_file in tqdm.tqdm(noise_files):
    compute_additive_noise_samples(clean_rvad_mask, normal_example_file, out_dir=out_dir, category='normal', noise_file=noise_file)

  if parallel:
    parallel_umap(b,
                  adv_files,
                  func_kws=dict(noise_files=noise_files,
                                normal_example_file=normal_example_file,
                                clean_rvad_mask=clean_rvad_mask))
  else:
    for adv_example_file in tqdm.tqdm(adv_files):  # Find adversarial examples matching original and use the clean rvad mask
      b(adv_example_file, packed=(noise_files, normal_example_file, clean_rvad_mask, out_dir))


def b(adv_example_file, packed):
  noise_files, normal_example_file, clean_rvad_mask, out_dir = packed
  if normal_example_file.name.split('-')[-1] in adv_example_file.name.split('-')[-1]:  # Same id in name
    for noise_file in tqdm.tqdm(noise_files):
      compute_additive_noise_samples(clean_rvad_mask, adv_example_file, out_dir=out_dir, category='adv', noise_file=noise_file)


if __name__ == '__main__':

  def main(parallel: bool = True):
    if True:
      for ss in (
          # 'AB',
          'A',
          #       'B'
          ):
        rp = DATA_ROOT_PATH / f'adversarial_dataset-{ss}'
        if ss == 'AB':
          rp = DATA_ROOT_PATH

        normal_files, adv_files = get_normal_adv_wav_files(rp)
        out_dir = DATA_ROOT_NOISED_A_PATH / ss

        noise_files = list((Path.home() / 'Data' / 'Audio' / 'noises').rglob('*.wav'))

        random.seed(0)
        numpy.random.seed(0)

        if parallel:
          parallel_umap(a,
                        normal_files,
                        func_kws=dict(adv_files=adv_files,
                                      noise_files=noise_files,
                                      out_dir=out_dir,
                                      parallel=False #deamonic process cannot have children

                                      ))
        else:
          for normal_example_file in tqdm.tqdm(normal_files):
            a(normal_example_file,
              packed=(adv_files,
                      noise_files,
                      out_dir,
                      parallel))


  main()
