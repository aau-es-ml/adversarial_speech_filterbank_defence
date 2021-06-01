import pathlib

import numpy
from scipy.io import wavfile

from apppath import ensure_existence
from configs.path_config import DATA_ROOT_PATH, DATA_ROOT_SS_SPLITS_UNPROCESSED_PATH
from draugr.tqdm_utilities import progress_bar, parallel_umap


__author__ = "Christian Heider Nielsen"
__doc__ = r"""Split the data into silence and speech parts"""

from data import AdversarialSpeechDataset
from external.rVADfast.rVADfast import speechproc
from external.rVADfast.rVADfast.rVAD_fast import get_rvad

n_too_short_speech = 0
n_too_short_silence = 0

__all__ = ["speech_silence_split"]

"""
def get_rms(data):
  '''
  root mean square
  :param data:
  :return:
  '''
  rms = numpy.sqrt(numpy.mean(data ** 2))
  return rms

def get_vad(file_name, block_size, step_size):
  sr, data = scipy.io.wavfile.read(file_name)  # load file
  data = data / numpy.max(data)
  data_len = len(data)

  speech = []
  silence = []
  vad = numpy.zeros((data_len,))
  blocks = (data_len - block_size) // step_size

  rms_full = get_rms(data)
  rms_plot = numpy.zeros((data_len,))

  for n in range(blocks):
    rms_block = get_rms(data[n * step_size:n * step_size + block_size])
    if rms_block > 0.2 * rms_full:
      vad[n * step_size:n * step_size + step_size] = 0.1

  for n in range(data_len):
    rms_plot[n] = rms_full
    if vad[n] == 0.1:
      speech.append(data[n])
    else:
      silence.append(data[n])
  #
  x = numpy.arange(0, data_len / sr, 1 / sr)
  return vad
"""


def speech_silence_split(file_name, speech_save_file_name, silence_save_file_name):
    sample_rate, data = wavfile.read(file_name)  # load file
    max_sample = numpy.max(data)
    data = data / max_sample
    # sample_rate, data=speechproc.speech_wave(file_name)
    speech = []
    silence = []
    voice_activity_mask = get_rvad(*speechproc.speech_wave(file_name))
    assert (
        len(voice_activity_mask) >= len(data) - 160
    ), f"{len(voice_activity_mask)}, {len(data)}"
    for v, d in zip(voice_activity_mask, data):
        if v == 1:
            speech.append(d)
        else:
            silence.append(d)

    skip_too_short = False
    skip_silence = False
    skip_speech = False
    if len(speech) < 8192:
        if skip_too_short:
            skip_speech = True

        if False:  # DISABLE FOR PARALLEL PROCESSING, COUNT DOES NO accumulate
            global n_too_short_speech
            n_too_short_speech += 1
            print(f"speech too short:{n_too_short_speech}")

    if len(silence) < 8192:
        if skip_too_short:
            skip_silence = True

        if False:  # DISABLE FOR PARALLEL PROCESSING, COUNT DOES NO accumulate
            global n_too_short_silence
            n_too_short_silence += 1
            print(f"silence too short:{n_too_short_silence}")

    if not skip_speech:
        ensure_existence(speech_save_file_name.parent)
        wavfile.write(
            speech_save_file_name,
            sample_rate,
            (numpy.asarray(speech) * max_sample).astype(numpy.int16),
        )

    if not skip_silence:
        ensure_existence(silence_save_file_name.parent)
        wavfile.write(
            silence_save_file_name,
            sample_rate,
            (numpy.asarray(silence) * max_sample).astype(numpy.int16),
        )


def compute_split(normal_f, packed):  # TODO: NAME proper and avoid using packed args
    data_s, adv_files, out_dir_speech, out_dir_silence = packed
    original = 1
    if data_s == "A":
        file_id = normal_f.name[-10:-4]  # for Dataset A
    else:
        file_id = normal_f.name[:8]  # for Dataset B

    for adv_f in adv_files:  # WHOA BAD COMPLEXITY! UGLY
        if file_id in adv_f.name:
            original = 0
            out_name_speech = out_dir_speech / "adv" / f"{file_id}_speech.wav"
            out_name_silence = out_dir_silence / "adv" / f"{file_id}_silence.wav"
            speech_silence_split(adv_f, out_name_speech, out_name_silence)

    if original == 1:
        out_name_speech = out_dir_speech / "normal" / f"{file_id}_speech.wav"
        out_name_silence = out_dir_silence / "normal" / f"{file_id}_silence.wav"
        speech_silence_split(normal_f, out_name_speech, out_name_silence)


def compute_speech_silence_splits(root: pathlib.Path = DATA_ROOT_PATH):
    """

    :param root:
    :return:
    """

    parallel = True  # SPEED UP!
    for data_s in [
        "A",
        # 'B' # ONLY A samples
    ]:

        normal_files, adv_files = AdversarialSpeechDataset(
            root / f"adversarial_dataset-{data_s}"
        ).get_all_samples_in_split()

        print(len(normal_files), len(adv_files))

        out_root = ensure_existence(DATA_ROOT_SS_SPLITS_UNPROCESSED_PATH / data_s)
        out_dir_speech = out_root / "speech"
        out_dir_silence = out_root / "silence"

        if parallel:
            parallel_umap(
                compute_split,
                normal_files,
                func_kws=dict(
                    data_s=data_s,
                    adv_files=adv_files,
                    out_dir_speech=out_dir_speech,
                    out_dir_silence=out_dir_silence,
                ),
            )
        else:
            for normal_f in progress_bar(normal_files):
                compute_split(
                    normal_f,
                    packed=(data_s, adv_files, out_dir_speech, out_dir_silence),
                )


if __name__ == "__main__":

    compute_speech_silence_splits()
