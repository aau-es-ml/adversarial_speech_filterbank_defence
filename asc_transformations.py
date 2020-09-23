from enum import Enum

__all__ = ['TransformationEnum']


class TransformationEnum(Enum):
  '''

  '''
  # short_time_frequency_transform = 'stft'
  mel_fcc = 'mfcc'
  inverse_mfcc = 'imfcc'
  gammatone_fcc = 'gfcc'
  inverse_gfcc = 'igfcc'
  linear_frequency_cepstral_coefficients = 'lfcc'
