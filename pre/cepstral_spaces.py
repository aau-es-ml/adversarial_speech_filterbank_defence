from enum import Enum

__all__ = ["CepstralSpaceEnum", "OtherSpacesEnum"]


class CepstralSpaceEnum(Enum):
    """
    Cepstral Spaces"""

    mel_fcc = "mfcc"
    inverse_mfcc = "imfcc"
    gammatone_fcc = "gfcc"
    inverse_gfcc = "igfcc"
    linear_fcc = "lfcc"  # linear_frequency_cepstral_coefficients


class OtherSpacesEnum(Enum):
    """
    Only used for plotting spectrograms, not for modelling"""

    short_term_ft = "stft"
    power_spec = "pspec"
