#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01-02-2021
           """

import numpy
from draugr.matlab_utilities import matlab_to_ndarray, ndarray_to_matlab

from pre.cepstral_spaces import CepstralSpaceEnum, OtherSpacesEnum

__all__ = ["cepstral_extractor"]


def cepstral_extractor(
    function: CepstralSpaceEnum,
    data: numpy.ndarray,
    *,
    matlab_engine_: object,
    sample_rate: int,
    cepstral_window_length_ms: int,
    num_fft: int,  # fft length in the matlab fcc function
    num_fcc: int,  # number of fcc filters in matlab
    concat_delta: bool = False,
    concat_delta_delta: bool = False,
) -> numpy.ndarray:  # chose what kind of processing to do
    """

      Matlab implementation, it is slow!

    :param concat_delta_delta:
    :param concat_delta:
    :param function:
    :param data:
    :param matlab_engine_:
    :param sample_rate:
    :param cepstral_window_length_ms:
    :param num_fft:
    :param num_fcc:
    :return:"""

    data_list_mat = ndarray_to_matlab(data)  # make data suited for matlab
    if function == CepstralSpaceEnum.mel_fcc:
        # future = matlab_engine_.extract_mfcc_edited(data_list_mat,background=True) #TODO: PARALLEL
        # ret = future.result()

        res = matlab_engine_.extract_mfcc_edited(
            data_list_mat,
            float(sample_rate),
            float(cepstral_window_length_ms),
            float(num_fft),
            float(num_fcc),
            nargout=3,
        )  # (speech,Fs,Window_Length,NFFT,No_Filter)
        mfcc = matlab_to_ndarray(res[0])
        if concat_delta:
            mfcc = numpy.stack([mfcc, matlab_to_ndarray(res[1])])
        if concat_delta_delta:
            mfcc = numpy.stack([mfcc, matlab_to_ndarray(res[2])])
        return mfcc

    elif function == CepstralSpaceEnum.inverse_mfcc:

        res = matlab_engine_.extract_imfcc_edited(
            data_list_mat,
            float(sample_rate),
            float(cepstral_window_length_ms),
            float(num_fft),
            float(num_fcc),
            nargout=3,
        )  # (speech,Fs,Window_Length,NFFT,No_Filter)
        imfcc = matlab_to_ndarray(res[0])
        if concat_delta:
            imfcc = numpy.stack([imfcc, matlab_to_ndarray(res[1])])
        if concat_delta_delta:
            imfcc = numpy.stack([imfcc, matlab_to_ndarray(res[2])])
        return imfcc
        """
elif function == CepstralSpaceEnum.fourier_bessel_cc:
res = matlab_engine_.extract_fbcc_edited(
    data_list_mat,
    float(sample_rate),
    float(cepstral_window_length_ms),
    float(num_fft),
    float(num_fcc),
    nargout=3,
)  # (speech,Fs,Window_Length,NFFT,No_Filter)
fbcc = matlab_to_ndarray(res[0])
if concat_delta:
    fbcc = numpy.stack([fbcc, matlab_to_ndarray(res[1])])
if concat_delta_delta:
    fbcc = numpy.stack([fbcc, matlab_to_ndarray(res[2])])
return fbcc
"""
    elif (
        function == CepstralSpaceEnum.gammatone_fcc
        or function == CepstralSpaceEnum.inverse_gfcc
    ):
        filterbank = matlab_to_ndarray(  # numpy MATRIX
            matlab_engine_.fft2gammatonemx(
                float(num_fft),
                float(sample_rate),
                float(num_fcc),
                float(1),
                float(50),
                float(8000),
                float(num_fft / 2 + 1),
                nargout=2,
            )[
                0
            ]  # (nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
        )
        # filterbank = filterbank.getH() # Hermitian conjugate
        filterbank = filterbank.conj().T  # Hermitian conjugate

        if function == CepstralSpaceEnum.inverse_gfcc:
            filterbank = filterbank[::-1]  # reverse the filterbank

        filterbank = ndarray_to_matlab(filterbank)

        res = matlab_engine_.extract_fbcc_edited(
            data_list_mat,
            float(sample_rate),
            float(cepstral_window_length_ms),
            float(num_fft),
            float(num_fcc),
            filterbank,
            nargout=3,
        )  # (speech,Fs,Window_Length,NFFT,No_Filter,filterbank)
        gfcc = matlab_to_ndarray(res[0])
        if concat_delta:
            gfcc = numpy.stack([gfcc, matlab_to_ndarray(res[1])])
        if concat_delta_delta:
            gfcc = numpy.stack([gfcc, matlab_to_ndarray(res[2])])
        return gfcc
    elif function == CepstralSpaceEnum.linear_fcc:
        res = matlab_engine_.extract_lfcc_edited(
            data_list_mat,
            float(sample_rate),
            float(cepstral_window_length_ms),
            float(num_fft),
            float(num_fcc),
            nargout=4,
        )  # (speech,Fs,Window_Length,NFFT,No_Filter)
        lfcc = matlab_to_ndarray(res[0])
        if concat_delta:
            lfcc = numpy.stack([lfcc, matlab_to_ndarray(res[1])])
        if concat_delta_delta:
            lfcc = numpy.stack([lfcc, matlab_to_ndarray(res[2])])
        return lfcc
    elif function == OtherSpacesEnum.short_term_ft:

        return matlab_to_ndarray(
            matlab_engine_.extract_lfcc_edited(
                data_list_mat,
                float(sample_rate),
                float(cepstral_window_length_ms),
                float(num_fft),
                float(num_fcc),
                nargout=5,
            )[
                -2
            ]  # (speech,Fs,Window_Length,NFFT,No_Filter)
        )

    elif function == OtherSpacesEnum.power_spec:
        a = matlab_to_ndarray(
            matlab_engine_.extract_lfcc_edited(
                data_list_mat,
                float(sample_rate),
                float(cepstral_window_length_ms),
                float(num_fft),
                float(num_fcc),
                nargout=5,
            )[
                -1
            ]  # (speech,Fs,Window_Length,NFFT,No_Filter)
        )
        return a

    raise Exception(f"{function} is not supported")
