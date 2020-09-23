import time
from copy import deepcopy
from pathlib import Path

from apppath import AppPath
from warg import NOD, map_permutations, map_product

__all__ = ['PROJECT_APP_PATH', 'DATA_ROOT_PATH', 'CONFIG']
__author__ = 'Christian Heider Nielsen'

LOAD_TIME = str(int(time.time()))

PROJECT_APP_PATH = AppPath('adversarial_speech', __author__)
RESULTS_PATH = PROJECT_APP_PATH.user_data / 'results'

DATA_ROOT_PATH = Path.home() / 'Data' / 'Audio' / 'adversarial_speech'
DATA_ROOT_NOISED_PATH = PROJECT_APP_PATH.user_data / 'unprocessed' / 'noised'
DATA_ROOT_SPLITS_PATH = PROJECT_APP_PATH.user_data / 'unprocessed' / 'splits'

DATA_NOISED_PATH = PROJECT_APP_PATH.user_data / 'processed' / 'noised'
DATA_SPLITS_PATH = PROJECT_APP_PATH.user_data / 'processed' / 'splits'
DATA_REGULAR_PATH = PROJECT_APP_PATH.user_data / 'processed' / 'regular'



DEFAULT_DISTRIBUTION = NOD(train_percentage=0.7, test_percentage=0.2, validation_percentage=0.1)
FULL_TRAIN_DISTRIBUTION = NOD(train_percentage=1.0, test_percentage=0, validation_percentage=0)
VAL_TEST_DISTRIBUTION = NOD(train_percentage=0, test_percentage=0.2 / .3, validation_percentage=0.1 / .3)

A = NOD(path=DATA_REGULAR_PATH / 'A', name='A', **DEFAULT_DISTRIBUTION)
B = NOD(path=DATA_REGULAR_PATH / 'B', name='B', **DEFAULT_DISTRIBUTION)

A_speech = NOD(path=DATA_SPLITS_PATH / 'A' / 'speech',
               name='A_speech',
               **DEFAULT_DISTRIBUTION)
A_silence = NOD(path=DATA_SPLITS_PATH / 'A' / 'silence',
                name='A_silence',
                **DEFAULT_DISTRIBUTION)

DATA_A_NOISED_SPEECH_PATH = DATA_NOISED_PATH / 'A'
A_noised_airport_SNR_0dB = NOD(path=DATA_A_NOISED_SPEECH_PATH / 'airport_SNR_0dB',
                               name='airport_SNR_0dB',
                               **DEFAULT_DISTRIBUTION)
A_noised_airport_SNR_5dB = NOD(path=DATA_A_NOISED_SPEECH_PATH / 'airport_SNR_5dB',
                               name='airport_SNR_5dB',
                               **DEFAULT_DISTRIBUTION)
A_noised_airport_SNR_10dB = NOD(path=DATA_A_NOISED_SPEECH_PATH / 'airport_SNR_10dB',
                                name='airport_SNR_10dB',
                                **DEFAULT_DISTRIBUTION)
A_noised_airport_SNR_15dB = NOD(path=DATA_A_NOISED_SPEECH_PATH / 'airport_SNR_15dB',
                                name='airport_SNR_15dB',
                                **DEFAULT_DISTRIBUTION)
A_noised_airport_SNR_20dB = NOD(path=DATA_A_NOISED_SPEECH_PATH / 'airport_SNR_20dB',
                                name='airport_SNR_20dB',
                                **DEFAULT_DISTRIBUTION)

Tnoise = NOD(path=None, **DEFAULT_DISTRIBUTION)
S0 = NOD(path=None, **DEFAULT_DISTRIBUTION)
S5 = NOD(path=None, **DEFAULT_DISTRIBUTION)
S10 = NOD(path=None, **DEFAULT_DISTRIBUTION)
S15 = NOD(path=None, **DEFAULT_DISTRIBUTION)
S20 = NOD(path=None, **DEFAULT_DISTRIBUTION)
cafT0 = NOD(path=None, **DEFAULT_DISTRIBUTION)
cafT5 = NOD(path=None, **DEFAULT_DISTRIBUTION)
cafT10 = NOD(path=None, **DEFAULT_DISTRIBUTION)
cafT15 = NOD(path=None, **DEFAULT_DISTRIBUTION)
cafT20 = NOD(path=None, **DEFAULT_DISTRIBUTION)
B128 = NOD(path=None, **DEFAULT_DISTRIBUTION)
BS = NOD(path=None, **DEFAULT_DISTRIBUTION)
BNS = NOD(path=None, **DEFAULT_DISTRIBUTION)
_40A = NOD(path=None, train_set=0.75, test_set=0.24, validation_set=0.01),
_40B = NOD(path=None, train_set=0.75, test_set=0.24, validation_set=0.01),
norm = NOD(path=None, **DEFAULT_DISTRIBUTION)

CONFIG = NOD(
    Experiments=NOD(
        # **{f'EQUAL_MAPPING_{k1}to{k2}':NOD(Train_Sets={k1:v1},
        #                                    Validation_Sets={k2:v2},
        #                                    Test_Sets={k2:v2}
        #                                    )
        #    for (k1, k2), (v1, v2) in map_product({'A':deepcopy(A), 'B':deepcopy(B)})},
        #
        # **{f'EQUAL_MAPPING_{k1}to{k2}':NOD(Train_Sets={k1:v1},
        #                                    Validation_Sets={k2:v2},
        #                                    Test_Sets={k2:v2}
        #                                    )
        #    for (k1, k2), (v1, v2) in map_product(dict(A_speech=deepcopy(A_speech), A_silence=deepcopy(A_silence)))},
        #
        # **{f'UNEQUAL_MAPPING_{k1}to{k2}':NOD(Train_Sets={k1:deepcopy(v1).update(**FULL_TRAIN_DISTRIBUTION)},
        #                                      Validation_Sets={k2:deepcopy(v2).update(**VAL_TEST_DISTRIBUTION)},
        #                                      Test_Sets={k2:deepcopy(v2).update(**VAL_TEST_DISTRIBUTION)}
        #                                      )
        #    for (k1, k2), (v1, v2) in map_permutations({'A':deepcopy(A), 'B':deepcopy(B)})},
        #
        # **{f'UNEQUAL_MAPPING_{k1}to{k2}':NOD(Train_Sets={k1:deepcopy(v1).update(**FULL_TRAIN_DISTRIBUTION)},
        #                                      Validation_Sets={k2:deepcopy(v2).update(**VAL_TEST_DISTRIBUTION)},
        #                                      Test_Sets={k2:deepcopy(v2).update(**VAL_TEST_DISTRIBUTION)}
        #                                      )
        #    for (k1, k2), (v1, v2) in map_permutations(dict(A_speech=deepcopy(A_speech), A_silence=deepcopy(A_silence)))},
        #
        # MERGE_AB_TEST_AB=NOD(Train_Sets=NOD.nod_of(A, B),
        #                      Validation_Sets=NOD.nod_of(A, B),
        #                      Test_Sets=NOD.nod_of(A, B)),
        #
        # MERGE_AB_TEST_A=NOD(Train_Sets={'A':A, 'B':deepcopy(B).update(**FULL_TRAIN_DISTRIBUTION)},
        #                     Validation_Sets=NOD.nod_of(A),
        #                     Test_Sets=NOD.nod_of(A)),
        #
        # MERGE_AB_TEST_B=NOD(Train_Sets={'A':deepcopy(A).update(**FULL_TRAIN_DISTRIBUTION), 'B':B},
        #                     Validation_Sets=NOD.nod_of(B),
        #                     Test_Sets=NOD.nod_of(B)),

        # **{f'EQUAL_MAPPING_Ato{k2}':NOD(Train_Sets={'A':A},
        #                                   Validation_Sets={k2:v2},
        #                                   Test_Sets={k2:v2}
        #                                   )
        #    for (k2, v2) in
        #    NOD.nod_of(A_noised_airport_SNR_0dB,
        #               A_noised_airport_SNR_5dB,
        #               A_noised_airport_SNR_10dB,
        #               A_noised_airport_SNR_15dB,
        #               A_noised_airport_SNR_20dB)},
        #
        **{f'EQUAL_MAPPING_{k2}to{k2}':NOD(Train_Sets={k2:v2},
                                        Validation_Sets={k2:v2},
                                        Test_Sets={k2:v2}
                                        )
           for (k2, v2) in
           NOD.nod_of(A_noised_airport_SNR_0dB,
                      A_noised_airport_SNR_5dB,
                      A_noised_airport_SNR_10dB,
                      A_noised_airport_SNR_15dB,
                      A_noised_airport_SNR_20dB)},

        ),

    Training_Methodology=NOD(
        epochs=50,  # 100,
        batch_size=64,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.999),

    Validation=NOD(
        val_interval=1  # 5
        ),

    Num_Runs=5,  # 10,
    # Train_Sets=[
    # 'A',
    # 'B',
    # 'A_speech',
    # 'A_silence',
    # 'AB',
    # '40A',
    # '40B',
    # 'norm',
    # 'Snoise',
    # 'S0',
    # 'S5',
    # 'S10',
    # 'S15',
    # 'S20',
    # 'cafT0',
    # 'cafT5',
    # 'cafT10',
    # 'cafT15',
    # 'cafT20'
    # ],
    processed_file_ending='.npz'

    )

if __name__ == '__main__':
  print(CONFIG.Experiments.UNEQUAL_MAPPING_BtoA)
