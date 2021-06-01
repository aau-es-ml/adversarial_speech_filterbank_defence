# Adversarial Speech

# Usage

install requirements from ``pip install -r requirements.txt``

## One go

run ``python run_all.py``

## Manually

    Firstly
    
    - Download the required dataset, no official hosting is available.

    - Setup dataset paths in path_config.py, under the ./config path of this project 
    
    - use Robust Voice Activity Detection (Speech/Silence) on the raw dataset to split dataset into speech and silence parts; asc_split_speech.py
    
    - asc_noise_augmentation.py
    
    Then 
    1. asc_cepstral_features.py
    2. asc_train.py
    3. asc_test.py
    4. asc_extract_csv.py
    5. asc_plot_agg.py

# Citations

<!-- - [1] a --> 