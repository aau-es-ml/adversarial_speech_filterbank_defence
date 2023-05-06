# Adversarial Speech Filterbank Defence

Can you distinguish these two samples?

https://user-images.githubusercontent.com/509350/236642144-126fbce0-53d0-474d-9745-523824938684.mov

https://user-images.githubusercontent.com/509350/236642150-5fb294d7-fb53-4463-851a-bba39345f573.mov

Sample 1 | Sample 2
:-: | :-:
[wav](sample/sample-000303.wav) | [wav](https://github.com/aau-es-ml/adversarial_speech_filterbank_defence/raw/master/sample/adv-short2short-000303.wav?raw=true)

And if we let you know, a common voice assistant would register one of these samples as "Open all doors"?

This work strives to detect these adversarial attacks by employing the underutilised power of cepstral coefficients 
computed using cleverly designed filter-banks and especially inverse filter-banks as shown in our article.

To get started trying to run our experiments yourself, just follow the steps below.    

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

Christian Heider Nielsen and Zheng-Hua Tan, "[Leveraging Domain Features for Detecting Adversarial Attacks Against Deep Speech Recognition in Noise](https://arxiv.org/pdf/2211.01621.pdf)," arXiv preprint arXiv:2211.01621 (2022).

Dataset is available [here](https://github.com/zhenghuatan/Audio-adversarial-examples).

