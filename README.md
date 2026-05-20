# ASD-SUPERB

* ASD-SUPERB includes a widely recognized Audio Spoof Detection downstream task that focuses on identifying whether an input audio signal is genuine (bona fide) or spoofed (fake). Spoofed audio refers to speech that has been artificially manipulated or generated using techniques such as text-to-speech (TTS) or voice conversion (VC).

* ASD-SUPERB serves as an extension of [S3PRL](https://github.com/s3prl/s3prl), a toolkit for SUPERB.

## Installation

Clone this repository to your workspace using the following command:
```ruby
git clone https://github.com/issflab/spoof_SUPERB.git
```

Create the conda environment from the provided YAML file:
```ruby
conda env create -f environment.yaml
```

Activate the environment:
```ruby
conda activate spoof_SUPERB
```

## Data preparation

1. Download the LA partition from [ASVSpoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) dataset, and unzip it.
```ruby
mkdir -p ASVSpoofData_2019

cd ASVSpoofData_2019

wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip

unzip LA.zip
```

2. Check the ASVSpoofData_2019 structure. You should have the following folder and files in LA directory at minimum. Read the Readme.txt to understand the folder and file structure.

```
ASVSpoofData_2019/
├── LA/
    ├── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019_LA_dev
    ├── ASVspoof2019_LA_eval
    ├── ASVspoof2019_LA_eval
    ├── README.LA.txt
```

## Training
To traia a model. First set up the config.py file with the required parameteres. By default, a linear head model will be trained on ASVSpoof 2019 data. The repository is configured to use only ASVSpoof 2019 LA data for training. Here is a sample command to run training. This command will train WavLM-LinearHead model with a batch size of 64 and number of epochs equal to 50. Use -h to look for more ssl_model options. 

```
python3 main.py --batch_size 64 --num_epochs 50 --ssl_model wavlm_large
```

## Evaluation

```
python3 main.py --eval --model_path "path_to_model" --ssl_feature "ssl_feature_name"
```





