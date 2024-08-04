# DDSP-QbE

This repository contains the source code of the paper *[Anonymising Elderly and Pathological Speech: Voice Conversion Using DDSP and Query-by-Example](https://www.researchgate.net/publication/381469769_Anonymising_Elderly_and_Pathological_Speech_Voice_Conversion_Using_DDSP_and_Query-by-Example), accepted in Interspeech 2024*.


![Concept of our method. For details we refer to our paper at .....](ddsp-qbe.png)

## Highlights:
- DDSP-QbE: an any-to-any voice conversion (VC) method designed to preserve prosody and domain characteristics in speech anonymization, even for unseen speakers from non-standard data, including pathological (stuttering) and elderly population.
- Builds on the concepts of query-by-example (QbE) and differentiable digital signal processing (DDSP).
- Utilizes a subtractive harmonic oscillator-based DDSP synthesizer, inspired by the human speech production model, for effective learning with limited data.
- Introduces an inductive bias for prosody preservation by:
  - Employing a novel loss function that uses emotional speech to separate prosodic and linguistic features.
  - Adding supplementary hand-crafted and deep learning-generated input features with prosodic knowledge from the source utterance.
- Experiments demonstrate its generalizability on the following benchmark datasets, across different genders, emotions, age, pathologies and cross-corpus conversions:
  - [Emotional Speech Dataset (ESD)](https://hltsingapore.github.io/ESD/)
  - [Sep28-K](https://machinelearning.apple.com/research/stuttering-event-detection)
  - [Dementia (ADReSS Challenge)](https://luzs.gitlab.io/adress/)

## Samples
Some of the samples can be found [here](https://github.com/suhitaghosh10/ddsp-qbe/tree/main/Samples). All samples could not be shared due to privacy issues.

## Demo
WIP

## Pre-requisites:
1. Python >= 3.11
2. Install the python dependencies mentioned in the requirements.txt

## Training:

### Before Training
1. Download Librispeech and ESD datasets and generate the WavLM 6th and 12th layers embeddings using the file preprocess.py.
2. All the training hyper parameter and paths are to be configured in resources/config.yaml file.
3. Mention the wavlm paths in config.yaml (train_path and val_path)
4. If you want to use emotion leakage specific loss, then please all the WavLM 6th layer embeddings for ESD files at location mentioned against the parameter 'emotion_files_wavlm6_path' in config.yaml. Further 'use_emo_loss' should be set as True in the config.yaml.

### Train
```bash
python main.py -g <gpu number> --config ./resources/config.yaml
```

### Model Weights
The model weights can be downloaded from [here](). 


## References and Acknowledgements

  
* In the original paper, the model was trained with ESD and data from Sep28-K and ADReSS Challenge. However due to privacy issues, we have provided the example training script using publicly available datasets. The pathological datasets can be obtained on request from their corresponding websites.
