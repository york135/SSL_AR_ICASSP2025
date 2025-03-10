# Similarity-based accent recognition

This is the official implementation of the following paper:

Jun-You Wang, Sheng Li, Li-An Lu, Sydney Chia-Chun Kao, Jyh-Shing Roger Jang, "Similarity-based accent recognition with continuous and discrete self-supervised speech representations," accepted at ICASSP 2025.

## Install dependencies

```
pip install -r requirements.txt
```

Feel free to let me know if I miss any requirement package.

## Dataset preparation

### Obtain datasets

We use the AESRC2020 dataset and the VCTK corpus in the experiments.

For **VCTK**, download the audios from [CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)](https://datashare.ed.ac.uk/handle/10283/3443), and the dataset partition & annotation file provided by Carofilis et al. from [VCTK_dataset_partitions_2023.csv](https://drive.google.com/file/d/1ECo8iK42HsQuyHlYemLSuPkPnAErxK6w/). 

For **AESRC2020**, according to the organizer of that dataset, it seems like one can obtain the dataset by following [https://github.com/R1ckShi/AESRC2020/issues/6](https://github.com/R1ckShi/AESRC2020/issues/6). However, I actually encountered "404 Page not found" when I tried to download it. 

As for the annotation of the test set, download it from [https://github.com/shangeth/AccentRecognition/blob/main/Dataset/AESRC2020TestData.csv](https://github.com/shangeth/AccentRecognition/blob/main/Dataset/AESRC2020TestData.csv).

In our case, Prof. Sheng Li (the second author) already had a copy of it, so we did not have to deal with this issue. I don't know if there is any way to get a copy of that dataset now. Maybe try to contact the organizer of the dataset?

### Arrange the dataset

Then, structure the datasets as follows. The two datasets are independent, so you only need to have at least one of them to run this repo.

```
# This is only the training set (along with the official 
# validation set) of AESRC2020. The test set is in another folder.
# We identify the ground-truth label by the folder name.
--AESRC2020
  |--chinese_english
    |-- G00021
      |--G00021S1053.wav
      |--G00021S1054.wav
    |-- G00183
  |--korean_english
  |--british_english
  |--japanese_english
  |--us_english
  |--russian_english
  |--indian_english
  |--portuguese_english

# This is AESRC2020's test set. It does not accompany with annotation.
--test
  |--AESRC2020-TESTSET-00001.wav
  |--AESRC2020-TESTSET-00002.wav

# This csv file serves as the annotation for AESRC2020's test set.
--AESRC2020TestData.csv

# Here is the VCTK corpus. Put the VCTK partition file inside it.
--VCTK-Corpus-0.92
  |--wav48_silence_trimmed
    |--p225
      |--p225_001_mic1.flac
      |--p225_001_mic2.flac
    |--p226
  |--VCTK_dataset_partitions_2023.csv
```

### Get AESRC2020's test set annotation

```
python build_test_set.py
```

This will generate ``test_annotation.pkl``, which serves as the ground-truth for AESRC2020's test set.

### Feature extraction

First, download the mHuBERT checkpoint from https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt , and mHuBERT L11 k-means checkpoint from https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin . These files should be put at `./`

Note that we do not have to download WavLM-large manually, as it will be done by [s3prl](https://github.com/s3prl/s3prl). We also do not have to download the WavLM-large k-means checkpoints manually, as they will be handled by [speechbrain](https://github.com/speechbrain/speechbrain).

Then, run:

```
python extract_aesrc.py
python extract_vctk.py
```

Note that these code only extract some of the layers' discrete features (L6, 7, 24 for WavLM-large, L11 for mHuBERT). You can modify them to extract other layers' features.

**CREDIT**

To obtain mHuBERT discrete features, we called ``speech2unit.py``, which was modified from [SpeechGPT](https://github.com/0nutation/SpeechGPT/blob/main/speechgpt/utils/speech2unit/speech2unit.py)'s speech2unit code.

## Training and inference

### Config file

We use a config yaml file to provide arguments for training. The config file formats are different for AESRC2020 and VCTK (mainly because the formats of the two datasets are different). For AESRC2020, the arguments include `train_cont_files`, `train_disc_files`, `test_cont_files`, and `test_disc_files`; for VCTK, the arguments only include `cont_files` and `disc_files`. They specify the feature files used as input for training/inference.

The arguments of our best setting can be found in:

For AESRC2020: `configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml`.

For VCTK: `configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml`.

### Training

For **AESRC2020**:

```
python accent_cls_aesrc.py --gpu_id [gpu id] 
  --config [config_file] \
  --seed [seed]
```

`gpu id`: specify which gpu device should be used for training. `-1` means no gpu.

`config_file`: specify the config file.

`seed`: specify the random seed.

For **VCTK**:

```
python accent_cls_vctk.py --gpu_id [gpu id] 
  --config [config_file] \
  --seed [seed]
```

An example can be found in `train_six_times.sh`, where the same training process will be repeated 6 times, with the random seed of 1, 2, 3, 4, 5, and 6. This is what we did in the experiments. 

### Inference / evaluation

For **AESRC2020**:

```
python test_ensemble_aesrc.py --gpu_id [gpu_id] \
 --config [config_file] \
 --seed [seeds] \
 --result_path [result_path]
```

`seeds`: ALL random seeds used for training that you want to evaluate.

`result_path`: This code will perform evaluation and compute accuracy and UAR. The evaluation results will be saved as a JSON file at `result_path`.

For example: `python test_ensemble_aesrc.py --gpu_id -1 --config configs/aesrc_config_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 1 2 3 4 5 6 --result_path aesrc_wavlm_cont_disc_mhubert_cont_disc.json`. 

For **VCTK**:

```
python test_ensemble_vctk.py --gpu_id [gpu_id] \
 --config [config_file] \
 --seed [seeds] \
 --result_path [result_path]
```

### Inference one audio clip (Issue #1)

Here is the code to inference one audio clip for the best proposed method.

```
python infer_one_audio.py --audio_path [audio_path] \
 --model_dir [model_dir] \
 --gpu_id [gpu_id] \
 --train_dataset [train_dataset]
```

 `audio_path`: Path to the input audio that we want to predict the accent.

`model_dir`: The directory to the pre-trained models. The code reads model_dir/{1:21}.pth.tar to perform inference.

`gpu_id`: specify which gpu device should be used for training. `-1` means no gpu.

`train_dataset`: specify the training dataset (either AESRC or VCTK). This affects the prediction settings (8-class or 5-class).



Here is a demo. The models can be downloaded [here](https://drive.google.com/drive/folders/1hZUzJlCMXDAEjAY-wFraA43T-GlEIiSp?usp=sharing).

```
python infer_one_audio.py --audio_path AESRC20/british_english/G00009/G00009S3401.wav \ 
 --model_dir aesrc_wavlm_cont_disc_mhubert_cont_disc/1/ \
 --train_dataset AESRC
```

The output will be:

```
Predicted accent: UK
All classes activations: tensor([[  8.0882,  14.6849,  -1.5777
,  -5.1962, -10.8787,   0.2356,  -4.1452, -1.5989]], device='cuda:0')
```

This is as expected, as this sample audio is in the AESRC2020 dataset's training set. It has to inference it correctly.



But, if we use the VCTK's checkpoint (i.e., cross-dataset evaluation):

```
python infer_one_audio.py --audio_path AESRC20/british_english/G00009/G00009S3401.wav \ 
 --model_dir vctk_wavlm_cont_disc_mhubert_cont/1/ \
 --train_dataset VCTK
```

The output will be:

```
Predicted accent: Irish
All classes activations: tensor([[-13.9888, -2.9950, 6.0587
    , 7.9860, 3.7857]]]], device='cuda:0')
```

Well, it says the speaker is Irish. Not that far away from the UK, but still an incorrect answer.

### Significance test

After obtaining the JSON files from `test_ensemble_aesrc.py` or `test_ensemble_vctk.py`, we can conduct significance test. Two scripts are provided: `ttest.py` and `sota_ttest.py`. `ttest.py` accepts two JSON files and test the statistical significance between them; `sota_ttest.py`  accepts one JSON file and test whether the results of this JSON file exceed the SOTA (0.8363 accuracy for AESRC2020; 0.356 UAR for VCTK).

```
python ttest.py [json file #1] [json file #2]
```

```
python sota_ttest.py [json file #1] [dataset name]
```

where `dataset name` should be either `AESRC` or `VCTK`. Based on the dataset name, `sota_ttest.py` chooses the proper SOTA and metric for comparison.

## **Misc**

### Regarding result reproduction

On Jan. 2025, when I organized the code after the acceptance of the paper, I re-ran the experiments to make sure everything works as expected, and found that our best setting achieved an accuracy of $$83.95\\% \pm 0.11\\%$$ on AESRC2020's test set, instead of $$84.01\\% \pm 0.14\\%$$ (as reported in the paper). That's probably due to some randomness  issue ($$p \approx 0.21$$ between the two attempts, i.e., no significant difference). Yet, this reproduction still outperforms previous SOTA ($$83.63\\%$$) significantly ($$p \approx 0.0004$$). This again confirms that the results and conclusions of our work can be reproduced and re-confirmed despite randomness.

The same observation can be found for VCTK (reported in the paper: $$50.03\\% \pm 0.40\\%$$; reproduced with current seeds: $$50.01\\% \pm 0.53\\%$$; $$p \approx 0.48$$ between the two attempts; $$p<10^{-8}$$ compared with the previous SOTA).

To access these results, see `jsons/`.

<<<<<<< HEAD
Anyway, I decided not to change the contents of the paper as the camera-ready version should not include too much changes from the original submission, but I report it here.
=======
Anyway, I decided not to change the contents of the paper (as the differences are rather minor), but report it here.
>>>>>>> 19d98b4b9c7cc0b99c19183023d7d15b6088b533

### Authors' contributions

**Jun-You Wang (Github @york135)** wrote the paper, conducted the experiments, and organized this repo.

**Sheng Li ([SHENG LI](https://halspeech.github.io/index.html))** suggested the idea of using discrete speech units and mHuBERT for accent recognition, preprocessed the AESRC2020 dataset, and provided valuable suggestions on the draft of the paper.

**Li-An Lu** and **Sydney Chia-Chun Kao** preprocessed the VCTK corpus, conducted preliminary experiments on the VCTK dataset using Wav2vec2 and HuBERT to prove the feasibility of accent recognition with SSL without accented ASR. Although these experiments are not mentioned in the paper due, I still recognize their contributions here.

**Jyh-Shing Roger Jang ([Roger Jang's Home Page](http://mirlab.org/jang/))** supervised this project.

### Some comments

- Although less emphasized in the paper, our "WavLM cont" setting already outperforms ([Li et al., 2023](https://www.isca-archive.org/interspeech_2023/li23aa_interspeech.html)), which also used WavLM-large for feature extraction (also extract continuous features), but with some data augmentation and accented ASR auxiliary task. In my opinion this is a very surprising finding.

- I somewhat acknowledge that, given the claims in our paper (that our method is particularly useful in low-resource scenarios), we should conduct experiments on smaller and more low-resource datasets, instead of AESRC2020 and VCTK, which are arguably not **very** small. But we have to do that as they were frequently used for experiments in previous works. I hope somebody will try our method on langauges/accents that are **really** low-resource. I am curious about the results!
