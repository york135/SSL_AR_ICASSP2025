import s3prl.hub as hub
import torch
import numpy as np
import librosa
import os, sys, pickle, csv
from tqdm import tqdm

from glob import glob
from speech2unit import *

device = 'cuda'  # or 'cpu'

def get_cont_feature(model, audio_path_list):
    all_features = {}
    with torch.no_grad():
        for audio_path in tqdm(audio_path_list):
            cur_features = []
            y, _ = librosa.load(audio_path, sr=16000)
            waveform = [torch.tensor(y).to(device)]
            reps = model(waveform)["hidden_states"]
            
            for i in range(1, len(reps)):
                features = torch.mean(reps[i], dim=1).cpu()
                cur_features.append(features)

            cur_features = torch.cat(cur_features, dim=0)
            all_features[audio_path] = cur_features

    return all_features

def get_disc_feature(model, audio_path_list):
    all_features = {}
    with torch.no_grad():
        for audio_path in tqdm(audio_path_list):
            cur_features = torch.zeros((1, 1000))
            y, _ = librosa.load(audio_path, sr=16000)
            waveform = torch.tensor(y).to(device).unsqueeze(0)

            embs, tokens = model(waveform)
            for i in range(tokens.shape[1]):
                cur_features[0][tokens[0][i]] += 1

            all_features[audio_path] = cur_features
    return all_features

def extract_wavlm_large_cont():
    model = getattr(hub, 'wavlm_large')()
    model = model.to(device)

    print ('Start extracting test set')
    # Test set
    audio_path_list = []
    with open('AESRC2020TestData.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[2][-3:] == 'wav':
                audio_path_list.append(os.path.join('test', row[2].split('/')[-1]))
                
    print (len(audio_path_list), audio_path_list[0])

    wavlm_features = get_cont_feature(model, audio_path_list)
    wavlm_pkl_path = 'aesrc_wavlm_large_cont_test.pkl'
    with open(wavlm_pkl_path, 'wb') as f:
        pickle.dump(wavlm_features, f)

    # Training set
    print ('Start extracting training set')
    audio_path_list = [y for x in os.walk('AESRC20') for y in glob(os.path.join(x[0], '*.wav'))]
    print (len(audio_path_list), audio_path_list[0])

    wavlm_features = get_cont_feature(model, audio_path_list)
    wavlm_pkl_path = 'aesrc_wavlm_large_cont_train.pkl'
    with open(wavlm_pkl_path, 'wb') as f:
        pickle.dump(wavlm_features, f)

def extract_wavlm_large_disc():
    from speechbrain.lobes.models.huggingface_transformers.discrete_wavlm import DiscreteWavLM
    model_hub = "microsoft/wavlm-large"
    save_path = "savedir"
    kmeans_repo_id = "speechbrain/SSL_Quantization"
    kmeans_filename_list = ["LibriSpeech-100-360-500/wavlm/LibriSpeech_wavlm_k1000_L6.pt",
                            "LibriSpeech-100-360-500/wavlm/LibriSpeech_wavlm_k1000_L7.pt",
                            "LibriSpeech-100-360-500/wavlm/LibriSpeech_wavlm_k1000_L24.pt"]

    for kmeans_filename in kmeans_filename_list:
        ssl_layer_num = int(kmeans_filename.split('_')[-1].split('.')[0][1:])
        print ('ssl_layer_num:', ssl_layer_num)
        kmeans_cache_dir = "savedir"

        model = DiscreteWavLM(model_hub, save_path, freeze=True,
            ssl_layer_num=ssl_layer_num,kmeans_repo_id=kmeans_repo_id, kmeans_filename=kmeans_filename
            , kmeans_cache_dir=kmeans_cache_dir).to(device)

        print ('Start extracting test set')
        # Test set
        audio_path_list = []
        with open('AESRC2020TestData.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if row[2][-3:] == 'wav':
                    audio_path_list.append(os.path.join('test', row[2].split('/')[-1]))
                    
        wavlm_features = get_disc_feature(model, audio_path_list)
        wavlm_pkl_path = 'aesrc_wavlm_large_disc_L' + str(ssl_layer_num) + '_test.pkl'
        with open(wavlm_pkl_path, 'wb') as f:
            pickle.dump(wavlm_features, f)

        # Training set
        print ('Start extracting training set')
        audio_path_list = [y for x in os.walk('AESRC20') for y in glob(os.path.join(x[0], '*.wav'))]
        # print (len(audio_path_list), audio_path_list[0])

        wavlm_features = get_disc_feature(model, audio_path_list)
        wavlm_pkl_path = 'aesrc_wavlm_large_disc_L' + str(ssl_layer_num) + '_train.pkl'
        with open(wavlm_pkl_path, 'wb') as f:
            pickle.dump(wavlm_features, f)

def extract_mhubert_cont():
    model = getattr(hub, 'mhubert_base_vp_en_es_fr_it3')()
    model = model.to(device)

    print ('Start extracting test set')
    # Test set
    audio_path_list = []
    with open('AESRC2020TestData.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[2][-3:] == 'wav':
                audio_path_list.append(os.path.join('test', row[2].split('/')[-1]))
                
    print (len(audio_path_list), audio_path_list[0])

    wavlm_features = get_cont_feature(model, audio_path_list)
    wavlm_pkl_path = 'aesrc_mhubert_cont_test.pkl'
    with open(wavlm_pkl_path, 'wb') as f:
        pickle.dump(wavlm_features, f)

    # Training set
    print ('Start extracting training set')
    audio_path_list = [y for x in os.walk('AESRC20') for y in glob(os.path.join(x[0], '*.wav'))]
    print (len(audio_path_list), audio_path_list[0])

    wavlm_features = get_cont_feature(model, audio_path_list)
    wavlm_pkl_path = 'aesrc_mhubert_cont_train.pkl'
    with open(wavlm_pkl_path, 'wb') as f:
        pickle.dump(wavlm_features, f)

def extract_mhubert_disc():
    s2u = Speech2Unit(ckpt_dir='./')

    ssl_layer_num = 11
    print ('ssl_layer_num:', ssl_layer_num)

    print ('Start extracting test set')
    # Test set
    audio_path_list = []
    with open('AESRC2020TestData.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[2][-3:] == 'wav':
                audio_path_list.append(os.path.join('test', row[2].split('/')[-1]))
                
    # print (len(audio_path_list), audio_path_list[0])

    wavlm_features = get_mhubert_discrete_feature(s2u, audio_path_list)
    wavlm_pkl_path = 'aesrc_mhubert_disc_L' + str(ssl_layer_num) + '_test.pkl'
    with open(wavlm_pkl_path, 'wb') as f:
        pickle.dump(wavlm_features, f)

    # Training set
    print ('Start extracting training set')
    audio_path_list = [y for x in os.walk('AESRC20') for y in glob(os.path.join(x[0], '*.wav'))]
    # print (len(audio_path_list), audio_path_list[0])

    wavlm_features = get_mhubert_discrete_feature(s2u, audio_path_list)
    wavlm_pkl_path = 'aesrc_mhubert_disc_L' + str(ssl_layer_num) + '_train.pkl'
    with open(wavlm_pkl_path, 'wb') as f:
        pickle.dump(wavlm_features, f)

if __name__ == "__main__":
    extract_mhubert_cont()
    extract_mhubert_disc()
    extract_wavlm_large_cont()
    extract_wavlm_large_disc()
