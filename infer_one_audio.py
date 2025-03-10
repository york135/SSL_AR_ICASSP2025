import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys, os, json, pickle
import numpy as np
import time
import argparse
from tqdm import tqdm
from sklearn.utils import class_weight
import yaml
import random
from extract_aesrc import get_cont_feature, get_disc_feature
import s3prl.hub as hub
from speech2unit import *
import warnings
warnings.filterwarnings("ignore")

class AccentCLS(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, device='cpu'):
        super(AccentCLS, self).__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.bn1 = nn.BatchNorm1d(self.input_dim)

        self.template = nn.Parameter(torch.randn((1, self.hidden_dim, self.num_class),
                                                 dtype=torch.float32, requires_grad=True))

        self.accent = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
        )

    def forward(self, x, return_x=False):
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = self.accent(x)

        x_broadcast = torch.repeat_interleave(x.unsqueeze(2), repeats=self.num_class, dim=2)
        template_broadcast = torch.repeat_interleave(self.template, repeats=x_broadcast.shape[0], dim=0)

        cosine_sim = F.cosine_similarity(x_broadcast, template_broadcast, dim=1)
        if return_x:
            return cosine_sim, x
        else:
            return cosine_sim

def extract_wavlm_large_cont(audio_path, device):
    model = getattr(hub, 'wavlm_large')()
    model = model.to(device)
    audio_path_list = [audio_path,]

    wavlm_features = get_cont_feature(model, audio_path_list)[audio_path]
    return wavlm_features

def extract_wavlm_large_disc(audio_path, device):
    from speechbrain.lobes.models.huggingface_transformers.discrete_wavlm import DiscreteWavLM
    model_hub = "microsoft/wavlm-large"
    save_path = "savedir"
    kmeans_repo_id = "speechbrain/SSL_Quantization"
    kmeans_filename_list = ["LibriSpeech-100-360-500/wavlm/LibriSpeech_wavlm_k1000_L6.pt",
                            "LibriSpeech-100-360-500/wavlm/LibriSpeech_wavlm_k1000_L7.pt",
                            "LibriSpeech-100-360-500/wavlm/LibriSpeech_wavlm_k1000_L24.pt"]

    wavlm_disc_features = []
    for kmeans_filename in kmeans_filename_list:
        ssl_layer_num = int(kmeans_filename.split('_')[-1].split('.')[0][1:])
        kmeans_cache_dir = "savedir"

        model = DiscreteWavLM(model_hub, save_path, freeze=True,
            ssl_layer_num=ssl_layer_num,kmeans_repo_id=kmeans_repo_id, kmeans_filename=kmeans_filename
            , kmeans_cache_dir=kmeans_cache_dir).to(device)

        audio_path_list = [audio_path,]
        wavlm_features = get_disc_feature(model, audio_path_list)[audio_path]
        wavlm_disc_features.append(wavlm_features / float(torch.sum(wavlm_features)))
    return torch.cat(wavlm_disc_features, dim=1)

def extract_mhubert_cont(audio_path, device):
    model = getattr(hub, 'mhubert_base_vp_en_es_fr_it3')()
    model = model.to(device)

    audio_path_list = [audio_path,]
    mhubert_features = get_cont_feature(model, audio_path_list)[audio_path]
    return mhubert_features


def extract_mhubert_disc(audio_path, device):
    s2u = Speech2Unit(ckpt_dir='./')

    ssl_layer_num = 11
    audio_path_list = [audio_path,]
    mhubert_features = get_mhubert_discrete_feature(s2u, audio_path_list)[audio_path]
    return mhubert_features

def preprocess_audio(audio_path, device):
    wavlm_cont_features = extract_wavlm_large_cont(audio_path, device).reshape(1, -1)
    wavlm_disc_features = extract_wavlm_large_disc(audio_path, device)
    mhubert_cont_features = extract_mhubert_cont(audio_path, device).reshape(1, -1)
    mhubert_disc_features = extract_mhubert_disc(audio_path, device)

    # print (wavlm_cont_features.shape, wavlm_disc_features.shape, mhubert_cont_features.shape, mhubert_disc_features.shape)
    return torch.cat((wavlm_cont_features, mhubert_cont_features, wavlm_disc_features, mhubert_disc_features), dim=1)


def test_ensemble_one_audio(model, device, test_x):
    with torch.no_grad():
        model_input = test_x.to(device)
        out_global = 0
        for epoch in range(len(model)):
            out_global = out_global + model[epoch](model_input)

        test_prediction = torch.argmax(out_global, dim=1)
    return out_global, test_prediction

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train_dataset == 'AESRC':
        class_label = {'US': 0, 'UK': 1, 'CHN': 2, 'IND': 3, 'JPN': 4, "KR": 5, "PT": 6, "RU": 7}
    elif args.train_dataset == 'VCTK':
        class_label = {'American': 0, 'British': 1, 'Canadian': 2, 'Irish': 3, 'Scottish': 4}

    num_class = len(class_label)
    # print('Number of classes:', num_class)

    test_x = preprocess_audio(args.audio_path, device)
    input_dim = test_x[0].shape[0]
    # print ('Input dim:', input_dim)

    model = [AccentCLS(input_dim=input_dim, hidden_dim=args.hidden_dim, num_class=num_class
        , device=device).to(device) for i in range(args.epochs)]
    for i in range(args.epochs):
        model_path = os.path.join(args.model_dir, str(i+1) + '.pth.tar')
        model[i].load_state_dict(torch.load(model_path, map_location='cpu'))
        model[i].eval()

    out_global, test_prediction = test_ensemble_one_audio(model, device, test_x)
    print ('Predicted accent:', list(class_label.keys())[test_prediction])
    print ('All classes activations:', out_global)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accent recognition inference code")
    parser.add_argument('--audio_path', type=str, required=True, help='Input audio path for accent recognition')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of the pretrained model')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--train_dataset', type=str, required=True, choices=['AESRC', 'VCTK'], help='Training dataset \
        (AESRC/VCTK) (affect the classification setting)')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=16, help='number of hidden dim (latent space)')

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print('GPU ID:', args.gpu_id)

    main(args)