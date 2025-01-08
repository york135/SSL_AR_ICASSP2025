import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
import os
import json
import pickle
import numpy as np
import time
import argparse
from tqdm import tqdm
from sklearn.utils import class_weight
import yaml
import random

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



def divide_data(dataset, num_class):
    train_x = []
    train_y = []

    train_class_dict = [0 for _ in range(num_class)]

    for i in range(len(dataset)):
        train_class_dict[dataset[i]['label']] = train_class_dict[dataset[i]['label']] + 1

    print("Train class distribution:", train_class_dict)

    for i in range(len(dataset)):
        train_x.append(dataset[i]['features'])
        train_y.append(dataset[i]['label'])

    return train_x, train_y

def parse_result(confusion_matrix):
    ACCENT_LIST = ["US", "UK", "CHN", "IND", "JPN", "KR", "PT", "RU"]
    ACCENT_NUM = len(ACCENT_LIST)
    utt_nums = [0] * ACCENT_NUM
    correct_nums = [0] * ACCENT_NUM
    
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            utt_nums[i] = utt_nums[i] + confusion_matrix[i][j]
            if i == j:
                correct_nums[i] = correct_nums[i] + confusion_matrix[i][j]

    acc_per_accent = [100.0 * correct_nums[i] / utt_nums[i] for i in range(ACCENT_NUM)]
    # for i in range(ACCENT_NUM):
    #     print('{} Accent Accuracy: {:.1f}'.format(ACCENT_LIST[i], acc_per_accent[i]))
    print('Average ACC: {} / {} = {:.1f}'.format(sum(correct_nums), sum(utt_nums), 100.0 * sum(correct_nums) / sum(utt_nums)))

    return 100.0 * sum(correct_nums) / sum(utt_nums)

def UAR(cm):
    c = len(cm)
    uar = sum((cm[i][i] / sum(cm[i][j] for j in range(c)) if sum(cm[i][j] for j in range(c)) != 0 else 0) for i in range(c)) / c
    return uar

def load_data(args):
    if args.train_cont_files is not None:
        train_cont_pkl = []
        for train_cont_file in args.train_cont_files:
            with open(train_cont_file, 'rb') as f:
                train_cont_pkl.append(pickle.load(f))
            print ('Finish reading', train_cont_file)
    else:
        train_cont_pkl = None

    if args.train_disc_files is not None:
        train_disc_pkl = []
        for train_disc_file in args.train_disc_files:
            with open(train_disc_file, 'rb') as f:
                train_disc_pkl.append(pickle.load(f))
            print ('Finish reading', train_disc_file)
    else:
        train_disc_pkl = None

    return train_cont_pkl, train_disc_pkl

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    train_confusion = [[0 for _ in range(model.num_class)] for _ in range(model.num_class)]
    correct_train_sample = 0.0
    total_train_sample = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batchs in pbar:
        model_input, gt = batchs[0].to(device), batchs[1].to(device)
        optimizer.zero_grad()
        out_global, _ = model(model_input, return_x=True)
        loss = criterion(out_global, gt)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_prediction = torch.argmax(out_global, dim=1)
        
        for j in range(len(train_prediction)):
            train_confusion[gt[j]][train_prediction[j]] += 1
            if train_prediction[j] == gt[j]:
                correct_train_sample += 1
            total_train_sample += 1

        pbar.set_postfix({"loss": loss.item()})

    train_loss /= len(train_loader)
    train_acc = correct_train_sample / total_train_sample
    train_uar = UAR(train_confusion)
    train_avg_acc = parse_result(train_confusion)

    print(f"Training loss: {train_loss:.4f}; Train UAR: {train_uar:.4f}; Train avg acc: {train_avg_acc:.4f}")
    return train_loss, train_uar, train_avg_acc


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_confusion = [[0 for _ in range(model.num_class)] for _ in range(model.num_class)]

    with torch.no_grad():
        # for batchs in tqdm(test_loader, desc="Testing"):
        for batchs in test_loader:
            model_input, gt = batchs[0].to(device), batchs[1].to(device)
            out_global = model(model_input)
            loss = criterion(out_global, gt)
            test_loss += loss.item()
            test_prediction = torch.argmax(out_global, dim=1)
            
            for j in range(len(test_prediction)):
                test_confusion[gt[j]][test_prediction[j]] += 1

    test_loss /= len(test_loader)
    test_uar = UAR(test_confusion)
    test_avg_acc = parse_result(test_confusion)

    print(f"Test loss: {test_loss:.4f}; UAR: {test_uar:.4f}; Test avg acc: {test_avg_acc:.4f}")
    return test_loss, test_uar, test_avg_acc

def prepare_dataset(train_cont_pkl, train_discrete_pkl, folder_label):
    whole_dataset = []
    speakers_list = {folder: [] for folder in folder_label.keys()}

    if train_cont_pkl is not None:
        train_key_list = train_cont_pkl[0].keys()
    else:
        train_key_list = train_discrete_pkl[0].keys()
    train_key_list = sorted(train_key_list)

    for key in tqdm(train_key_list):
        split_audio_path = key.split('/')
        speaker_name = split_audio_path[2]
        folder = split_audio_path[1].split('_')[0]
        if speaker_name not in speakers_list[folder]:
            speakers_list[folder].append(speaker_name)

    val_speaker_num = 0

    for key in tqdm(train_key_list, desc="Preparing training data"):
        folder = key.split('/')[1].split('_')[0]
        cur_label = folder_label[folder]
        partition = 'Train'

        if train_cont_pkl is not None:
            all_features = [pkl[key].view(-1) for pkl in train_cont_pkl]
        else:
            all_features = []

        if train_discrete_pkl is not None:
            discrete_feature = torch.cat([pkl[key] / float(torch.sum(pkl[key])) for pkl in train_discrete_pkl], dim=0)
            all_features.append(discrete_feature.view(-1))

        features = torch.cat(all_features, dim=0)
        whole_dataset.append({'partition': partition, 'features': features, 'label': cur_label})

    return whole_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(0)
    return g

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g = set_seed(seed=args.seed)

    folder_label = {'us': 0, 'british': 1, 'chinese': 2, 'indian': 3, 'japanese': 4, "korean": 5, "portuguese": 6, "russian": 7}
    num_class = len(folder_label)
    print('Number of classes:', num_class)

    train_cont_pkl, train_discrete_pkl = load_data(args)
    whole_dataset = prepare_dataset(train_cont_pkl, train_discrete_pkl, folder_label)
    train_x, train_y = divide_data(whole_dataset, num_class)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)

    train_x, train_y = torch.stack(train_x), torch.tensor(train_y)

    train_dataset = TensorDataset(train_x, train_y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    input_dim = train_x[0].shape[0]
    print ('Input dim:', input_dim)
    model = AccentCLS(input_dim=input_dim, hidden_dim=args.hidden_dim, num_class=num_class, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device).float())
    
    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_uar, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        if args.save_model:
            target_model_path = os.path.join(args.output_model_dir, f'{epoch + 1}.pth.tar')
            torch.save(model.state_dict(), target_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accent Classification")
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print('GPU ID:', args.gpu_id)

    args.batch_size = int(config['batch_size'])
    args.epochs = int(config['epochs'])
    args.hidden_dim = int(config['hidden_dim'])
    args.learning_rate = float(config['learning_rate'])
    args.num_workers = int(config['num_workers'])
    args.output_model_dir = config['output_model_dir']
    args.save_model = config['save_model']
    args.weight_decay = float(config['weight_decay'])
    args.train_cont_files = config['train_cont_files']
    args.train_disc_files = config['train_disc_files']
    # args.test_cont_files = config['test_cont_files']
    # args.test_disc_files = config['test_disc_files']
    # args.test_annotation_file = config['test_annotation_file']

    args.output_model_dir = os.path.join(args.output_model_dir, str(args.seed))

    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)

    print (args)
    main(args)