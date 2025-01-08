import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
import os
import json
import pickle
import numpy as np
import time, csv
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
    ACCENT_LIST = ["American", "British", "Canadian", "Irish", "Scottish"]
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

def load_vctk_data(args):
    if args.cont_files is not None:
        cont_pkl = []
        for cont_file in args.cont_files:
            with open(cont_file, 'rb') as f:
                cont_pkl.append(pickle.load(f))
            print ('Finish reading', cont_file)
    else:
        cont_pkl = None

    if args.disc_files is not None:
        disc_pkl = []
        for disc_file in args.disc_files:
            with open(disc_file, 'rb') as f:
                disc_pkl.append(pickle.load(f))
            print ('Finish reading', disc_file)
    else:
        disc_pkl = None

    all_label = {}
    with open(args.annotation_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0] == 'File':
                continue
            audio_path = os.path.join('VCTK-Corpus-0.92/wav48_silence_trimmed', row[0].split('.')[0] + '.flac')
            all_label[audio_path] = [row[1], row[2]]

    return cont_pkl, disc_pkl, all_label

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


def prepare_vctk_dataset(cont_pkl, disc_pkl, all_label, class_label):
    whole_dataset = []

    if cont_pkl is not None:
        key_list = cont_pkl[0].keys()
    else:
        key_list = disc_pkl[0].keys()
    key_list = sorted(key_list)

    for key in tqdm(key_list, desc="Preparing data"):
        cur_label = class_label[all_label[key][0]]
        if all_label[key][1] == 'Validation' or all_label[key][1] == 'PCA' or all_label[key][1] == 'Train':
            partition = 'Train'
        else:
            continue

        if cont_pkl is not None:
            all_features = [pkl[key].view(-1) for pkl in cont_pkl]
        else:
            all_features = []

        if disc_pkl is not None:
            discrete_feature = torch.cat([pkl[key] / float(torch.sum(pkl[key])) for pkl in disc_pkl], dim=0)
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

    class_label = {'American': 0, 'British': 1, 'Canadian': 2, 'Irish': 3, 'Scottish': 4}
    num_class = len(class_label)
    print('Number of classes:', num_class)

    cont_pkl, disc_pkl, all_label = load_vctk_data(args)
    whole_dataset = prepare_vctk_dataset(cont_pkl, disc_pkl, all_label, class_label)
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
    args.cont_files = config['cont_files']
    args.disc_files = config['disc_files']
    args.annotation_file = config['annotation_file']

    args.output_model_dir = os.path.join(args.output_model_dir, str(args.seed))

    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)

    print (args)
    main(args)