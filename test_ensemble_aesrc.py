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
    test_x = []
    test_y = []
    val_x = []
    val_y = []

    train_class_dict = [0 for _ in range(num_class)]
    valid_class_dict = [0 for _ in range(num_class)]
    test_class_dict = [0 for _ in range(num_class)]

    for i in range(len(dataset)):
        if dataset[i]['partition'] == 'Train' or dataset[i]['partition'] == 'PCA':
            train_class_dict[dataset[i]['label']] = train_class_dict[dataset[i]['label']] + 1
        elif dataset[i]['partition'] == 'Test':
            test_class_dict[dataset[i]['label']] = test_class_dict[dataset[i]['label']] + 1
        else:
            valid_class_dict[dataset[i]['label']] = valid_class_dict[dataset[i]['label']] + 1

    # print("Test class distribution:", test_class_dict)
    for i in range(len(dataset)):
        if dataset[i]['partition'] == 'Test':
            test_x.append(dataset[i]['features'])
            test_y.append(dataset[i]['label'])

    return train_x, train_y, val_x, val_y, test_x, test_y

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
    if args.test_cont_files is not None:     
        test_cont_pkl = []
        for test_cont_file in args.test_cont_files:
            with open(test_cont_file, 'rb') as f:
                test_cont_pkl.append(pickle.load(f))
            # print ('Finish reading', test_cont_file)
    else:
        test_cont_pkl = None

    if args.test_disc_files is not None:
        test_disc_pkl = []
        for test_disc_file in args.test_disc_files:
            with open(test_disc_file, 'rb') as f:
                test_disc_pkl.append(pickle.load(f))
            # print ('Finish reading', test_disc_file)
    else:
        test_disc_pkl = None

    with open(args.test_annotation_file, 'rb') as f:
        test_label = pickle.load(f)

    return test_cont_pkl, test_disc_pkl, test_label

def test_ensemble(model, device, test_loader):
    test_confusion = [[0 for _ in range(model[0].num_class)] for _ in range(model[0].num_class)]

    with torch.no_grad():
        for batchs in test_loader:
            model_input, gt = batchs[0].to(device), batchs[1].to(device)
            out_global = 0
            for epoch in range(len(model)):
                out_global = out_global + model[epoch](model_input)

            test_prediction = torch.argmax(out_global, dim=1)
            
            for j in range(len(test_prediction)):
                test_confusion[gt[j]][test_prediction[j]] += 1

    test_uar = UAR(test_confusion)
    test_avg_acc = parse_result(test_confusion)

    # print(f"UAR: {test_uar:.4f}; Test avg acc: {test_avg_acc:.4f}")
    return test_uar, test_avg_acc

def prepare_dataset(test_cont_pkl, test_discrete_pkl, test_label, class_label, folder_label):
    whole_dataset = []
    if test_cont_pkl is not None:
        test_key_list = test_cont_pkl[0].keys()
    else:
        test_key_list = test_discrete_pkl[0].keys()
    test_key_list = sorted(test_key_list)

    for key in test_key_list:
        try:
            cur_label = class_label[test_label[key]]

            if test_cont_pkl is not None:
                all_features = [pkl[key].view(-1) for pkl in test_cont_pkl]
            else:
                all_features = []

            if test_discrete_pkl is not None:
                discrete_feature = torch.cat([pkl[key] / float(torch.sum(pkl[key])) for pkl in test_discrete_pkl], dim=0)
                all_features.append(discrete_feature.view(-1))

            features = torch.cat(all_features, dim=0)
            whole_dataset.append({'partition': 'Test', 'features': features, 'label': cur_label})
        except KeyError:
            pass

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

    class_label = {'US': 0, 'UK': 1, 'CHN': 2, 'IND': 3, 'JPN': 4, "KR": 5, "PT": 6, "RU": 7}
    folder_label = {'us': 0, 'british': 1, 'chinese': 2, 'indian': 3, 'japanese': 4, "korean": 5, "portuguese": 6, "russian": 7}
    num_class = len(class_label)
    # print('Number of classes:', num_class)

    test_cont_pkl, test_discrete_pkl, test_label = load_data(args)
    whole_dataset = prepare_dataset(test_cont_pkl, test_discrete_pkl, test_label, class_label, folder_label)
    train_x, train_y, val_x, val_y, test_x, test_y = divide_data(whole_dataset, num_class)

    test_x = torch.stack(test_x)
    test_y = torch.tensor(test_y)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers, drop_last=False)

    input_dim = test_x[0].shape[0]
    # print ('Input dim:', input_dim)

    model = [AccentCLS(input_dim=input_dim, hidden_dim=args.hidden_dim, num_class=num_class
        , device=device).to(device) for i in range(args.epochs)]
    for i in range(args.epochs):
        model_path = os.path.join(args.output_model_dir, str(i+1) + '.pth.tar')
        model[i].load_state_dict(torch.load(model_path, map_location='cpu'))
        model[i].eval()

    print (time.time())
    test_uar, test_acc = test_ensemble(model, device, test_loader)
    print (time.time())
    return test_uar, test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accent Classification")
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--seed', type=int, nargs='+', required=True, help='Random seed')
    parser.add_argument('--result_path', type=str, required=True, help='Result path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print('GPU ID:', args.gpu_id)

    args.batch_size = int(config['batch_size'])
    args.epochs = int(config['epochs'])
    args.hidden_dim = int(config['hidden_dim'])
    args.num_workers = int(config['num_workers'])
    args.output_model_dir = config['output_model_dir']
    args.weight_decay = float(config['weight_decay'])
    args.test_cont_files = config['test_cont_files']
    args.test_disc_files = config['test_disc_files']
    args.test_annotation_file = config['test_annotation_file']

    seed_list = list(args.seed)
    root_output_dir = args.output_model_dir
    print (args)
    print (seed_list)

    cur_result_list = []
    for cur_seed in seed_list:
        args.output_model_dir = os.path.join(root_output_dir, str(cur_seed))
        args.seed = cur_seed
        test_uar, test_acc = main(args)
        cur_result_list.append({'UAR': test_uar, 'Accuracy': test_acc})
    print (cur_result_list)

    with open(args.result_path, 'w') as f:
        json.dump(cur_result_list, f, indent=2, ensure_ascii=False)