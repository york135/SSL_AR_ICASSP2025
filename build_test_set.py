import numpy as np
import librosa
import os, sys, pickle, csv
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    ACCENT_LIST = ["US", "UK", "CHN", "IND", "JPN", "KR", "PT", "RU"]
    results = {}
    with open('AESRC2020TestData.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # print (row[1], row[2])
            if row[1] == 'american':
                results['test/' + row[2].split('/')[-1]] = 'US'
            elif row[1] == 'british':
                results['test/' + row[2].split('/')[-1]] = 'UK'
            elif row[1] == 'chinese':
                results['test/' + row[2].split('/')[-1]] = 'CHN'
            elif row[1] == 'indian':
                results['test/' + row[2].split('/')[-1]] = 'IND'
            elif row[1] == 'japanese':
                results['test/' + row[2].split('/')[-1]] = 'JPN'
            elif row[1] == 'korean':
                results['test/' + row[2].split('/')[-1]] = 'KR'
            elif row[1] == 'portuguese':
                results['test/' + row[2].split('/')[-1]] = 'PT'
            elif row[1] == 'russian':
                results['test/' + row[2].split('/')[-1]] = 'RU'

    # print (results)

    annotation_pkl_path = 'test_annotation.pkl'

    with open(annotation_pkl_path, 'wb') as f:
        pickle.dump(results, f)