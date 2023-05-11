'''
Created by Chengyu on 2023/4/20.
This script plots the results of FLOSS
'''
import os
from miniutils import create_path, txt_filter
import pandas as pd
import numpy as np
from TSpy.eval import evaluate_clustering

dataset_list = ['PAMAP2', 'MoCap', 'USC-HAD', 'UCR-SEG', 'synthetic_data', 'ActRecTut']

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, 'data/Time2State_format')
state_seq_save_path = os.path.join(script_path, 'output_Time2State/')

def evaluate(dataset):
    fig_save_path = os.path.join(script_path, 'figs_Time2State/%s/'%(dataset))
    create_path(fig_save_path)
    file_list = txt_filter(os.listdir(os.path.join(data_path, dataset)))
    ari_list = []
    nmi_list = []
    for file_name in file_list:
        # print(file_name)
        # SimpleSynthetic_125_3000_5000.txt in the UCR-SEG dataset is a special case.
        # if file_name == 'SimpleSynthetic_125_3000_5000.txt':
        #     # print('simple synthetic')
        #     groundtruth = seg_to_label({3000:0, 5000:1, 8000:0})
        #     prediction = np.load(os.path.join(state_seq_save_path, '%s/state_seq/%s.npy'%(dataset, file_name)))
        #     ari, anmi, nmi  = evaluate_clustering(groundtruth.flatten(), prediction.flatten())
        #     print(ari,nmi)
        #     ari_list.append(ari)
        #     nmi_list.append(nmi)
        #     continue

        data = pd.read_csv(os.path.join(script_path, 'data/Time2State_format/%s/%s'%(dataset, file_name)), header=None).to_numpy()
        groundtruth = pd.read_csv(os.path.join(script_path, 'data/Time2State_format/%s/%s.label'%(dataset, file_name[:-4])), header=None).to_numpy()
        prediction = np.load(os.path.join(state_seq_save_path, '%s/state_seq/%s.npy'%(dataset, file_name)))
        ari, anmi, nmi  = evaluate_clustering(groundtruth.flatten(), prediction.flatten())
        ari_list.append(ari)
        nmi_list.append(nmi)
    print('%s: Average ARI: %f, NMI: %f'%(dataset, np.mean(ari_list), np.mean(nmi_list)))
    return np.mean(ari_list), np.mean(nmi_list)

for dataset in dataset_list:
    ari, nmi = evaluate(dataset)
