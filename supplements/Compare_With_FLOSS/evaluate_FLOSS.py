'''
Created by Chengyu on 2023/4/20.
This script plots the results of FLOSS
'''
import os
import json
import pandas as pd
import numpy as np
from TSpy.eval import evaluate_clustering

# this list if for euclidean
dataset_list = ['UCR-SEG', 'ActRecTut', 'synthetic_data', 'MoCap', 'USC-HAD', 'PAMAP2']

# this list is for dtw
# dataset_list = ['ActRecTut', 'synthetic_data', 'MoCap', 'USC-HAD', 'UCR-SEG']

# Modify here to change distance metric
# remember to switch the dataset list above.
# euclidean or dtw
metric='euclidean'

script_path = os.path.dirname(__file__)
# raw_data_path = os.path.join(script_path)
crosscount_save_path = os.path.join(script_path, 'output_FLOSS-%s/crosscount/'%(metric))
clustering_result_save_path = os.path.join(script_path, 'output_FLOSS-%s/clustering'%(metric))

def evaluate(dataset):
    cps_save_path = os.path.join(script_path, 'extracted_seg_pos/%s_result.json'%(dataset))

    if not os.path.exists(cps_save_path):
        return

    with open(cps_save_path, 'r') as f:
        cps_json = json.load(f)

    name_list = list(cps_json)
    ari_list = []
    nmi_list = []

    for file_name in name_list:
        groundtruth = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s.label'%(dataset, file_name[:-4])), header=None).to_numpy()
        prediction = np.load(os.path.join(clustering_result_save_path, '%s/%s.npy'%(dataset, file_name[:-4])))
        ari, anmi, nmi = evaluate_clustering(groundtruth.flatten(), prediction.flatten())        
        ari_list.append(ari)
        nmi_list.append(nmi)
    print('%s: Average ARI is %f, Average NMI is %f'%(dataset, np.mean(ari_list), np.mean(nmi_list)))

for dataset in dataset_list:
    evaluate(dataset)
