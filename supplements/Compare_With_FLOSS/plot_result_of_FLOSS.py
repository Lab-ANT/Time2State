'''
Created by Chengyu on 2023/4/20.
This script plots the results of FLOSS
'''
import os
import json
from miniutils import create_path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# For euclidean
dataset_list = ['UCR-SEG', 'ActRecTut', 'synthetic_data', 'MoCap', 'USC-HAD', 'PAMAP2']

# For dtw
# dataset_list = ['UCR-SEG', 'ActRecTut', 'synthetic_data', 'MoCap', 'USC-HAD']

metric='euclidean'

script_path = os.path.dirname(__file__)
crosscount_save_path = os.path.join(script_path, 'output_FLOSS-%s/crosscount/'%(metric))
clustering_result_save_path = os.path.join(script_path, 'output_FLOSS-%s/clustering'%(metric))

def plot_result_of(dataset):
    fig_save_path = os.path.join(script_path, 'figs_FLOSS-%s/%s/'%(metric,dataset))
    create_path(fig_save_path)
    cps_save_path = os.path.join(script_path, 'extracted_seg_pos/%s_result.json'%(dataset))
    with open(cps_save_path, 'r') as f:
        cps_json = json.load(f)

    name_list = list(cps_json)
    for file_name in name_list:
        print(file_name)
        data = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s'%(dataset, file_name)), header=None).to_numpy()
        true_crosscount = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s.cp'%(dataset, file_name[:-4])), header=None).to_numpy()
        predicted_crosscount = pd.read_csv(os.path.join(crosscount_save_path, '%s/%s'%(dataset, file_name)))
        groundtruth = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s.label'%(dataset, file_name[:-4])), header=None).to_numpy()
        prediction = np.load(os.path.join(clustering_result_save_path, '%s/%s.npy'%(dataset, file_name[:-4])))

        plt.style.use('classic')
        plt.figure(figsize=(12,6))
        grid = plt.GridSpec(6,1)
        ax1 = plt.subplot(grid[0:2])
        ax2 = plt.subplot(grid[2:4])
        ax3 = plt.subplot(grid[4:5])
        ax4 = plt.subplot(grid[5:6])
        
        ax1.plot(data, alpha=0.6)
        ax1.set_xlim([0, len(data)])
        ax1.set_title('Raw Data and Ground Truth Boundaries')
        ax2.plot(predicted_crosscount)
        ax2.set_title('CAC and Predicted Segment Boundaries')
        ax2.set_xlim([0, len(data)])

        ax2.set_yticks([0,0.5,1])
        ax3.set_yticks([])
        ax4.set_yticks([])

        max_v = np.max(data)
        min_v = np.min(data)
        for cp in true_crosscount:
            ax1.vlines(cp, min_v, max_v, color="black")

        # draw cut points.
        max_v = np.max(predicted_crosscount)
        min_v = np.min(predicted_crosscount)
        for cp in cps_json[file_name]:
            ax2.vlines(cp, min_v, max_v, color="black")
        ax3.set_title('Ground Truth State Sequence')
        ax3.imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
        ax4.set_title('Predicted State Sequence by FLOSS+TSKMeans')
        ax4.imshow(prediction.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_path, file_name+'.png'))
        plt.close()

for dataset in dataset_list:
    plot_result_of(dataset)
