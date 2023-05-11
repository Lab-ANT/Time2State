'''
Created by Chengyu on 2023/4/20.
This script plots the results of FLOSS
'''
import os
from miniutils import create_path, txt_filter, find_cp_from_state_seq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_list = ['PAMAP2', 'MoCap', 'USC-HAD', 'UCR-SEG', 'synthetic_data', 'ActRecTut']

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, 'data/Time2State_format')
state_seq_save_path = os.path.join(script_path, 'output_Time2State/')

def plot_result_of(dataset):
    fig_save_path = os.path.join(script_path, 'figs_Time2State/%s/'%(dataset))
    create_path(fig_save_path)
    file_list = txt_filter(os.listdir(os.path.join(data_path, dataset)))
    for file_name in file_list:
        print(file_name)
        data = pd.read_csv(os.path.join(script_path, 'data/Time2State_format/%s/%s'%(dataset, file_name)), header=None).to_numpy()
        groundtruth = pd.read_csv(os.path.join(script_path, 'data/Time2State_format/%s/%s.label'%(dataset, file_name[:-4])), header=None).to_numpy()
        prediction = np.load(os.path.join(state_seq_save_path, '%s/state_seq/%s.npy'%(dataset, file_name)))
        true_cps = pd.read_csv(os.path.join(script_path, 'data/Time2State_format/%s/%s.cp'%(dataset, file_name[:-4])), header=None).to_numpy()
        print(data.shape, prediction)
        plt.style.use('classic')
        plt.figure(figsize=(8,4))
        grid = plt.GridSpec(4,1)
        ax1 = plt.subplot(grid[0:2])
        ax2 = plt.subplot(grid[2:3])
        ax3 = plt.subplot(grid[3:4])
        ax1.plot(data, alpha=0.6)
        ax1.set_xlim([0, len(data)])
        ax1.set_title('Raw Data and Ground Truth Boundaries')
        max_v = np.max(data)
        min_v = np.min(data)
        for cp in true_cps:
            ax1.vlines(cp, min_v, max_v, color="black")

        found_cps = find_cp_from_state_seq(prediction)
        for cp in found_cps:
            ax3.vlines(cp, -0.5, 0.5, color="black")

        ax2.set_title('Ground Truth State Sequence')
        ax2.set_yticks([])
        ax2.imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
        ax3.set_yticks([])
        ax3.set_title('Predicted State Sequence by Time2State')
        ax3.imshow(prediction.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_path, file_name+'.png'))
        plt.close()

for dataset in dataset_list:
    plot_result_of(dataset)

# script_path = os.path.dirname(__file__)
# data_path = os.path.join(script_path, 'data/FLOSS_format')
# state_seq_save_path = os.path.join(script_path, 'output_Time2State/')
# clustering_result_save_path = os.path.join(script_path, 'output_FLOSS/state_seq')

# def plot_result_of(dataset):
#     fig_save_path = os.path.join(script_path, 'figs_Time2State/%s/'%(dataset))
#     create_path(fig_save_path)
#     file_list = txt_filter(os.listdir(os.path.join(data_path, dataset)))
#     for file_name in file_list:
#         print(file_name)
#         data = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s'%(dataset, file_name)), header=None).to_numpy()
#         groundtruth = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s.label'%(dataset, file_name[:-4])), header=None).to_numpy()
#         prediction = np.load(os.path.join(state_seq_save_path, '%s/state_seq/%s.npy'%(dataset, file_name)))
#         true_cps = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s.cp'%(dataset, file_name[:-4])), header=None).to_numpy()
#         print(data.shape, prediction)
#         plt.style.use('classic')
#         plt.figure(figsize=(8,4))
#         grid = plt.GridSpec(4,1)
#         ax1 = plt.subplot(grid[0:2])
#         ax2 = plt.subplot(grid[2:3])
#         ax3 = plt.subplot(grid[3:4])
#         ax1.plot(data, alpha=0.6)
#         ax1.set_xlim([0, len(data)])
#         ax1.set_title('Raw Data and Ground Truth Boundaries')
#         max_v = np.max(data)
#         min_v = np.min(data)
#         for cp in true_cps:
#             ax1.vlines(cp, min_v, max_v, color="black")

#         found_cps = find_cp_from_state_seq(prediction)
#         for cp in found_cps:
#             ax3.vlines(cp, -0.5, 0.5, color="black")

#         ax2.set_title('Ground Truth State Sequence')
#         ax2.set_yticks([])
#         ax2.imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
#         ax3.set_yticks([])
#         ax3.set_title('Predicted State Sequence by Time2State')
#         ax3.imshow(prediction.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_save_path, file_name+'.png'))
#         plt.close()

# for dataset in dataset_list:
#     plot_result_of(dataset)
