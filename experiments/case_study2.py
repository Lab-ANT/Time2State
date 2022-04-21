import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from TSpy.corr import state_correlation, cluster_corr
import os
import sys
sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *

# The traffic dataset.
# data_path = os.path.join(os.path.dirname(__file__), '../data/multivariate-time-series-data/traffic/traffic.txt')
# result_path = os.path.join(os.path.dirname(__file__), '../output/case5/')

# The electricity dataset.
data_path = os.path.join(os.path.dirname(__file__), '../data/multivariate-time-series-data/electricity.txt')
result_path = os.path.join(os.path.dirname(__file__), '../output/electricity128_2/')
# result_path = os.path.join(os.path.dirname(__file__), '../output/electricity128_3/')
# result_path = os.path.join(os.path.dirname(__file__), '../output/electricity128_adaptive/')

def scale(data):
    max_ = np.max(data)
    min_ = np.mean(data)
    data = 2*(data-min_)/(max_-min_)
    return data

def plot_sample(id_list, show_label=True):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=len(id_list), sharex=True, figsize=(18,3))
    for i, id in enumerate(id_list):
        dta = np.loadtxt(result_path+str(id)+'.txt', delimiter=' ')
        data = dta[:,0]
        label = dta[:,1]
        data = scale(data)
        ax[i].plot(data, color= '#348ABC')
        if show_label:
            ax[i].step(np.arange(len(label)),label,lw=2)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig('fig.png')
    plt.show()

def segment(start_idx, end_idx):
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    params_LSE['compared_length'] = 256
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    dta = np.loadtxt(data_path, delimiter=',')
    for i in range(start_idx, end_idx):
        data = np.array(dta[:,i], dtype=float).reshape(len(dta[:,i]), 1)
        # print(data.shape)
        t2s = Time2State(256, 50, CausalConv_LSE_Adaper(params_LSE), KMeansClustering(3)).fit(data, 256, 50)
        prediction = t2s.state_seq
        estimated_state_num = len(set(prediction))
        result = np.vstack([data.flatten(), prediction.flatten()]).T
        np.savetxt(result_path + str(i)+'.txt', result)
        print('sample %d done, estimated state num is %d'%(i,estimated_state_num))

def calculate_pearson_correlation_matrix(start_idx, end_idx):
    seq_list = []
    for i in range(start_idx, end_idx):
        data = np.loadtxt(result_path+str(i)+'.txt')[:,0].flatten()
        seq_list.append(data)
    seq_list = np.array(seq_list)
    df = pd.DataFrame(seq_list.T)
    correlation_matrix = df.corr('pearson').to_numpy()
    np.savetxt(result_path+'pearson.txt', correlation_matrix)

def calculate_spearman_correlation_matrix(start_idx, end_idx):
    seq_list = []
    for i in range(start_idx, end_idx):
        data = np.loadtxt(result_path+str(i)+'.txt')[:,0].flatten()
        seq_list.append(data)
    seq_list = np.array(seq_list)
    print(seq_list.shape)

    df = pd.DataFrame(seq_list.T)
    correlation_matrix = df.corr('spearman').to_numpy()
    np.savetxt(result_path+'spearman.txt', correlation_matrix)

def calculate_state_correlation_matrix(start_idx, end_idx):
    state_seq_list = []
    for i in range(start_idx, end_idx):
        prediction = np.loadtxt(result_path+str(i)+'.txt', delimiter=' ')[:,1].flatten().astype(int)
        state_seq_list.append(prediction)
    correlation_matrix = state_correlation(state_seq_list)
    np.savetxt(result_path+'state.txt', correlation_matrix)

def show_corr_matrix(metric='state'):
    if metric == 'state':
        fname = 'state.txt'
    elif metric == 'pearson':
        fname = 'pearson.txt'
    else:
        fname = 'spearman.txt'   
        
    if not os.path.exists(result_path + fname):
        if metric == 'state':
            calculate_state_correlation_matrix(0, 301)
        elif metric == 'pearson':
            calculate_pearson_correlation_matrix(0, 301)
        else:
            calculate_spearman_correlation_matrix(0, 301)
    
    correlation_matrix = np.loadtxt(result_path + fname)#[:50,:50]
    correlation_matrix, label, idx = cluster_corr(correlation_matrix)
    print(idx[190:200])
    sns.heatmap(correlation_matrix, cmap='gray')
    plt.savefig('output/matrix_'+fname[:-4]+'.png')
    plt.show()


def find_top_K_in_matrix(id, k, metric='state'):
    if metric == 'state':
        fname = 'state_correlation_matrix.txt'
    elif metric == 'pearson':
        fname = 'pearson.txt'
    else:
        fname = 'spearman.txt'

    if not os.path.exists(result_path + fname):
        if metric == 'state':
            calculate_state_correlation_matrix(0, 301)
        elif metric == 'pearson':
            calculate_pearson_correlation_matrix(0, 301)
        else:
            calculate_spearman_correlation_matrix(0, 301)
    
    correlation_matrix = np.loadtxt(result_path + fname)
    corr_list = correlation_matrix[id,:].flatten()
    sorted_idx = np.argsort(corr_list)[::-1]
    # return top k+1, the 1st must be id itself.
    print(sorted_idx[:k+1])
    print(np.round([corr_list[idx] for idx in sorted_idx[:k+1]], 2))
    return sorted_idx[:k+1]

# segment(0, 301)

# metrics: 'state', 'pearson', 'spearman'
# find top-k most correlated time series.
# 6, 60, 24, 80, 62, 76, 86, 74. 3, 
# plot_sample([283,281,280,230,279,275,273,272,271,268])
top_k_idx = find_top_K_in_matrix(85, 5, metric='state')
plot_sample(top_k_idx, show_label=False)

# show correlation matrix.
show_corr_matrix(metric='spearman')
show_corr_matrix(metric='state')
show_corr_matrix(metric='pearson')