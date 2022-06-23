import os, sys, time
from turtle import st

from sklearn import cluster, metrics

sys.path.append("Baselines/ClaSP/")

import numpy as np
# np.random.seed(1379)

import pandas as pd

from TSpy.utils import len_of_file
from TSpy.label import seg_to_label
from TSpy.eval import evaluate_clustering

from src.clasp import extract_clasp_cps, extract_clasp_cps_from_multivariate_ts
from dtw import dtw

import matplotlib.pyplot as plt
import os

''' Define relative path. '''
script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../../../data')

dataset_info = {'amc_86_01.4d':{'n_segs':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02.4d':{'n_segs':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03.4d':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07.4d':{'n_segs':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08.4d':{'n_segs':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09.4d':{'n_segs':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10.4d':{'n_segs':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11.4d':{'n_segs':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14.4d':{'n_segs':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

''' Define DTW '''
manhattan_distance = lambda x, y: np.abs(x - y)
def dtw_distance(x,y):
    d, cost_matrix, acc_cost_matrix, path = dtw(x,y,dist=manhattan_distance)
    return d

from tslearn.clustering import TimeSeriesKMeans

# ts_kmeans = TimeSeriesKMeans(n_clusters=2,metric='dtw').fit(d)
# print(kmeans.labels_)

def padding(X, max_length):
    _, dim = X.shape
    data = np.zeros((max_length, dim))
    length = len(X)
    data[:length] = X
    return data

def padding_and_stack(seg_list):
    max_length = 0
    for seg in seg_list:
        length = len(seg)
        if length > max_length:
            max_length = length
    new_seg_list = []
    for seg in seg_list:
        new_seg_list.append(padding(seg, max_length))
    result = np.stack(new_seg_list)
    # print(result.shape)
    return result

def calculate_seg_len_list(found_cps,length):
    length_list = []
    prev = 0
    for cp in found_cps:
        length_list.append(int(cp-prev))
        prev=cp
    length_list.append(length-prev)
    return length_list

def cluster_segs(X, found_cps):
    seg_list = []
    length = len(X)
    start = 0
    for cp in found_cps:
        seg_list.append(X[start:cp])
        start = cp
    seg_list.append(X[start:length])
    segments = padding_and_stack(seg_list)
    ts_kmeans = TimeSeriesKMeans(n_clusters=5,metric='dtw').fit(segments)
    labels = ts_kmeans.labels_
    print(ts_kmeans.labels_.shape,ts_kmeans.labels_)
    seg_label_list = []
    length_list = calculate_seg_len_list(found_cps,length)
    print(length_list, np.sum(np.array(length_list)))

    for label, length in zip(labels, length_list):
        seg_label_list.append(label*np.ones(length))
    result = np.hstack(seg_label_list)
    print(result.shape)
    return result

def run_clasp(X, window_size, num_cps):
    profile_, found_cps, _ = extract_clasp_cps_from_multivariate_ts(X, window_size, num_cps)
    found_cps.sort()
    prediction = cluster_segs(X, found_cps)
    return prediction

def exp_on_synthetic(verbose=False):
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation/test')
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()

        prediction = run_clasp(data, 50, 20)

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_MoCap(verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    score_list = []
    for fname in os.listdir(base_path):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        # print(len(groundtruth))
        prediction = run_clasp(data, 50, 20)

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_UCR_SEG(verbose=False):
    score_list = []
    dataset_path = os.path.join(data_path,'UCR-SEG/')
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        # f = info_list[0]
        window_size = int(info_list[1])
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        groundtruth = seg_to_label(seg_info)
        
        df = pd.read_csv(dataset_path+fname, header=None)
        data = df.to_numpy().flatten()
        # print(fname, data.shape, len(seg_info)-1, seg_info)
        # K = len(seg_info)-1
        K = 10
        profile_, found_cps, _ = extract_clasp_cps(data, window_size, K)
        print(found_cps)
        predicted_seg_info = {}
        j = 0
        found_cps.sort()
        for cp in found_cps:
            predicted_seg_info[cp]=j
            j+=1
        predicted_seg_info[len_of_file(dataset_path+fname)]=j
        prediction = seg_to_label(predicted_seg_info)

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

if __name__ == '__main__':
    # exp_on_UCR_SEG(verbose=True)
    exp_on_MoCap(verbose=True)
    # exp_on_synthetic(verbose=True)
    pass