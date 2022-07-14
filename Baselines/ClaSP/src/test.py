import os, sys, time
sys.path.append("Baselines/ClaSP/")

import numpy as np
# np.random.seed(1379)

import pandas as pd
import scipy

from TSpy.utils import len_of_file
from TSpy.label import *
from TSpy.eval import evaluate_clustering
from TSpy.utils import *
from TSpy.dataset import *

from src.clasp import extract_clasp_cps_from_multivariate_ts
# from dtw import dtw

import os
from tslearn.clustering import TimeSeriesKMeans

''' Define relative path. '''
script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../../../data')
output_path = os.path.join(script_path, '../../../results/output_ClaSP')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

dataset_info = {'amc_86_01.4d':{'n_states':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02.4d':{'n_states':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03.4d':{'n_states':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07.4d':{'n_states':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08.4d':{'n_states':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09.4d':{'n_states':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10.4d':{'n_states':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11.4d':{'n_states':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14.4d':{'n_states':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

# ''' Define DTW '''
# manhattan_distance = lambda x, y: np.abs(x - y)
# def dtw_distance(x,y):
#     d, cost_matrix, acc_cost_matrix, path = dtw(x,y,dist=manhattan_distance)
#     return d

def padding(X, max_length):
    if len(X.shape) == 1:
        dim = 1
        X = np.expand_dims(X, axis=1)
    else:
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
    return result

def calculate_seg_len_list(found_cps,length):
    length_list = []
    prev = 0
    for cp in found_cps:
        length_list.append(int(cp-prev))
        prev=cp
    length_list.append(length-prev)
    return length_list

def cluster_segs(X, found_cps, n_states):
    seg_list = []
    length = len(X)
    start = 0
    for cp in found_cps:
        seg_list.append(X[start:cp])
        start = cp
    seg_list.append(X[start:length])
    segments = padding_and_stack(seg_list)
    # dtw, euclidean, softdtw
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_states,metric='euclidean').fit(segments)
    labels = ts_kmeans.labels_
    seg_label_list = []
    length_list = calculate_seg_len_list(found_cps,length)

    for label, length in zip(labels, length_list):
        seg_label_list.append(label*np.ones(length))
    result = np.hstack(seg_label_list)
    return result

def run_clasp(X, window_size, num_cps, n_states, offset):
    profile_, found_cps, _ = extract_clasp_cps_from_multivariate_ts(X, window_size, num_cps, offset)
    found_cps.sort()
    prediction = cluster_segs(X, found_cps, n_states)
    return prediction

def exp_on_synthetic(win_size, num_seg, verbose=False):
    out_path = os.path.join(output_path,'synthetic2')
    create_path(out_path)
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation2/test')
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()

        prediction = run_clasp(data, win_size, num_seg, 5)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,str(i)), result)

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_ActRecTut(win_size, num_seg, verbose=False):
    out_path = os.path.join(output_path,'ActRecTut')
    create_path(out_path)
    score_list = []
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
        data = scipy.io.loadmat(dataset_path)
        groundtruth = data['labels'].flatten()
        groundtruth = reorder_label(groundtruth)
        data = data['data'][:,0:10]
        data = normalize(data)
        n_states = len(set(groundtruth))
        prediction = run_clasp(data, win_size, num_seg, n_states)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,dir_name), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(dir_name, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_MoCap(win_size, num_seg, offset, verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    out_path = os.path.join(output_path,'MoCap2')
    create_path(out_path)
    score_list = []
    for fname in os.listdir(base_path):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        n_states=dataset_info[fname]['n_states']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        prediction = run_clasp(data, win_size, num_seg, n_states, offset)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_UCR_SEG(win_size, num_seg, offset, verbose=False):
    out_path = os.path.join(output_path,'UCR-SEG2')
    create_path(out_path)
    score_list = []
    dataset_path = os.path.join(data_path,'UCR-SEG/UCR_datasets_seg/')
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
        df = pd.read_csv(os.path.join(dataset_path,fname), header=None)
        data = df.to_numpy().flatten()
        prediction = run_clasp(data, window_size, num_seg, len(seg_info), offset)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,fname[:-4]), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_USC_HAD(win_size, num_seg, offset, verbose=False):
    out_path = os.path.join(output_path,'USC-HAD2')
    create_path(out_path)
    score_list = []
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            data = normalize(data)
            groundtruth = groundtruth
            n_states = len(set(groundtruth))

            prediction = run_clasp(data, win_size, num_seg, n_states, offset)

            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path,'s%d_t%d'%(subject,target)), result)

            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

def exp_on_PAMAP2(win_size, num_seg, offset, verbose=False):
    out_path = os.path.join(output_path,'PAMAP2')
    create_path(out_path)
    score_list = []
    for i in range(1, 9):
        dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(i)+'.dat')
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:,1],dtype=int)
        hand_acc = data[:,4:7]
        chest_acc = data[:,21:24]
        ankle_acc = data[:,38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        data = normalize(data)
        
        groundtruth = groundtruth[::10]
        n_states = len(set(groundtruth))
        prediction = run_clasp(data[::10], win_size, num_seg, n_states, offset)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,'10'+str(i)), result)
        
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

if __name__ == '__main__':
    # exp_on_UCR_SEG(50, 40, 0.05, verbose=True)
    # exp_on_ActRecTut(50, 40, verbose=True)
    exp_on_USC_HAD(50, 40, 0.02, verbose=True)
    # exp_on_MoCap(50, 40, 0.04, verbose=True)
    # exp_on_synthetic(100, 40, verbose=True)
    # exp_on_PAMAP2(50, 40, 0.02, verbose=True)