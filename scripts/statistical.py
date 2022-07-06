from cProfile import label
import numpy as np
import os
import pandas as pd

from TSpy.label import seg_to_label
from TSpy.utils import len_of_file
from TSpy.dataset import load_USC_HAD
import scipy.io

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')

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

def calculate_num_segs(X):
    pre = X[0]
    num_segs = 0
    for e in X:
        if e != pre:
            num_segs+=1
            pre = e
    return num_segs

def exp_on_ActRecTut():
    len_list = []
    state_num_list = []
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        dataset_path = os.path.join(script_path, '../data/ActRecTut/')
        f_path = dataset_path+dir_name+'/data.mat'
        data = scipy.io.loadmat(f_path)
        groundtruth = data['labels'].flatten()
        length = data['data'].shape[0]
        state_num = len(set(groundtruth))
        num_segs = calculate_num_segs(groundtruth)
        print('length: %d, state_num: %d, num segs: %s'%(length, state_num, num_segs))
        len_list.append(length)
        state_num_list.append(state_num)
    print('AVG ---- length: %f, state_num: %f'%(np.mean(len_list), np.mean(state_num_list)))

def PAMAP2():
    len_list = []
    state_num_list = []
    for i in range(1,10):
        dataset_path = os.path.join(script_path, '../data/PAMAP2/Protocol/subject10'+str(i)+'.dat')
        data = np.loadtxt(dataset_path)
        groundtruth = np.array(data[:,1],dtype=int)
        num_segs = calculate_num_segs(groundtruth)
        length = data.shape[0]
        state_num = len(set(groundtruth))
        print('length: %d, state_num: %d, seg_num: %d'%(length, state_num, num_segs))
        len_list.append(length)
        state_num_list.append(state_num)
    print('AVG ---- length: %f, state_num: %f'%(np.mean(len_list), np.mean(state_num_list)))

def MoCap():
    len_list = []
    state_num_list = [4,8,7,6,9,5,4,4,3]
    dataset_path = os.path.join(script_path, '../data/MoCap/4d/')
    for fname in os.listdir(dataset_path):
        file_path = dataset_path+fname
        data = np.loadtxt(file_path)
        length = data.shape[0]
        print('length: %d'%(length))
        len_list.append(length)
        num_segs = calculate_num_segs(seg_to_label(dataset_info[fname]['label']))
        print(num_segs)
    print('AVG ---- length: %f, state_num: %f'%(np.mean(len_list), np.mean(state_num_list)))

def UCR_SEG():
    len_list = []
    state_num_list = []
    dataset_path = os.path.join(script_path, '../data/UCR-SEG/UCR_datasets_seg/')
    f_list = os.listdir(dataset_path)
    max_len = 0
    min_len = 100000
    min_num_states = 100
    max_num_states = 0
    for fname in f_list:
        info_list = fname[:-4].split('_')
        f = info_list[0]
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        groundtruth = seg_to_label(seg_info)[:-1]
        length = len_of_file(dataset_path+fname)
        n_states = len(seg_info)
        if length > max_len:
            max_len = length
        if length < min_len:
            min_len = length
            
        if n_states > max_num_states:
            max_num_states = n_states
        if n_states < min_num_states:
            min_num_states = n_states

        state_num = len(set(groundtruth))
        print('length: %d, state_num: %d'%(length, state_num))
        len_list.append(length)
        state_num_list.append(state_num)
    print('AVG ---- length: %f, state_num: %f, total: %d'%(np.mean(len_list), np.mean(state_num_list), len(f_list)))
    print('MAX length: %d, MIN length: %d, total: %d'%(max_len, min_len, len(f_list)))
    print('MAX num states: %d, MIN num states: %d'%(max_num_states, min_num_states))

def exp_on_USC_HAD():
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            print(len(set(groundtruth)),calculate_num_segs(groundtruth)+1)

import matplotlib.pyplot as plt
def synthetic():
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation2/test')
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        print(len(data),len(set(groundtruth)),calculate_num_segs(groundtruth)+1)
        # plt.plot(data)
        # plt.step(np.arange(len(groundtruth)),groundtruth)
        # plt.savefig('2.png')


# exp_on_USC_HAD()
# synthetic()
# PAMAP2()
# MoCap()
UCR_SEG()
# exp_on_ActRecTut()