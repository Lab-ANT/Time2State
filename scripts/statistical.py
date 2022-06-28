import numpy as np
import os
from TSpy.label import seg_to_label
from TSpy.utils import len_of_file
import scipy.io

script_path = os.path.dirname(__file__)

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
        dataset_path = '../data/ActRecTut/'+dir_name+'/data.mat'
        data = scipy.io.loadmat(dataset_path)
        groundtruth = data['labels'].flatten()
        length = data['data'].shape[0]
        state_num = len(set(groundtruth))
        print('length: %d, state_num: %d'%(length, state_num))
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
    for fname in os.listdir('../data/MoCap/4d/'):
        dataset_path = '../data/MoCap/4d/'+fname
        data = np.loadtxt(dataset_path)
        length = data.shape[0]
        print('length: %d'%(length))
        len_list.append(length)
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

# PAMAP2()
# MoCap()
UCR_SEG()
# exp_on_ActRecTut()