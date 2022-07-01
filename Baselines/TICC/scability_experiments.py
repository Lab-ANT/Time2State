from TICC_solver import TICC
import numpy as np
import pandas as pd
import scipy.io as io
import time
import scipy.io

def effect_of_length():
    time_list = []
    data = scipy.io.loadmat('../data/ActRecTut/subject1_walk/data.mat')
    data = data['data']
    data = np.concatenate([data[:,:4] for x in range(10)])
    # print(data.shape)
    for i in range(1,21):
        ticc = TICC(window_size=1, number_of_clusters=5, lambda_parameter=1e-4, beta=2200, maxIters=1, threshold=2e-4,
                write_out_file=False, prefix_string="output_folder/", num_proc=10)
        time_start=time.time()
        ticc.fit_transform(data[:i*10000,:])
        time_end=time.time()
        print(time_end-time_start)
        time_list.append(time_end-time_start)
    print(np.array(time_list).round(2))

def effect_of_dimension():
    time_list = []
    data = scipy.io.loadmat('../data/ActRecTut/subject1_walk/data.mat')
    data = data['data']
    for i in range(2,21):
        _data = data[:,:i]
        print(_data.shape)
        time_start=time.time()
        ticc = TICC(window_size=1, number_of_clusters=6, lambda_parameter=1e-3, beta=2200, maxIters=1, threshold=1e-3,
                write_out_file=False, prefix_string="output_folder/", num_proc=10)
        (cluster_assignment, cluster_MRFs) = ticc.fit_transform(_data)
        time_end=time.time()
        print(time_end-time_start)
        time_list.append(time_end-time_start)
    print(np.array(time_list).round(2))

# effect_of_length()
effect_of_dimension()