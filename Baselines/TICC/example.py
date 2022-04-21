from TICC_solver import TICC
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import steval.segmentation as ss
import steval.label as sl
import os
import scipy.io as io
from sklearn import metrics
import steval.segmentation as ss
import steval.label as sl

# def load_data(input_file):
#     Data = np.loadtxt(input_file, delimiter=' ',dtype=np.int)
#     (m, n) = Data.shape  # m: num of observations, n: size of observation vector
#     return Data, m, n

def load_data(input_file):
    Data = pd.read_csv(input_file, sep=',', usecols=range(0,13))
    Data = np.array(Data)
    (m, n) = Data.shape  # m: num of observations, n: size of observation vector
    return Data, m, n

def run(in_path, out_path, n_clusters):
    ticc = TICC(window_size=3, number_of_clusters=6, lambda_parameter=1e-4, beta=2200, maxIters=10, threshold=2e-4,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)
    (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=in_path)
    cluster_assignment = np.array(cluster_assignment, dtype=np.int)
    df = pd.DataFrame(cluster_assignment)
    df.to_csv(out_path, index=None)
    # np.savetxt(out_path,cluster_assignment)

    # groundtruth = sl.seg_to_label({588:1,1200:2,2006:1,2530:3,3282:1,4048:4,4579:3})[3:]
    # print(cluster_assignment.shape, groundtruth.shape)
    # print(ss.ARI(groundtruth,cluster_assignment),ss.AMI(groundtruth,cluster_assignment))


MoCap = {'amc_86_01':{'n_segs':4, 'label':{588:1,1200:2,2006:1,2530:3,3282:1,4048:4,4579:3}},
         'amc_86_02':{'n_segs':8, 'label':{1009:1,1882:2,2677:3,3158:4,4688:5,5963:1,7327:6,8887:7,9632:8,10617:1}},
         'amc_86_03':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
         'amc_86_07':{'n_segs':6, 'label':{1060:1,1897:2,2564:3,3665:2,4405:3,5169:4,5804:5,6962:1,7806:6,8702:1}},
         'amc_86_08':{'n_segs':9, 'label':{1062:1,1904:2,2661:3,3282:4,3963:5,4754:6,5673:7,6362:5,7144:8,8139:9,9206:1}},
         'amc_86_09':{'n_segs':5, 'label':{921:1,1275:2,2139:3,2887:4,3667:5,4794:1}},
         'amc_86_10':{'n_segs':4, 'label':{2003:1,3720:2,4981:1,5646:3,6641:4,7583:1}},
         'amc_86_11':{'n_segs':4, 'label':{1231:1,1693:2,2332:3,2762:2,3386:4,4015:3,4665:2,5674:1}},
         'amc_86_14':{'n_segs':3, 'label':{671:1,1913:2,2931:1,4134:3,5051:1,5628:2,6055:3}},}

def exp_on_ActRecTut():
    ticc = TICC(window_size=3, number_of_clusters=12, lambda_parameter=1e-4, beta=2200, maxIters=3, threshold=2e-4,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)
    (cluster_assignment, cluster_MRFs) = ticc.fit(input_file='../../data/ActRecTut/subject1_gesture/data_labeled.csv')
    cluster_assignment = np.array(cluster_assignment, dtype=np.int)
    df = pd.DataFrame(cluster_assignment)
    df.to_csv('../../data/ActRecTut/subject1_gesture/result.csv', index=None)
    # np.savetxt(out_path,cluster_assignment)

def score():
    df = pd.read_csv('../../data/ActRecTut/subject1_gesture/result.csv')
    prediction = np.array(df).flatten()
    df2 = pd.read_csv('../../data/ActRecTut/subject1_gesture/data_labeled.csv',skiprows=1,usecols=[14])
    gt = np.array(df2).flatten()[2:10000]
    print(ss.ARI(gt,prediction),ss.AMI(gt,prediction))

def get_groundtruth():
    prefix = '../../data/USC-HAD/subject3/'
    fname_prefix = 'a'
    fname_postfix = 't5.mat'

    series_list1 = []
    series_list2 = []
    series_list3 = []

    label_seg = {}
    total_length = 0
    for i in range(1,13):
        data = io.loadmat(prefix+fname_prefix+str(i)+fname_postfix)
        series = data['sensor_readings'][:,5]
        series_list1.append(series)
        series = data['sensor_readings'][:,4]
        series_list2.append(series)
        series = data['sensor_readings'][:,3]
        series_list3.append(series)
        total_length += len(data['sensor_readings'][:,3])
        label_seg[total_length]=i
    

    result1 = np.concatenate(series_list1)
    result2 = np.concatenate(series_list2)
    result3 = np.concatenate(series_list3)
    data = np.array([result1,result2,result3]).T

    import steval.label as sl

    fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
    # ax[0].plot(result1)
    # ax[0].plot(result2)
    # ax[0].plot(result3)

    groundtruth = sl.seg_to_label(label_seg)[:-3]
    # ax[1].step(np.arange(len(groundtruth)),groundtruth)
    prediction = pd.read_csv('result.csv').to_numpy().flatten() #df.read_csv('result.csv', index=None)
    print(groundtruth.shape,prediction.shape)
    print(ss.ARI(groundtruth,prediction),ss.AMI(groundtruth,prediction),metrics.f1_score(ss.adjust_label(groundtruth),ss.adjust_label(prediction)+1,average='macro'))
    return groundtruth

import time
if __name__ == '__main__':

    ticc = TICC(window_size=3, number_of_clusters=13, lambda_parameter=1e-4, beta=2200, maxIters=3, threshold=2e-4,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)
    # (cluster_assignment, cluster_MRFs) = ticc.fit(input_file='test.csv')

    file_list = []
    time_list = []
    for fname in file_list:
        (cluster_assignment, cluster_MRFs) = ticc.fit(input_file='test.csv')

    # ticc = TICC(window_size=3, number_of_clusters=13, lambda_parameter=1e-4, beta=2200, maxIters=3, threshold=2e-4,
    #             write_out_file=False, prefix_string="output_folder/", num_proc=1)
    # (cluster_assignment, cluster_MRFs) = ticc.fit(input_file='test.csv')
    # cluster_assignment = np.array(cluster_assignment, dtype=np.int)
    # df = pd.DataFrame(cluster_assignment)
    # df.to_csv('result.csv', index=None)
    # plt.step(np.arange(len(cluster_assignment)),cluster_assignment)
    # plt.savefig('step.png')
    # np.savetxt(out_path,cluster_assignment)
    # get_groundtruth()

    # Mocap
    # in_dir = '../../data/MoCap/4d/'
    # out_dir = '../../data/MoCap/result_TICC/'
    # for f in os.listdir(in_dir):
    #     # print(MoCap[f[:-3]]['n_segs'])
    #     run(in_dir+f, out_dir+f[:-3]+'.seg',MoCap[f[:-3]]['n_segs'])

    # # Gesture
    # in_dir = '../../data/MoCap/4d/'
    # out_dir = '../../data/MoCap/result_TICC/'
    # for f in os.listdir(in_dir):
    #     # print(MoCap[f[:-3]]['n_segs'])
    #     run(in_dir+f, out_dir+f[:-3]+'.seg',MoCap[f[:-3]]['n_segs'])

    # score()
    # exp_on_ActRecTut()

    # np.savetxt('Results2.txt', cluster_assignment, fmt='%d', delimiter=',')
    # times_series_arr, time_series_rows_size, time_series_col_size = load_data(fname)
    # print(time_series_col_size,time_series_rows_size)
    # print(times_series_arr)
    
    # plt.plot(times_series_arr)
    # plt.plot(cluster_assignment)
    # plt.savefig('result2.png')

    # import matplotlib.pyplot as plt

    # data = pd.read_csv('example_data.txt',header=None)
    # print(data.shape)
    # plt.plot(data.iloc[:,1])
    # plt.plot(data.iloc[:,2])
    # plt.plot(data.iloc[:,3])
    # plt.show()
    
    