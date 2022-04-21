import numpy as np
import matplotlib.pyplot as plt
import os
import re
from numpy.core.fromnumeric import mean
import pandas as pd
from sklearn import metrics
import scipy.io as io

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

    groundtruth = sl.seg_to_label(label_seg)
    return groundtruth

# JMeasure
def JMeasure(seq1, seq2, lag):
    result_NMI=metrics.normalized_mutual_info_score(seq1, seq2)
    return result_NMI

def read_original_data(path):
    data = pd.read_csv(path, index_col=False, names=['col0'], header=None, sep=' ')
    # print(data)
    return np.array(data)

def read_result_dir(path, original_file_path):
    results = os.listdir(path)
    # TODO: fix this but.
    results = [s if re.match('segment.\d', s) != None else None for s in results]
    results = np.array(results)
    idx = np.argwhere(results!=None)
    results = results[idx].flatten()

    dta = pd.read_csv(original_file_path, names=['col0','col1','col2','col3'], index_col=False, header=None, sep=" ")
    length = len(dta)
    # label = np.zeros(length)
    label = np.zeros(26745)

    l = 0
    for r in results:
        data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
        start = data.col0
        end = data.col1
        for s, e in zip(start,end):
            label[s:e+1]=l
        l+=1
    return label

def decompose_state_sequence(state_seq):
    result = []
    # state_seq = np.array(state_seq)
    seq_len = len(state_seq)
    state_num = len(set(state_seq))
    for index in range(state_num):
        single_state_seq = np.zeros(seq_len)
        single_state_seq[np.argwhere(state_seq==index)] = 1
        result.append(single_state_seq)
    return result

    # label_list = []
    # dta = pd.read_csv(original_file_path, names=['col0','col1','col2','col3'], index_col=False, header=None, sep=" ")
    # length = len(dta)

    # l = 0
    # for r in results:
    #     labels = np.zeros(length)
    #     data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
    #     start = data.col0
    #     end = data.col1
    #     for s, e in zip(start,end):
    #         labels[s:e+1]=l
    #     l+=1
    #     label_list.append(labels)
    # return label_list
def order_state_sequence(seq):
    pass

def plot_from_dir(state_seq_path, raw_data_path):
    state_seq = read_result_dir(state_seq_path, raw_data_path)
    raw_data = read_original_data(raw_data_path)
    plt.axes([0.125,0.7,0.775,0.2])
    plt.plot(raw_data)
    plt.axis('off')

    plt.axes([0.125,0.6,0.775,0.05])
    plt.step(np.arange(len(state_seq)), state_seq, color='purple')
    plt.axis('off')

    plt.axes([0.125,0.45,0.775,0.1])
    single_seq_list = decompose_state_sequence(state_seq)

    i = 0
    for s in single_seq_list:
        plt.step(np.arange(len(s)), s+i)
        i+=2
    # plt.axes().get_yaxis().set_visible(False)
    plt.axis('off')
    plt.show()

# plot data.
# raw_data_path = '_dat/t4_4.1d'
# state_seq_path = '_out/dat_tmp/dat16/'

# state_seq = read_result_dir(state_seq_path, raw_data_path)
# raw_data = read_original_data(raw_data_path)
# plt.axes([0.125,0.7,0.775,0.2])
# plt.plot(raw_data)
# plt.axis('off')

# plt.axes([0.125,0.6,0.775,0.05])
# plt.step(np.arange(len(state_seq)), state_seq, color='purple')
# plt.axis('off')

# plt.axes([0.125,0.45,0.775,0.1])
# single_seq_list = decompose_state_sequence(state_seq)

# i = 0
# for s in single_seq_list:
#     plt.step(np.arange(len(s)), s+i)
#     i+=2
# # plt.axes().get_yaxis().set_visible(False)
# plt.axis('off')
# plt.show()



def corr(seq1, seq2):
    single_seq_list1 = decompose_state_sequence(seq1)
    single_seq_list2 = decompose_state_sequence(seq2)

    result_list = []
    for s1 in single_seq_list1:
        result = []
        for s2 in single_seq_list2:
            result.append(JMeasure(s1,s2,0))
        result_list.append(result)
    return result_list

raw_data_path = '_dat2/amc_86_02.4d'
state_seq_path = '_out2/dat_tmp/dat1/'

state_seq_1 = read_result_dir(state_seq_path, raw_data_path)
# plot_from_dir(state_seq_path, raw_data_path)

raw_data_path = '_dat2/amc_86_02.4d'
state_seq_path = '_out2/dat_tmp/dat1/'
state_seq_2 = read_result_dir(state_seq_path, raw_data_path)
# plot_from_dir(state_seq_path, raw_data_path)

# print(corr(state_seq_1,state_seq_2))
import steval.segmentation as ss
import steval.label as sl

# print(state_seq_1)

# dataset = {'amc_86_01.4d':{'n_segs':4, 'label':{588:1,1200:2,2006:1,2530:3,3282:1,4048:4,4579:3}},
#         'amc_86_02.4d':{'n_segs':8, 'label':{1009:0,1882:1,2677:0,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
#         'amc_86_03.4d':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
#         'amc_86_07.4d':{'n_segs':6, 'label':{1060:1,1897:2,2564:3,3665:2,4405:3,5169:4,5804:5,6962:1,7806:6,8702:1}},
#         'amc_86_08.4d':{'n_segs':9, 'label':{1062:1,1904:2,2661:3,3282:4,3963:5,4754:6,5673:7,6362:5,7144:8,8139:9,9206:1}},
#         'amc_86_09.4d':{'n_segs':5, 'label':{921:1,1275:2,2139:3,2887:4,3667:5,4794:1}},
#         'amc_86_10.4d':{'n_segs':4, 'label':{2003:1,3720:2,4981:1,5646:3,6641:4,7583:1}},
#         'amc_86_11.4d':{'n_segs':4, 'label':{1231:1,1693:2,2332:3,2762:2,3386:4,4015:3,4665:2,5674:1}},
#         'amc_86_14.4d':{'n_segs':3, 'label':{671:1,1913:2,2931:1,4134:3,5051:1,5628:2,6055:3}},
# }

MoCap = {'amc_86_01':{'n_segs':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02':{'n_segs':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07':{'n_segs':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08':{'n_segs':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09':{'n_segs':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10':{'n_segs':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11':{'n_segs':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14':{'n_segs':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

file_list = ['amc_86_01','amc_86_02','amc_86_03','amc_86_07','amc_86_08','amc_86_09','amc_86_10','amc_86_11','amc_86_14']

from sklearn import metrics
import matplotlib.pyplot as plt

# plt.hist([10,20,30,40,50], histtype="stepfilled", bins=30, alpha=0.8)
# plt.show()

# F1_list = []
# i = 1
# for f in file_list:
#     raw_data_path = '_dat2/'+f+'.4d'
#     state_seq_path = '_out2/dat_tmp/dat'+str(i)+'/'
#     state_seq = read_result_dir(state_seq_path, raw_data_path)
#     gt = sl.seg_to_label(MoCap[f]['label'])
#     prediction = ss.adjust_label(state_seq)
#     print(gt.shape, state_seq.shape)
#     f1 = metrics.f1_score(gt,prediction,average='macro')
#     F1_list.append(f1)
#     print(f1)
#     i+=1
# print('avg:%f', np.mean(F1_list))

# raw_data_path = '_dat2/amc_86_02.4d'
# state_seq_path = '_out_USC/dat_tmp/dat1/'
# state_seq = read_result_dir(state_seq_path, raw_data_path)
# gt = get_groundtruth()
# # print(gt.shape, state_seq.shape)
# print(metrics.f1_score(gt,ss.adjust_label(state_seq),average='macro'))

# i=1
# for f in ['hx','hy','hz','lhx','lhy','lhz','lwx','lwy','lwz','rhx','rhy','rhz']:
#     raw_data_path = 'a1_raw/'+f+'.1d'
#     state_seq_path = '_out2/dat_tmp/dat'+str(i)+'/'
#     state_seq_2 = read_result_dir(state_seq_path, raw_data_path)
#     plot_from_dir(state_seq_path, raw_data_path)
#     i+=1