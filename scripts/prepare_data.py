import pandas as pd
import numpy as np
import os
import scipy.io
from TSpy.utils import *
from TSpy.dataset import *

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

data_path = os.path.join(os.path.dirname(__file__), '../data/')

def USC_HAD_AutoPlait():
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            print(data.shape)
            if not os.path.exists(data_path+'USC-HAD_for_AutoPlait'):
                os.makedirs(data_path+'USC-HAD_for_AutoPlait')
            np.savetxt(data_path+'USC-HAD_for_AutoPlait/s'+str(subject)+'t'+str(target)+'.txt', data, delimiter=' ')

    with open(data_path+'USC-HAD_for_AutoPlait/list', 'w') as f:
        for subject in range(1,15):
            for target in range(1,6):
                f.writelines(data_path+'/USC-HAD_for_AutoPlait/s'+str(subject)+'t'+str(target)+'.txt\n')

def ActRecTut_AutoPlait():
    for s in ['subject1_walk', 'subject2_walk']:
        data = scipy.io.loadmat('../data/ActRecTut/'+s+'/data.mat')
        groundtruth = data['labels'].flatten()
        data = data['data'][:,:10]
        np.savetxt('../data/ActRecTut/data_for_AutoPlait/'+s+'.txt', groundtruth)
        np.savetxt('../data/ActRecTut/data_for_AutoPlait/'+s+'_groundtruth.txt', groundtruth)
    # with open('../data/ActRecTut/data_for_AutoPlait/list', 'w') as f:
    #     for i in range(1,21):
    #         f.writelines('../data/ActRecTut/data_for_AutoPlait/'+s+'.txt\n')

def PAMAP2_AutoPlait():
    if not os.path.exists(data_path+'/PAMAP2/data_for_AutoPlait'):
        os.makedirs(data_path+'/PAMAP2/data_for_AutoPlait')
    for i in range(1,10):
        dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(i)+'.dat')
        data = np.loadtxt(dataset_path)
        groundtruth = np.array(data[:,1],dtype=int)
        hand_acc = data[:,4:7]
        chest_acc = data[:,21:24]
        ankle_acc = data[:,38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        np.savetxt(data_path+'/PAMAP2/data_for_AutoPlait/subject10'+str(i)+'.txt', data)
        np.savetxt(data_path+'/PAMAP2/data_for_AutoPlait/groundtruth_subject10'+str(i)+'.txt', groundtruth)
    with open(data_path+'/PAMAP2/data_for_AutoPlait/list', 'w') as f:
        for i in range(1,21):
            f.writelines(data_path+'/PAMAP2/data_for_AutoPlait/subject10'+str(i)+'.txt\n')

def synthetic_for_AutoPlait_length():
    data = np.loadtxt(data_path+'/synthetic_data_for_segmentation/test0.csv', delimiter=',')
    data = np.concatenate([data[:,:4] for x in range(15)])
    for i in range(1,21):
        np.savetxt(data_path+'/effect_of_length/'+str(i)+'.txt', data[:i*10000,:].round(4))
    with open(data_path+'/effect_of_length/list', 'w') as f:
        for i in range(1,21):
            f.writelines(data_path+'/effect_of_length/'+str(i)+'.txt\n')

def synthetic_for_AutoPlait_dim():
    data = np.loadtxt(data_path+'/synthetic_data_for_segmentation/test0.csv', delimiter=',')
    data = np.hstack([data, data, data, data, data])
    data = np.vstack([data, data])
    data = data[:30000,:]
    for i in range(1,21):
        np.savetxt('../data/effect_of_dim/'+str(i)+'.txt', data[:,:i].round(4))
    with open('../data/effect_of_dim/list', 'w') as f:
        for i in range(1,21):
            f.writelines('../data/effect_of_dim/'+str(i)+'.txt\n')


# synthetic_for_AutoPlait_length()
# synthetic_for_AutoPlait_dim()
PAMAP2_AutoPlait()
# ActRecTut_AutoPlait()
USC_HAD_AutoPlait()