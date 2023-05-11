'''
Created by Chengyu on 2023/4/20.
Convert data to FLOSS format.
'''

import os
import pandas as pd
from scipy import io
from TSpy.label import reorder_label, seg_to_label
import numpy as np
from TSpy.dataset import load_USC_HAD
from TSpy.utils import len_of_file
from miniutils import fill_nan, find_cp_from_state_seq

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, 'data/')
target_path = os.path.join(script_path, 'data/FLOSS_format/')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_ActRecTut(use_data):
    dir_name = use_data
    dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
    data = io.loadmat(dataset_path)
    groundtruth = data['labels'].flatten()
    groundtruth = reorder_label(groundtruth)
    data = data['data'][:,0:10]
    return data, groundtruth

def convert_USC_HAD(downsample_rate=None):
    create_path(os.path.join(target_path, 'USC-HAD'))
    for s in range(1,15):
        for t in range(1,6):
            data, groundtruth = load_USC_HAD(s, t, data_path)

            if downsample_rate is not None:
                data = data[::downsample_rate].round(4)
                groundtruth = groundtruth[::downsample_rate]

            df = pd.DataFrame(data)
            df_groundtruth = pd.DataFrame(groundtruth)
            cp_list = pd.DataFrame(find_cp_from_state_seq(groundtruth))
            # print(df.shape, cp_list.shape)
            df.to_csv(os.path.join(target_path, 'USC-HAD/s%.2dt%.2d.txt'%(s,t)), index=None, header=None)
            cp_list.to_csv(os.path.join(target_path, 'USC-HAD/s%.2dt%.2d.cp'%(s,t)), index=None, header=None)
            df_groundtruth.to_csv(os.path.join(target_path, 'USC-HAD/s%.2dt%.2d.label'%(s,t)), index=None, header=None)

def convert_ActRecTut(downsample_rate=None):
    create_path(os.path.join(target_path, 'ActRecTut'))
    for i in range(1,3):
        data, groundtruth = load_ActRecTut('subject%d_walk'%(i))

        if downsample_rate is not None:
            data = data[::downsample_rate]
            groundtruth = groundtruth[::downsample_rate]

        df = pd.DataFrame(data)
        df_groundtruth = pd.DataFrame(groundtruth)
        cp_list = pd.DataFrame(find_cp_from_state_seq(groundtruth))
        print(df.shape, cp_list.shape)
        df.to_csv(os.path.join(target_path, 'ActRecTut/subject%d_walk.txt'%(i)), index=None, header=None)
        cp_list.to_csv(os.path.join(target_path, 'ActRecTut/subject%d_walk.cp'%(i)), index=None, header=None)
        df_groundtruth.to_csv(os.path.join(target_path, 'ActRecTut/subject%d_walk.label'%(i)), index=None, header=None)

def convert_synthetic(downsample_rate=None):
    create_path(os.path.join(target_path, 'synthetic_data'))
    prefix = os.path.join(data_path, 'synthetic_data/test')
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy().round(4)
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()

        if downsample_rate is not None:
            data = data[::downsample_rate].round(4)
            groundtruth = groundtruth[::downsample_rate]

        df_groundtruth = pd.DataFrame(groundtruth)
        df = pd.DataFrame(data)
        cp_list = pd.DataFrame(find_cp_from_state_seq(groundtruth))
        print(df.shape, cp_list.shape)
        df.to_csv(os.path.join(target_path, 'synthetic_data/test%.2d.txt'%(i)), index=None, header=None)
        cp_list.to_csv(os.path.join(target_path, 'synthetic_data/test%.2d.cp'%(i)), index=None, header=None)
        df_groundtruth.to_csv(os.path.join(target_path, 'synthetic_data/test%.2d.label'%(i)), index=None, header=None)

def convert_MoCap(downsample_rate=None):
    create_path(os.path.join(target_path, 'MoCap'))
    file_list = os.listdir(os.path.join(data_path, 'MoCap'))
    for file_name in file_list:
        # print(file_name)
        df = pd.read_csv(os.path.join(data_path, 'Mocap/%s'%(file_name)), header=None)
        content = df.to_numpy()
        data = content[:,:4]
        groundtruth = content[:,4].astype(int).reshape(-1,1)

        df_groundtruth = pd.DataFrame(groundtruth)
        df = pd.DataFrame(data)
        cp_list = pd.DataFrame(find_cp_from_state_seq(groundtruth))
        print(df.shape, groundtruth.shape, cp_list.shape)
        df.to_csv(os.path.join(target_path, 'MoCap/%s.txt'%(file_name[:-4])), index=None, header=None)
        cp_list.to_csv(os.path.join(target_path, 'MoCap/%s.cp'%(file_name[:-4])), index=None, header=None)
        df_groundtruth.to_csv(os.path.join(target_path, 'MoCap/%s.label'%(file_name[:-4])), index=None, header=None)

def convert_PAMAP2(downsample_rate=None):
    create_path(os.path.join(target_path, 'PAMAP2'))
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

        if downsample_rate is not None:
            data = data[::downsample_rate]
            groundtruth = groundtruth[::downsample_rate]

        df_groundtruth = pd.DataFrame(groundtruth)
        df = pd.DataFrame(data)
        cp_list = pd.DataFrame(find_cp_from_state_seq(groundtruth))
        print(df.shape, cp_list.shape)
        df.to_csv(os.path.join(target_path, 'PAMAP2/subject10%d.txt'%(i)), index=None, header=None)
        cp_list.to_csv(os.path.join(target_path, 'PAMAP2/subject10%d.cp'%(i)), index=None, header=None)
        df_groundtruth.to_csv(os.path.join(target_path, 'PAMAP2/subject10%d.label'%(i)), index=None, header=None)

def convert_UCR_SEG():
    create_path(os.path.join(target_path, 'UCR-SEG'))
    dataset_path = os.path.join(data_path,'UCR-SEG/')
    for fname in os.listdir(dataset_path):
        print(fname)
        info_list = fname[:-4].split('_')
        cp_list = [int(cp) for cp in info_list[2:]]
        # window_size = int(info_list[1])
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)-1]=i
        num_state=len(seg_info)

        data = pd.read_csv(dataset_path+fname).to_numpy()
        groundtruth = seg_to_label(seg_info)

        # SimpleSynthetic_125_3000_5000.txt in the UCR-SEG dataset is a special case.
        if fname == 'SimpleSynthetic_125_3000_5000.txt':
            groundtruth = seg_to_label({3000:0, 5000:1, 8000:0})

        df_groundtruth = pd.DataFrame(groundtruth)
        df = pd.DataFrame(data)
        cp_list = pd.DataFrame(cp_list)
        print(df.shape, cp_list.shape)
        df.to_csv(os.path.join(target_path, 'UCR-SEG/%s.txt'%(fname[:-4])), index=None, header=None)
        cp_list.to_csv(os.path.join(target_path, 'UCR-SEG/%s.cp'%(fname[:-4])), index=None, header=None)
        df_groundtruth.to_csv(os.path.join(target_path, 'UCR-SEG/%s.label'%(fname[:-4])), index=None, header=None)

# For TSKMeans-euclidean
convert_PAMAP2(4)
convert_ActRecTut()
convert_USC_HAD(2)
convert_synthetic()
convert_MoCap()
convert_UCR_SEG()

# For TSKMeans-dtw
# convert_PAMAP2(10)
# convert_ActRecTut(5)
# convert_USC_HAD(5)
# convert_synthetic(5)
# convert_MoCap()
# convert_UCR_SEG()