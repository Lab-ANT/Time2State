# Created by Chengyu on 2022/1/26

from TICC_solver import TICC
import numpy as np
import pandas as pd
import os
import scipy.io
from TSpy.eval import evaluate_clustering, evaluate_cut_point
from TSpy.label import seg_to_label
from TSpy.utils import *
from TSpy.dataset import *
import time

script_path = os.path.dirname(__file__)
# The root path of datasets.
data_path = os.path.join(script_path, '../../data/')
# Path for saving resutls.
output_path = os.path.join(script_path, 'output_TICC')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

dataset = {'amc_86_01.4d':{'n_segs':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02.4d':{'n_segs':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03.4d':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07.4d':{'n_segs':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08.4d':{'n_segs':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09.4d':{'n_segs':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10.4d':{'n_segs':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11.4d':{'n_segs':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14.4d':{'n_segs':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

def exp_on_ActRecTut(beta=2200, lambda_parameter=1e-3, threshold=1e-4, verbose=False):
    score_list = []
    dir_list = ['subject1_walk', 'subject2_walk']
    out_path = os.path.join(output_path,'ActRecTut')
    create_path(out_path)
    for dir_name in dir_list:
        for i in range(10):
            dataset_path = os.path.join(data_path, 'ActRecTut/'+dir_name+'/data.mat')
            data = scipy.io.loadmat(dataset_path)
            groundtruth = data['labels'].flatten()[:-2]
            num_state = len(set(groundtruth))
            data = data['data'][:,0:10]
            ticc = TICC(window_size=3, number_of_clusters=num_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=10, threshold=threshold,
                        write_out_file=False, prefix_string="output_folder/", num_proc=10)
            prediction, _ = ticc.fit_transform(data)
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path,dir_name+str(i)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(dir_name, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- , ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_MoCap(beta=2200, lambda_parameter=1e-3, threshold=1e-4, verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    out_path = os.path.join(output_path,'MoCap')
    create_path(out_path)
    score_list = []
    for fname in os.listdir(base_path):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        n_state=dataset[fname]['n_segs']
        ticc = TICC(window_size=5, number_of_clusters=n_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=10, threshold=threshold,
                write_out_file=False, prefix_string="output_folder/", num_proc=10)
        prediction, _ = ticc.fit_transform(data)
        groundtruth = seg_to_label(dataset[fname]['label'])[:-5]
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- , ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_UCR_SEG(beta=2200, lambda_parameter=1e-3, threshold=1e-4, verbose=False):
    score_list = []
    dataset_path = os.path.join(data_path,'UCR-SEG/UCR_datasets_seg/')
    out_path = os.path.join(output_path,'UCR-SEG')
    create_path(out_path)
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        # f = info_list[0]
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        df = pd.read_csv(dataset_path+fname)
        data = df.to_numpy()
        win_size=3
        num_state=len(seg_info)
        ticc = TICC(window_size=win_size, number_of_clusters=num_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=10, threshold=threshold,
                write_out_file=False, prefix_string="output_folder/", num_proc=10)
        prediction, _ = ticc.fit_transform(data)
        prediction = prediction.astype(int)
        groundtruth = seg_to_label(seg_info)[win_size:]
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

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

def exp_on_PAMAP2(beta=2200, lambda_parameter=1e-3, threshold=1e-4, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'PAMAP2')
    create_path(out_path)
    for i in range(1,9):
        dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(i)+'.dat')
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:,1],dtype=int)
        
        hand_acc = data[:,4:7]
        chest_acc = data[:,21:24]
        ankle_acc = data[:,38:41]
        # hand_gy = data[:,10:13]
        # chest_gy = data[:,27:30]
        # ankle_gy = data[:,44:47]
        # data = np.hstack([hand_acc, chest_acc, ankle_acc, hand_gy, chest_gy, ankle_gy])
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        # down sampling.
        data = data[::20,:]
        groundtruth = groundtruth[::20]
        groundtruth = groundtruth[:-2]
        n_state = len(set(groundtruth))
        ticc = TICC(window_size=3, number_of_clusters=n_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=3, threshold=threshold,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)
        prediction, _ = ticc.fit_transform(data)
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

# def exp_on_PAMAP(beta=2200, lambda_parameter=1e-3, threshold=1e-4, verbose=False):
#     score_list = []
#     # Indoor
#     for i in range(1,9):
#         dataset_path = os.path.join(data_path,'PAMAP/Indoor/subject'+str(i)+'.dat')
#         df = pd.read_csv(dataset_path, sep=' ', header=None)
#         data = df.to_numpy()
#         groundtruth = np.array(data[:,1],dtype=int)
#         # groundtruth = groundtruth[::50]
#         # hand = data[:,4:10]
#         # chest = data[:,18:24]
#         # shoe = data[:,32:38]
#         hand = data[:,4:7]
#         chest = data[:,18:21]
#         shoe = data[:,32:35]
#         data = np.hstack([hand, chest, shoe])
#         # data = data[::50,:]
#         n_state = len(set(groundtruth))
#         ticc = TICC(window_size=3, number_of_clusters=n_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=3, threshold=threshold,
#                 write_out_file=False, prefix_string="output_folder/", num_proc=1)
#         prediction, _ = ticc.fit_transform(data)
#         groundtruth = groundtruth[:-2]
#         f1, precision, recall = adjusted_macro_F_measure(groundtruth, prediction)
#         score_list.append(np.array([f1, precision, recall]))
#         if verbose:
#             print('ID: %s, F1: %f, Precision: %f, Recall: %f' %('Indoor'+str(i), f1, precision, recall))
#     # Outdoor
#     for i in range(2,9):
#         dataset_path = os.path.join(data_path,'PAMAP/Outdoor/subject'+str(i)+'.dat')
#         df = pd.read_csv(dataset_path, sep=' ', header=None)
#         data = df.to_numpy()
#         groundtruth = np.array(data[:,1],dtype=int)
#         hand = data[:,4:7]
#         chest = data[:,18:21]
#         shoe = data[:,32:35]
#         data = np.hstack([hand, chest, shoe])

#         n_state = len(set(groundtruth))
#         ticc = TICC(window_size=3, number_of_clusters=n_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=3, threshold=threshold,
#                 write_out_file=False, prefix_string="output_folder/", num_proc=1)
#         prediction, _ = ticc.fit_transform(data)

#         ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
#         f1, p, r = evaluate_cut_point(groundtruth, prediction)
#         score_list.append(np.array([f1, precision, recall]))
#         if verbose:
#             print('ID: %s, F1: %f, Precision: %f, Recall: %f' %('Outdoor'+str(i), f1, precision, recall))
#     score_list = np.vstack(score_list)
#     print('AVG ---- F1: %f, Precision: %f, Recall: %f' %(np.mean(score_list[:,0])\
#         ,np.mean(score_list[:,1])
#         ,np.mean(score_list[:,2])))

def exp_on_synthetic(beta=2200, lambda_parameter=1e-3, threshold=1e-4, verbose=False):
    base_path = os.path.join(data_path,'synthetic_data_for_segmentation2/')
    out_path = os.path.join(output_path,'synthetic2')
    create_path(out_path)
    score_list = []
    for i in range(100):
        fname = 'test'+str(i)+'.csv'
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, usecols=range(0,4))
        data = df.to_numpy()
        label = df = pd.read_csv(dataset_path, usecols=[4]).to_numpy().flatten()
        n_state= len(set(label))
        ticc = TICC(window_size=5, number_of_clusters=n_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=10, threshold=threshold,
                write_out_file=False, prefix_string="output_folder/", num_proc=10)
        prediction, _ = ticc.fit_transform(data)
        groundtruth = label[:-4]
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

def exp_on_USC_HAD(beta=2200, lambda_parameter=1e-3, threshold=1e-4, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'USC-HAD')
    create_path(out_path)
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            n_state= len(set(groundtruth))
            ticc = TICC(window_size=3, number_of_clusters=n_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=10, threshold=threshold,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)
            prediction, _ = ticc.fit_transform(data)
            groundtruth = groundtruth[:-2]
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path,'s%d_t%d'%(subject,target)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(str(subject)+'t'+str(target), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def run_exp():
    for beta in [500, 1000, 1500, 2000, 2500, 3000]:
        for lambda_parameter in [1e-3]:
            for threshold in [1e-3]:
                print('beta: %d, lambda_parameter: %f, threshold: %f' %(beta, lambda_parameter, threshold))
                time_start=time.time()
                # exp_on_PAMAP2(beta, lambda_parameter, threshold, verbose=True)
                # exp_on_synthetic(beta, lambda_parameter, threshold, verbose=False)
                exp_on_MoCap(beta, lambda_parameter, threshold, verbose=True)
                # exp_on_ActRecTut(beta, lambda_parameter, threshold, verbose=True)
                # exp_on_UCR_SEG(beta, lambda_parameter, threshold, verbose=False)
                # exp_on_USC_HAD(beta, lambda_parameter, threshold, verbose=False)
                time_end=time.time()
                print('time',time_end-time_start)

if __name__ == '__main__':
    # run_exp()
    ''' These are the best params found by us. '''
    ''' We also found that the effect of lambda and threshold is quite trivial. '''
    # exp_on_MoCap(2500, 1e-3, 1e-3, verbose=True)
    exp_on_synthetic(2500, 1e-3, 1e-3, verbose=True)
    # exp_on_UCR_SEG(500, 1e-3, 1e-3, verbose=True)
    # exp_on_ActRecTut(3000, 1e-3, 1e-3, verbose=True)
    # exp_on_USC_HAD(500, 1e-3, 1e-3, verbose=True)
    # exp_on_PAMAP2(3000, 1e-3, 1e-3, verbose=True)