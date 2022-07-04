import numpy as np
import os
import re
import pandas as pd
import scipy.io
from TSpy.label import seg_to_label
from TSpy.eval import *
from TSpy.utils import len_of_file
from TSpy.view import *
from TSpy.dataset import *

data_path = os.path.join(os.path.dirname(__file__), '../../../data/')
output_path = os.path.join(os.path.dirname(__file__), '../output/')

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

def read_result_dir2(path,length):
    results = os.listdir(path)
    results = [s if re.match('segment.\d', s) != None else None for s in results]
    results = np.array(results)
    idx = np.argwhere(results!=None)
    results = results[idx].flatten()

    label = np.zeros(length,dtype=int)

    l = 0
    for r in results:
        data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
        start = data.col0
        end = data.col1
        for s, e in zip(start,end):
            label[s:e+1]=l
        l+=1
    return label
   
def read_result_dir(path,original_file_path):
    results = os.listdir(path)
    results = [s if re.match('segment.\d', s) != None else None for s in results]
    results = np.array(results)
    idx = np.argwhere(results!=None)
    results = results[idx].flatten()

    length = len_of_file(original_file_path)
    # print(length)
    label = np.zeros(length,dtype=int)

    l = 0
    for r in results:
        data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
        start = data.col0
        end = data.col1
        for s, e in zip(start,end):
            label[s:e+1]=l
        l+=1
    return label

def evaluation_on_USC_HAD():
    prefix = os.path.join('../data/USC-HAD/Subject3/')
    fname_prefix = 'a'
    fname_postfix = 't5.mat'

    series_list1 = []
    series_list2 = []
    series_list3 = []

    label_seg = {}
    total_length = 0
    for i in range(1,13):
        data = scipy.io.loadmat(prefix+fname_prefix+str(i)+fname_postfix)
        series = data['sensor_readings'][:,5]
        series_list1.append(series)
        series = data['sensor_readings'][:,4]
        series_list2.append(series)
        series = data['sensor_readings'][:,3]
        series_list3.append(series)
        total_length += len(data['sensor_readings'][:,3])
        label_seg[total_length]=i

    groundtruth = seg_to_label(label_seg)
    prediction = read_result_dir2('output/_out_USC/dat1/',len(groundtruth))+1
    print(set(groundtruth), set(prediction))
    ari = ARI(groundtruth,prediction)
    ami = AMI(groundtruth,prediction)
    accuracy = macro_precision(groundtruth, prediction)
    macrof1 = macro_f1score(groundtruth,prediction)
    print(' ARI: %f, AMI: %f, F1: %f, ACC: %f' %(ari, ami, macrof1, accuracy))

def evaluation_on_MoCap():
    score_list = []
    for n,i in zip(['01','02','03','07','08','09','10','11','14'], range(1,10)):
        prediction = read_result_dir(os.path.join(output_path, '_out_MoCap/dat'+str(i)+'/'), os.path.join(data_path, 'MoCap/4d/amc_86_'+n+'.4d'))
        groundtruth = seg_to_label(dataset['amc_86_'+str(n)+'.4d']['label'])
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        f1, p, r = evaluate_cut_point(groundtruth, prediction, 100)
        score_list.append(np.array([ari, anmi, nmi, f1, p, r]))
        print('ID: %s, ARI: %f, ANMI: %f, NMI: %f, F1: %f, P: %f, R: %f' %(i, ari, anmi, nmi, f1, p, r))
    score_list = np.vstack(score_list)
    print('AVG ---- , ARI: %f, ANMI: %f, NMI: %f, 1F1: %f, P: %f, R: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])
        ,np.mean(score_list[:,3])
        ,np.mean(score_list[:,4])
        ,np.mean(score_list[:,5])))

# def evaluation_on_PAMAP():
#     prediction = read_result_dir('output/_out_PAMAP/dat1/','../data/PAMAP_Dataset/Indoor/subject1.dat')
#     dataset_path = '../data/PAMAP_Dataset/Indoor/subject1.dat'
#     df = pd.read_csv(dataset_path, sep=' ', header=None)
#     data = df.to_numpy()
#     groundtruth = np.array(data[:,1],dtype=int)
#     ari = ARI(groundtruth,prediction)
#     ami = AMI(groundtruth,prediction)
#     accuracy = macro_precision(groundtruth, prediction)
#     macrof1 = macro_f1score(groundtruth,prediction)
#     print(' ARI: %f, AMI: %f, F1: %f, ACC: %f' %(ari, ami, macrof1, accuracy))

def evaluation_on_ActRecTut():
    score_list = []
    for i, s in enumerate(['subject1_walk', 'subject2_walk']):
        full_path = data_path+'/ActRecTut/data_for_AutoPlait/'+s+'_groundtruth.txt'
        groundtruth = np.loadtxt(full_path)
        prediction = read_result_dir2(output_path+'/_out_ActRecTut/dat'+str(i+1)+'/', len(groundtruth))
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(s, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def evaluation_on_UCR_SEG():
    base = os.path.join(data_path, 'UCR-SEG/UCR_datasets_seg/')
    f_list = os.listdir(base)
    score_list = []
    for fname,n in zip(f_list, range(1,len(f_list)+1)):
        prediction = read_result_dir(os.path.join(output_path, '_out_UCR_SEG/dat'+str(n)+'/'), base+fname)
        info_list = fname[:-4].split('_')
        f = info_list[0]
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(base+fname)]=i
        groundtruth = seg_to_label(seg_info)[:]
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        # f1, p, r = evaluate_cut_point(groundtruth, prediction, 200)
        score_list.append(np.array([ari, anmi, nmi]))
        print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def evaluation_on_PAMAP2():
    score_list = []
    for i in range(1,10):
        groundtruth = np.loadtxt(data_path+'/PAMAP2/data_for_AutoPlait/groundtruth_subject10'+str(i)+'.txt')#[::10]
        prediction = read_result_dir(output_path+'/_out_PAMAP2/dat'+str(i)+'/', data_path+'/PAMAP2/data_for_AutoPlait/subject10'+str(i)+'.txt')#[::10]
        print(groundtruth.shape, prediction.shape)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        # metrics.adjusted_rand_score()
        print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def evaluation_on_synthetic():
    # base = '../data/synthetic_data_for_AutoPlait/'
    base = os.path.join(data_path, 'synthetic_data_for_Autoplait/')
    f_list = os.listdir(base)
    f_list.remove('list')
    # f_list.remove('convert.py')
    f_list = np.sort(f_list)
    score_list = []
    # for fname,n in zip(f_list, range(1,len(f_list)+1)):
    for fname in f_list:
        # print(fname)
        n = int(fname[4:-4])
        data = pd.read_csv(base+'test'+str(n)+'.csv', sep=' ', usecols=range(4)).to_numpy()
        prediction = read_result_dir(os.path.join(output_path, '_out_synthetic/dat'+str(n+1)+'/'),base+fname)[:-1]
        groundtruth = pd.read_csv(base+'test'+str(n)+'.csv', sep=' ', usecols=[4]).to_numpy().flatten()
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        f1, p, r = evaluate_cut_point(groundtruth, prediction, 200)
        score_list.append(np.array([ari, anmi, nmi, f1, p, r]))
        print('ID: %s, ARI: %f, ANMI: %f, NMI: %f, F1: %f, P: %f, R: %f' %(fname, ari, anmi, nmi, f1, p, r))
        plot_mulvariate_time_series_and_label_v2(data , groundtruth=groundtruth, label=prediction)
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f, 1F1: %f, P: %f, R: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])
        ,np.mean(score_list[:,3])
        ,np.mean(score_list[:,4])
        ,np.mean(score_list[:,5])))

def evaluation_on_USC_HAD():
    base = os.path.join(data_path, 'USC-HAD_for_AutoPlait/')
    f_list = os.listdir(base)
    f_list.remove('list')
    f_list = np.sort(f_list)
    score_list = []
    i = 1
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            prediction = read_result_dir2(os.path.join(output_path, '_out_USC-HAD/dat'+str(i)+'/'),len(groundtruth))[:-1]
            i+=1
            groundtruth = groundtruth[:-1]
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))

    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

# evaluation_on_ActRecTut()
# evaluation_on_PAMAP()
# evaluation_on_UCR_SEG()
evaluation_on_synthetic()
# evaluation_on_MoCap()
# evaluation_on_PAMAP2()
# evaluation_on_ActRecTut()
# evaluation_on_USC_HAD()