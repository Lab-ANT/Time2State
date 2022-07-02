import numpy as np
import os
import re
import pandas as pd
from TSpy.label import seg_to_label
from TSpy.eval import *
from TSpy.utils import len_of_file
from TSpy.view import *
from TSpy.dataset import *

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../../../data/')
output_path = os.path.join(script_path, '../output/')
redirect_path = os.path.join(script_path, '../../../results/output_AutoPlait')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def redirect_USC_HAD():
    data_redirect_path = os.path.join(redirect_path, 'USC-HAD')
    create_path(data_redirect_path)
    for subject in range(1,15):
        for target in range(1,6):
            _, groundtruth = load_USC_HAD(subject, target, data_path)
            data_idx = (subject-1)*5+target
            prediction = read_result_dir2(os.path.join(output_path, '_out_USC-HAD/dat'+str(data_idx)+'/'),len(groundtruth))
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(data_idx, ari, anmi, nmi))
            print(os.path.join(data_redirect_path,'s%d_t%d'%(subject,target)))
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(data_redirect_path,'s%d_t%d'%(subject,target)), result)

def redirect_MoCap():
    data_redirect_path = os.path.join(redirect_path, 'MoCap')
    create_path(data_redirect_path)
    for n,i in zip(['01','02','03','07','08','09','10','11','14'], range(1,10)):
        prediction = read_result_dir(os.path.join(output_path, '_out_MoCap/dat'+str(i)+'/'), os.path.join(data_path, 'MoCap/4d/amc_86_'+n+'.4d'))
        groundtruth = seg_to_label(dataset_info['amc_86_'+str(n)+'.4d']['label'])
        fname = 'amc_86_'+str(n)+'.4d'
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(data_redirect_path, fname), result)

def redirect_ActRecTut():
    data_redirect_path = os.path.join(redirect_path, 'ActRecTut')
    create_path(data_redirect_path)
    for i, s in enumerate(['subject1_walk', 'subject2_walk']):
        full_path = data_path+'/ActRecTut/data_for_AutoPlait/'+s+'_groundtruth.txt'
        groundtruth = np.loadtxt(full_path)
        prediction = read_result_dir2(output_path+'/_out_ActRecTut/dat'+str(i+1)+'/', len(groundtruth))
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(data_redirect_path, s), result)

def redirect_UCR_SEG():
    data_redirect_path = os.path.join(redirect_path, 'UCR-SEG')
    create_path(data_redirect_path)
    base = os.path.join(data_path, 'UCR-SEG/UCR_datasets_seg/')
    f_list = os.listdir(base)
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
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(data_redirect_path, fname), result)

def redirect_PAMAP2():
    data_redirect_path = os.path.join(redirect_path, 'PAMAP2')
    create_path(data_redirect_path)
    for i in range(1,10):
        groundtruth = np.loadtxt(data_path+'/PAMAP2/data_for_AutoPlait/groundtruth_subject10'+str(i)+'.txt')#[::10]
        prediction = read_result_dir(output_path+'/_out_PAMAP2/dat'+str(i)+'/', data_path+'/PAMAP2/data_for_AutoPlait/subject10'+str(i)+'.txt')#[::10]
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(data_redirect_path, '10'+str(i)), result)

def redirect_synthetic():
    data_redirect_path = os.path.join(redirect_path, 'synthetic')
    create_path(data_redirect_path)
    base = os.path.join(data_path, 'synthetic_data_for_Autoplait/')
    f_list = os.listdir(base)
    f_list.remove('list')
    f_list = np.sort(f_list)
    for fname in f_list:
        n = int(fname[4:-4])
        data = pd.read_csv(base+'test'+str(n)+'.csv', sep=' ', usecols=range(4)).to_numpy()
        prediction = read_result_dir(os.path.join(output_path, '_out_synthetic/dat'+str(n+1)+'/'),base+fname)[:-1]
        groundtruth = pd.read_csv(base+'test'+str(n)+'.csv', sep=' ', usecols=[4]).to_numpy().flatten()
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(data_redirect_path, fname[4:-4]), result)

redirect_ActRecTut()
redirect_synthetic()
redirect_UCR_SEG()
redirect_PAMAP2()
redirect_MoCap()
redirect_USC_HAD()