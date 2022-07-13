import numpy as np
from TSpy.label import reorder_label, seg_to_label
from TSpy.eval import *
from TSpy.utils import len_of_file
from TSpy.view import *
from TSpy.dataset import *
import scipy.io
import os

data_path = os.path.join(os.path.dirname(__file__), '../../data/')
result_path = os.path.join(os.path.dirname(__file__), 'HVGHlearn/')

idx = 0

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

def  dilate_label(label, f, max_len):
    slice_list = []
    for e in label:
        # print(e)
        slice_list.append(e*np.ones(f, dtype=int))
    # print(len(slice_list))
    return np.concatenate(slice_list)[:max_len]

def evaluation_on_MoCap():
    idx = 100
    score_list = []
    for fname in dataset:
        label = np.loadtxt(result_path+'MoCap/'+fname+'/001/segm000.txt')[:,0].astype(int)
        groundtruth_json = dataset[fname]['label']
        groundtruth = seg_to_label(groundtruth_json)
        prediction = reorder_label(dilate_label(label, 50, len(groundtruth)))
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        print('HVGH,dataset%d,%f'%(idx,ari))
        idx+=1
        # print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def evaluation_on_PAMAP2():
    score_list = []
    for i in range(1,10):
        groundtruth = np.loadtxt(data_path+'/PAMAP2/Protocol/subject10'+str(i)+'.dat')[:,1].astype(int)    
        prediction = np.loadtxt(result_path+'/PAMAP2/subject10'+str(i)+'/001/segm000.txt')[:,0].astype(int)    
        prediction = dilate_label(prediction, 100, len(groundtruth))
        print(prediction.shape, groundtruth.shape)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
         # plot_mulvariate_time_series_and_label(data[0].T, label=prediction, groundtruth=groundtruth)
        # print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def evaluation_on_synthetic():
    idx = 0
    score_list = []
    for i in range(100):
        groundtruth = np.loadtxt(data_path+'/synthetic_data_for_segmentation/test'+str(i)+'.csv', delimiter=',')[:,4].astype(int)    
        prediction = np.loadtxt(result_path+'/synthetic/test'+str(i)+'/001/segm000.txt')[:,0].astype(int)    
        prediction = dilate_label(prediction, 100, len(groundtruth))
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        print('HVGH,dataset%d,%f'%(idx,ari))
        idx+=1
        # print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def evaluation_on_ActRecTut():
    idx = 211
    score_list = []
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        dataset_path = data_path+'ActRecTut/'+dir_name+'/data.mat'
        data = scipy.io.loadmat(dataset_path)
        groundtruth = data['labels'].flatten()[:30000]
        if not os.path.exists(result_path+'/ActRecTut/'+dir_name+'/001/segm000.txt'):
            print('HVGH,dataset%d,%f'%(idx,0))
            idx+=1
            continue
        prediction = np.loadtxt(result_path+'/ActRecTut/'+dir_name+'/001/segm000.txt')[:,0].astype(int)    
        prediction = dilate_label(prediction, 100, 30000)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        print('HVGH,dataset%d,%f'%(idx,ari))
        idx+=1
        # score_list.append(np.array([ari, anmi, nmi]))
        # print('ID: %s, ARI: %f, ANMI: %f, AMI: %f' %(dir_name, ari, anmi, nmi))
    # score_list = np.vstack(score_list)
    # print('AVG ---- F1: %f, Precision: %f, Recall: %f,  ARI: %f, AMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2])))

def evaluation_on_PAMAP2():
    idx = 213
    score_list = []
    for i in range(1,9):
        groundtruth = np.loadtxt(data_path+'/PAMAP2/Protocol/subject10'+str(i)+'.dat')[:,1].astype(int)    
        prediction = np.loadtxt(result_path+'/PAMAP2/subject10'+str(i)+'/001/segm000.txt')[:,0].astype(int)    
        prediction = dilate_label(prediction, 100, len(groundtruth))
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        print('HVGH,dataset%d,%f'%(idx,ari))
        idx+=1
        score_list.append(np.array([ari, anmi, nmi]))
         # plot_mulvariate_time_series_and_label(data[0].T, label=prediction, groundtruth=groundtruth)
        # print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))


def evaluation_on_USC_HAD():
    idx = 109
    score_list = []
    for subject in range(1,15):
        for target in range(1,6):
            _, groundtruth = load_USC_HAD(subject, target, data_path)
            if not os.path.exists(result_path+'/USC-HAD/subject'+str(subject)+'_target'+str(target)+'/001/segm000.txt'):
                print('HVGH,dataset%d,%f'%(idx,0))
                idx+=1
                continue
            prediction = np.loadtxt(result_path+'/USC-HAD/subject'+str(subject)+'_target'+str(target)+'/001/segm000.txt')[:,0].astype(int)    
            prediction = dilate_label(prediction, 100, len(groundtruth))
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            print('HVGH,dataset%d,%f'%(idx,ari))
            idx+=1
            score_list.append(np.array([ari, anmi, nmi]))
            # print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(subject, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def evaluation_on_UCR_SEG():
    idx = 179
    base = os.path.join(data_path, 'UCR-SEG/UCR_datasets_seg/')
    f_list = os.listdir(base)
    f_list.sort()
    score_list = []
    for fname,n in zip(f_list, range(1,len(f_list)+1)):
        info_list = fname[:-4].split('_')
        f = info_list[0]
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(base+fname)]=i
        groundtruth = seg_to_label(seg_info)[:]
        if not os.path.exists(result_path+'/UCR-SEG/'+fname+'/009/segm000.txt'):
            print('HVGH,dataset%d,%f'%(idx,0))
            idx+=1
            continue
        prediction = np.loadtxt(result_path+'/UCR-SEG/'+fname+'/009/segm000.txt')[:,0].astype(int)    
        prediction = dilate_label(prediction, 100, len(groundtruth))
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        print('HVGH,dataset%d,%f'%(idx,ari))
        idx+=1
        # ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        # f1, p, r = evaluate_cut_point(groundtruth, prediction, 200)
        score_list.append(np.array([ari, anmi, nmi]))
        # print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

evaluation_on_synthetic()
evaluation_on_MoCap()
evaluation_on_USC_HAD()
evaluation_on_UCR_SEG()
evaluation_on_ActRecTut()
evaluation_on_PAMAP2()