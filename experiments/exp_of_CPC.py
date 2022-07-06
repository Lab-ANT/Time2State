from re import sub
import pandas as pd
import sys
import os
from TSpy.eval import *
from TSpy.label import *
from TSpy.utils import *
from TSpy.view import *
from TSpy.dataset import *

sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')
output_path = os.path.join(script_path, '../results/output_CPC')

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

def exp_on_UCR_SEG(win_size, step, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'UCR-SEG')
    create_path(out_path)
    params_CPC['in_channels'] = 1
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 40
    params_CPC['out_channels'] = 2
    dataset_path = os.path.join(data_path,'UCR-SEG/UCR_datasets_seg/')
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        # f = info_list[0]
        # window_size = int(info_list[1])
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        num_state=len(seg_info)
        df = pd.read_csv(dataset_path+fname)
        data = df.to_numpy()
        data = normalize(data)
        t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        groundtruth = seg_to_label(seg_info)[:-1]
        prediction = t2s.state_seq
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

def exp_on_MoCap(win_size, step, verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    out_path = os.path.join(output_path,'MoCap')
    create_path(out_path)
    score_list = []
    params_CPC['in_channels'] = 4
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    for fname in os.listdir(base_path):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_synthetic(win_size=512, step=100, verbose=False):
    out_path = os.path.join(output_path,'synthetic2')
    create_path(out_path)
    params_CPC['in_channels'] = 4
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation2/test')
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
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

def exp_on_ActRecTut(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'ActRecTut')
    create_path(out_path)
    params_CPC['in_channels'] = 10
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    score_list = []

    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        for i in range(10):
            dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
            data = scipy.io.loadmat(dataset_path)
            groundtruth = data['labels'].flatten()
            groundtruth = reorder_label(groundtruth)
            data = data['data'][:,0:10]
            data = normalize(data)
            # print(data.shape)
            # true state number is 6
            t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
            # t2s.predict(data, win_size, step)
            prediction = t2s.state_seq+1
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path,dir_name+str(i)), result)
            f_cut, p_cut, r_cut = evaluate_cut_point(groundtruth, prediction, 100)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi, f_cut, p_cut, r_cut]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(dir_name, ari, anmi, nmi))
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

def exp_on_PAMAP2(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'PAMAP2')
    create_path(out_path)
    params_CPC['in_channels'] = 9
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(1)+'.dat')
    df = pd.read_csv(dataset_path, sep=' ', header=None)
    data = df.to_numpy()
    groundtruth = np.array(data[:,1],dtype=int)
    hand_acc = data[:,4:7]
    chest_acc = data[:,21:24]
    ankle_acc = data[:,38:41]
    data = np.hstack([hand_acc, chest_acc, ankle_acc])
    data = fill_nan(data)
    data = normalize(data)
    t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
    score_list = []
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
        data = normalize(data)
        t2s.predict(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,'10'+str(i)), result)
        # print(groundtruth.shape, prediction.shape)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_USC_HAD(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'USC-HAD')
    create_path(out_path)
    score_list = []
    params_CPC['in_channels'] = 6
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 10
    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(train, win_size, step)
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            data2 = data
            # the true num_state is 13
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path,'s%d_t%d'%(subject,target)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

if __name__ == '__main__':
    # print("Results of CPC on MoCap")
    # exp_on_MoCap(256, 50, verbose=True)
    # print("Results of CPC on PAMAP2")
    # exp_on_PAMAP2(512,100, verbose=True)
    # print("Results of CPC on ActRecTut")
    # exp_on_ActRecTut(256, 50, verbose=True)
    print("Results of CPC on synthetic")
    exp_on_synthetic(128, 50, verbose=True)
    # print("Results of CPC on USC-HAD")
    # exp_on_USC_HAD(256, 50, verbose=True)
    # print("Results of CPC on UCR-SEG")
    # exp_on_UCR_SEG(256, 50, verbose=True)