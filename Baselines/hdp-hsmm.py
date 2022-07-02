import pyhsmm
from pyhsmm.util.text import progprint_xrange
import numpy as np

np.seterr(divide='ignore') # these warnings are usually harmless for this code
import os
from TSpy.eval import *
from TSpy.view import *
from TSpy.label import *
from TSpy.dataset import *
from TSpy.utils import *
import pandas as pd
import time

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

data_path = os.path.join(os.path.dirname(__file__), '../data/')
script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../results/output_HDP_HSMM')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class HDP_HSMM:
    def __init__(self, alpha, beta, n_iter=20):
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def fit(self, X):
        data = X
        # Set the weak limit truncation level
        Nmax = 60
        # and some hyperparameters
        obs_dim = data.shape[1]
        obs_hypparams = {'mu_0':np.zeros(obs_dim),
                        'sigma_0':np.eye(obs_dim),
                        'kappa_0':0.25,
                        'nu_0':obs_dim+2}
        dur_hypparams = {'alpha_0':self.alpha,
                        'beta_0':self.beta}

        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
                alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
                init_state_concentration=6., # pretty inconsequential
                obs_distns=obs_distns,
                dur_distns=dur_distns)
        posteriormodel.add_data(data,trunc=600) # duration truncation speeds things up when it's possible

        for idx in progprint_xrange(self.n_iter):
            posteriormodel.resample_model()
        # posteriormodel.plot()
        # plt.show()
        return posteriormodel.stateseqs[0]

def exp_on_synthetic(alpha, beta, n_iter=20, verbose=False):
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation/test')
    out_path = os.path.join(output_path,'synthetic')
    create_path(out_path)
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        prediction = HDP_HSMM(alpha, beta, n_iter=n_iter).fit(data)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,str(i)), result)
        # print(groundtruth.shape, prediction.shape)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_MoCap(alpha, beta, n_iter=20, verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    score_list = []
    out_path = os.path.join(output_path,'MoCap')
    create_path(out_path)
    for fname in os.listdir(base_path):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        data = normalize(data)
        # n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        prediction = HDP_HSMM(alpha, beta, n_iter).fit(data)
        prediction = prediction.astype(int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,str(fname)), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_USC_HAD(alpha, beta, n_iter=20, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'USC-HAD')
    create_path(out_path)
    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)[::2]
            groundtruth = groundtruth[::2]
            prediction = HDP_HSMM(alpha, beta, n_iter).fit(data)
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

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

def exp_on_PAMAP2(alpha, beta, n_iter=20, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'PAMAP2')
    create_path(out_path)
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
        data = normalize(data)[::10]
        groundtruth = groundtruth[::10]
        prediction = HDP_HSMM(alpha, beta, n_iter).fit(data)
        # print(groundtruth.shape, prediction.shape)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,'10'+str(i)), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_ActRecTut(alpha, beta, n_iter=20, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'ActRecTut')
    create_path(out_path)
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
        data = scipy.io.loadmat(dataset_path)
        groundtruth = data['labels'].flatten()
        groundtruth = reorder_label(groundtruth)
        data = data['data'][:,0:10]
        prediction = HDP_HSMM(alpha, beta, n_iter).fit(data)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,dir_name), result)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(dir_name, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_UCR_SEG(alpha, beta, n_iter=20, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'UCR-SEG')
    create_path(out_path)
    dataset_path = os.path.join(data_path,'UCR-SEG/UCR_datasets_seg/')
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
        prediction = HDP_HSMM(alpha, beta, n_iter).fit(data)
        prediction = prediction.astype(int)
        groundtruth = seg_to_label(seg_info)[:-1]
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

def effect_of_length():
    data = np.loadtxt(data_path+'synthetic_data_for_segmentation/test0.csv', delimiter=',')
    data = np.concatenate([data[:,:4] for x in range(15)])

    time_list = []
    for length in range(1,21):
        time_start=time.time()
        prediction = HDP_HSMM(1e4, 20, 10).fit(data[:length*10000,:])
        time_end=time.time()
        print(length,time_end-time_start)
        time_list.append(time_end-time_start)
    time_list = np.array(time_list)
    print(time_list.round(2))
    return time_list.round(2)

def effect_of_dimension():
    time_list = []
    data = np.loadtxt(data_path+'../data/synthetic_data_for_segmentation/test0.csv', delimiter=',')[:,:4]
    data = np.hstack([data, data, data, data, data])
    data = np.vstack([data, data])
    data = data[:30000,:]
    for i in range(1,21):
        time_start=time.time()
        prediction = HDP_HSMM(1e4, 20, 10).fit(data[:,:i])
        time_end=time.time()
        print(i, time_end-time_start)
        time_list.append(time_end-time_start)
    print(np.array(time_list).round(2))
    return np.array(time_list).round(2)

# time_start=time.time()
# exp_on_synthetic(1e4, 20, n_iter=20, verbose=True)
# exp_on_MoCap(1e4, 20, n_iter=20, verbose=True)
# exp_on_USC_HAD(1e4, 20, n_iter=20, verbose=True)
# exp_on_UCR_SEG(1e4, 20, n_iter=20, verbose=True)
# exp_on_PAMAP2(1e4, 20, n_iter=20, verbose=True)
exp_on_ActRecTut(1e4, 20, n_iter=20, verbose=True)
# time_end=time.time()
# print('time',time_end-time_start)

# effect_of_length()
# effect_of_dimension()