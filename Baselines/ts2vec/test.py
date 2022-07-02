from torch import embedding
from ts2vec import TS2Vec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('../')
from Time2State.clustering import *
from TSpy.utils import *

from TSpy.label import *
from TSpy.eval import *
from TSpy.dataset import *

class TS2VECEncoder():
    def __init__(self):
        self.model = TS2Vec(4,output_dims=2)
        self.__clustering_component = DPGMM(None)

    def train(self, X):
        data = X

        length, dim = data.shape
        train_seg_len = 1000

        train_seg_num = length/train_seg_len if length%train_seg_len==0 else length//train_seg_len+1

        pad_length = train_seg_len-(length%train_seg_len)
        data = np.pad(data,((0,pad_length),(0,0)),'constant')

        train_seg_list = []

        for i in range(train_seg_num):
            train_seg = data[i*train_seg_len:(i+1)*train_seg_len]
            train_seg_list.append(train_seg)

        data = np.array(train_seg_list)
        self.model.fit(data, n_epochs=10, n_iters=10)
        out = self.model.encode(data, encoding_window='multiscale')

        out = np.vstack(out)[:length]
        self.__embeddings = out[:,-4:]

    def clustering(self):
        self.__embedding_label = self.__clustering_component.fit(self.__embeddings)

    def fit(self,X):
        self.train(X)
        self.clustering()
        return self.__embedding_label

    def draw(self, X):
        data = X

        length, dim = data.shape
        train_seg_len = 1000

        train_seg_num = length/train_seg_len if length%train_seg_len==0 else length//train_seg_len+1

        pad_length = train_seg_len-(length%train_seg_len)
        data = np.pad(data,((0,pad_length),(0,0)),'constant')
        # print(data.shape)

        train_seg_list = []

        for i in range(train_seg_num):
            train_seg = data[i*train_seg_len:(i+1)*train_seg_len]
            train_seg_list.append(train_seg)

        data = np.array(train_seg_list)
        print(data.shape)
        self.model.fit(data, n_epochs=100, n_iters=100)
        out = self.model.encode(data, encoding_window='multiscale')

        out = np.vstack(out)
        print(out.shape)
        return out

# t2v = TS2VECEncoder()

# # df = pd.read_csv('../data/synthetic_data_for_segmentation/test0.csv', usecols=range(4), skiprows=1)
# # df_gt = pd.read_csv('../data/synthetic_data_for_segmentation/test0.csv', usecols=[4], skiprows=1)
# # groundtruth = df_gt.to_numpy().flatten()
# # data = df.to_numpy()

data_path = os.path.join(os.path.dirname(__file__), '../data/')
# data, groundtruth = load_USC_HAD(1, 1, data_path)

# prediction = t2v.fit(data)
# print(prediction.shape)

# result = evaluate_clustering(groundtruth, prediction)
# print(result)



# # out = t2v.draw(data)
# # groundtruth = groundtruth[:out.shape[0]]
# # state_set = set(groundtruth)

# # for state in state_set:
# #     idx = np.argwhere(groundtruth==state)
# #     plt.scatter(out[idx,18],out[idx,19])
# # plt.savefig('fig.png')

# # plt.imshow(out[::100].T)
# # plt.savefig('fig.png')

def exp_on_USC_HAD(verbose=False):
    score_list = []
    f_list = []
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            # the true num_state is 13
            t2v = TS2VECEncoder()
            prediction = t2v.fit(data)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

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

def exp_on_MoCap(verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    score_list = []
    for fname in os.listdir(base_path):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        
        t2v = TS2VECEncoder()
        prediction = t2v.fit(data)

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_synthetic(verbose=False):
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation/test')
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        
        t2v = TS2VECEncoder()
        prediction = t2v.fit(data)

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

# exp_on_USC_HAD(verbose=True)
# exp_on_MoCap(verbose=True)
exp_on_synthetic(verbose=True)