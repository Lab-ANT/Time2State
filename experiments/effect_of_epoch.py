import sys
sys.path.append('./')
import pandas as pd
import os
from TSpy.eval import *
from TSpy.label import *
from TSpy.utils import *
from TSpy.view import *
from TSpy.dataset import *
from Time2Seg.models.Time2Seg import *

data_path = os.path.join(os.path.dirname(__file__), '../data/')

def exp_on_synthetic(win_size=512, step=100, nb_steps=10, verbose=False):
    hyperparameters['in_channels'] = 4
    hyperparameters['nb_steps'] = nb_steps
    loss_list_list = []
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation/test')
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        segmentor = Time2Seg(win_size, step, CausalConvEncoder_A(hyperparameters), DPGMM(None)).fit(data, win_size, step)
        loss_list = np.array(segmentor.loss_list)
        loss_list_list.append(loss_list)
        print(loss_list.flatten())
        prediction = segmentor.state_seq
        f1, precision, recall, ari, ami = evaluation(groundtruth[:len(prediction)], prediction)
        # f_cut, p_cut, r_cut = evaluate_cut_point(groundtruth, prediction, 100)
        score_list.append(np.array([f1, precision, recall, ari, ami]))
        if verbose:
            print('ID: %d, F1: %f, Precision: %f, Recall: %f,  ARI: %f, AMI: %f' %(i, f1, precision, recall, ari, ami))
            # print('\t, Cut-point F1: %f, Precision: %f, Recall: %f' %(f_cut, p_cut, r_cut))
    score_list = np.vstack(score_list)
    print('AVG ---- F1: %f, Precision: %f, Recall: %f,  ARI: %f, AMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])
        ,np.mean(score_list[:,3])
        ,np.mean(score_list[:,4])))
    return loss_list_list

loss_list_list = exp_on_synthetic(256, 50, 20, verbose=True)

import matplotlib.pyplot as plt

mean_list = []
loss_list_list = np.array(loss_list_list)
for i in range(20):
    mean_list.append(np.mean(loss_list_list[:,i]))

plt.plot(mean_list)
plt.show()
