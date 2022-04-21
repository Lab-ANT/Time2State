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

data_path = os.path.join(os.path.dirname(__file__), '../data/')

def exp_on_synthetic(win_size=512, step=100, verbose=False):
    params_Triplet['in_channels'] = 4
    params_Triplet['compared_length'] = win_size
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation/test')
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        # segmentor = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        segmentor = Time2State(win_size, step, CausalConv_LSE_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        # segmentor = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        prediction = segmentor.state_seq
        # print(segmentor.embeddings.shape, prediction.shape)
        f1, precision, recall, ari, ami = evaluation(groundtruth[:len(prediction)], prediction)
        # f_cut, p_cut, r_cut = evaluate_cut_point(groundtruth, prediction, 100)
        score_list.append(np.array([f1, precision, recall, ari, ami]))
        # plot_mulvariate_time_series_and_label_v2(data, label=prediction, groundtruth=groundtruth)
        # embedding_space(segmentor.embeddings, show=True, s=5, label=segmentor.embedding_label)
        if verbose:
            print('ID: %d, F1: %f, Precision: %f, Recall: %f,  ARI: %f, AMI: %f' %(i, f1, precision, recall, ari, ami))
            # print('\t, Cut-point F1: %f, Precision: %f, Recall: %f' %(f_cut, p_cut, r_cut))
    score_list = np.vstack(score_list)
    print('AVG ---- F1: %f, Precision: %f, Recall: %f,  ARI: %f, AMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])
        ,np.mean(score_list[:,3])
        ,np.mean(score_list[:,4])))

if __name__ == '__main__':
    for step_size in range(110, 210, 10):
        exp_on_synthetic(256, step_size, verbose=False)