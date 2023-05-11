import pandas as pd
import sys
import os
from TSpy.eval import *
from TSpy.label import *
from TSpy.utils import *
from TSpy.view import *
from TSpy.dataset import *
from miniutils import create_path, txt_filter

sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, 'data/FLOSS_format')
result_save_path = os.path.join(script_path, 'output_Time2State')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_on(dataset, win_size, step):
    dataset_result_save_path = os.path.join(result_save_path, '%s/state_seq'%(dataset))
    dataset_emb_velocity_save_path = os.path.join(result_save_path, '%s/emb_velocity'%(dataset))
    create_path(dataset_result_save_path)
    create_path(dataset_emb_velocity_save_path)
    file_list = txt_filter(os.listdir(os.path.join(data_path, dataset)))
    ari_list = []
    nmi_list = []
    for file_name in file_list:
        print(file_name)
        df = pd.read_csv(os.path.join(data_path, '%s/%s'%(dataset, file_name)), index_col=None, header=None)
        data = df.to_numpy()
        data = normalize(data)
        groundtruth = pd.read_csv(os.path.join(data_path, '%s/%s.label'%(dataset, file_name[:-4])), index_col=None, header=None).to_numpy().flatten()
        print(data.shape, groundtruth.shape)
        n_channels = data.shape[1]
        params_LSE['compared_length'] = win_size
        params_LSE['M'] = 10
        params_LSE['N'] = 4
        params_LSE['nb_steps'] = 40
        params_LSE['out_channels'] = 4
        params_LSE['in_channels'] = n_channels
        t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(len(set(groundtruth)))).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), KMeansClustering(len(set(groundtruth)))).fit(data, win_size, step)
        prediction = t2s.state_seq
        ari, anmi, nmi  = evaluate_clustering(groundtruth, prediction)
        print(ari, nmi)
        ari_list.append(ari)
        nmi_list.append(nmi)
        np.save(os.path.join(dataset_emb_velocity_save_path, '%s'%(file_name)), t2s.velocity)
        np.save(os.path.join(dataset_result_save_path, '%s'%(file_name)), prediction)
    print('Average ARI: %f, NMI: %f'%(np.mean(ari_list), np.mean(nmi_list)))

# run_on('MoCap', 512, 50)
# run_on('ActRecTut', 128, 50)
# run_on('synthetic_data', 256, 50)
# run_on('USC-HAD', 256, 50)
run_on('UCR-SEG', 512, 50)
# run_on('PAMAP2', 512, 50)