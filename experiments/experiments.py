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
output_path = os.path.join(script_path, '../results/output_Time2State')

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

def exp_on_MoCap(win_size, step, verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    out_path = os.path.join(output_path,'MoCap')
    create_path(out_path)
    score_list = []
    params_LSE['in_channels'] = 4
    params_LSE['win_size'] = win_size
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 4
    params_LSE['nb_steps'] = 40
    params_LSE['win_type'] = 'hanning'
    f_list = os.listdir(base_path)
    f_list.sort()
    for idx, fname in enumerate(f_list):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, Transformer_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

if __name__ == '__main__':
    exp_on_MoCap(512, 50, verbose=True)