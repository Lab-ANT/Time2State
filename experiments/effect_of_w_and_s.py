import sys
import pandas as pd
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
    params_LSE['in_channels'] = 4
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20

    ari_matrix_list = []
    f_list = os.listdir(base_path)
    f_list.sort()
    for idx, fname in enumerate(f_list):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]

        ari_matrix = np.ones((10,10))
        for i, win_size in enumerate([100, 150, 200, 250, 300, 350, 400, 450, 500, 550]):
            params_LSE['compared_length'] = win_size
            t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
            for j, step in enumerate([10,20,30,40,50,60,70,80,90,100]):
                # print('window size: %d, step size: %d' %(win_size, step))
                t2s.set_step(step)
                t2s.predict(data, win_size, step)
                prediction = t2s.state_seq
                ari, _, _ = evaluate_clustering(groundtruth, prediction)
                ari_matrix[i,j] = ari
        ari_matrix_list.append(ari_matrix)
        print(ari_matrix.shape)
    ari_matrix = np.array(ari_matrix_list)
    result = np.mean(ari_matrix, axis=0)
    print(result)

def exp_on_UCR_SEG(verbose=False):
    params_LSE['in_channels'] = 4
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20

    ari_matrix_list = []
    dataset_path = os.path.join(data_path,'UCR-SEG/UCR_datasets_seg/')
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        # f = info_list[0]
        window_size = int(info_list[1])
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
        groundtruth = seg_to_label(seg_info)[:-1]

        ari_matrix = np.ones((10,10))
        for i, win_size in enumerate([100, 150, 200, 250, 300, 350, 400, 450, 500, 550]):
            params_LSE['compared_length'] = win_size
            t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
            for j, step in enumerate([10,20,30,40,50,60,70,80,90,100]):
                print('window size: %d, step size: %d' %(win_size, step))
                t2s.set_step(step)
                t2s.predict(data, win_size, step)
                prediction = t2s.state_seq
                ari, _, _ = evaluate_clustering(groundtruth, prediction)
                ari_matrix[i,j] = ari
        ari_matrix_list.append(ari_matrix)

def exp_on_ActRecTut(verbose=False):
    params_LSE['in_channels'] = 10
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20

    ari_matrix_list = []
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
        data = scipy.io.loadmat(dataset_path)
        groundtruth = data['labels'].flatten()
        groundtruth = reorder_label(groundtruth)
        data = data['data'][:,0:10]
        data = normalize(data)

        ari_matrix = np.ones((10,10))
        for i, win_size in enumerate([100, 150, 200, 250, 300, 350, 400, 450, 500, 550]):
            params_LSE['compared_length'] = win_size
            t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
            for j, step in enumerate([10,20,30,40,50,60,70,80,90,100]):
                print('window size: %d, step size: %d' %(win_size, step))
                t2s.set_step(step)
                t2s.predict(data, win_size, step)
                prediction = t2s.state_seq
                ari, _, _ = evaluate_clustering(groundtruth, prediction)
                ari_matrix[i,j] = ari
        ari_matrix_list.append(ari_matrix)
        print(ari_matrix.shape)
    ari_matrix = np.array(ari_matrix_list)
    result = np.mean(ari_matrix, axis=0)
    print(result)

def exp_on_USC_HAD(t2s, win_size, step, verbose=False):
    score_list = []
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            # the true num_state is 13
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))
    return np.mean(score_list[:,0])

def run_exp():
    for win_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        ARI_list = []
        params_LSE['in_channels'] = 6
        params_LSE['compared_length'] = win_size
        params_LSE['M'] = 20
        params_LSE['N'] = 5
        params_LSE['nb_steps'] = 40
        train, _ = load_USC_HAD(1, 1, data_path)
        train = normalize(train)
        t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(train)
        for step in [10,20,30,40,50,60,70,80,90,100]:
            print('window size: %d, step size: %d' %(win_size, step))
            t2s.set_step(step)
            ari = exp_on_USC_HAD(t2s, win_size, step, verbose=False)
            ARI_list.append(ari)
        print(ARI_list)

if __name__ == '__main__':
    # run_exp()
    exp_on_ActRecTut(verbose=True)
    # exp_on_MoCap(verbose=True)