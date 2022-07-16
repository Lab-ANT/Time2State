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
    win_size=256
    base_path = os.path.join(data_path,'MoCap/4d/')
    params_LSE['in_channels'] = 4
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20
    params_LSE['compared_length'] = win_size

    f_list = os.listdir(base_path)
    f_list.sort()
    ari_list_of_all = []
    for idx, fname in enumerate(f_list):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]

        t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(15)).fit_encoder(data)
        ari_list = []
        for step in [24, 52, 76, 102, 128, 154, 180, 204, 230, 256]:
            # print('window size: %d, step size: %d' %(win_size, step))
            t2s.set_step(step)
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            ari, _, _ = evaluate_clustering(groundtruth, prediction)
            ari_list.append(ari)
        ari_list = np.array(ari_list)
        ari_list_of_all.append(ari_list)
    ari_list_of_all = np.array(ari_list_of_all)
    result = np.mean(ari_list_of_all, axis=0)
    print(result)
    return result

def exp_on_synthetic(verbose=False):
    win_size=128
    params_LSE['in_channels'] = 4
    params_LSE['compared_length'] = win_size
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20
    params_LSE['out_channels'] = 4
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation2/test')
    ari_list_of_all = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
        ari_list = []
        # for step in range(10,201,10):
        for step in [12, 24, 38, 52, 64, 76, 90, 102, 115, 128]:
            # print('i: %d, window size: %d, step size: %d' %(i, win_size, step))
            t2s.set_step(step)
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            ari, _, _ = evaluate_clustering(groundtruth, prediction)
            ari_list.append(ari)
        print(ari_list)
        ari_list = np.array(ari_list)
        ari_list_of_all.append(ari_list)
    ari_list_of_all = np.array(ari_list_of_all)
    result = np.mean(ari_list_of_all, axis=0)
    print(result)
    return result

def exp_on_UCR_SEG(verbose=False):
    win_size=256
    params_LSE['in_channels'] = 1
    params_LSE['out_channels'] = 2
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20
    params_LSE['compared_length'] = win_size

    ari_list_of_all = []
    dataset_path = os.path.join(data_path,'UCR-SEG/UCR_datasets_seg/')
    for fname in os.listdir(dataset_path):
        print(fname)
        info_list = fname[:-4].split('_')
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        df = pd.read_csv(dataset_path+fname)
        data = df.to_numpy()
        data = normalize(data)
        groundtruth = seg_to_label(seg_info)[:-1]

        t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(6)).fit_encoder(data)
        ari_list = []
        for step in [24, 52, 76, 102, 128, 154, 180, 204, 230, 256]:
            # print('window size: %d, step size: %d' %(win_size, step))
            t2s.set_step(step)
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            ari, _, _ = evaluate_clustering(groundtruth, prediction)
            ari_list.append(ari)
        ari_list = np.array(ari_list)
        ari_list_of_all.append(ari_list)
    ari_list_of_all = np.array(ari_list_of_all)
    result = np.mean(ari_list_of_all, axis=0)
    print(result)
    return result

def run_exp():
    # repeat 10 times:
    result = []
    for i in range(10):
        print(i)
        ari_list = exp_on_ActRecTut(verbose=True)
        # ari_list = exp_on_MoCap(verbose=True)
        # ari_list = exp_on_UCR_SEG(verbose=True)
        # ari_list = exp_on_synthetic(verbose=True)
        # ari_list = exp_on_USC_HAD(verbose=True)
        # ari_list = exp_on_PAMAP2(verbose=True)
        result.append(ari_list)
    result = np.array(result)
    result = np.mean(result, axis=0)
    print(result)
        
def exp_on_ActRecTut(verbose=False):
    win_size = 128
    # win_size = 256
    params_LSE['in_channels'] = 10
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20
    params_LSE['compared_length'] = win_size

    ari_list_of_all = []
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
        data = scipy.io.loadmat(dataset_path)
        groundtruth = data['labels'].flatten()
        groundtruth = reorder_label(groundtruth)
        data = data['data'][:,0:10]
        data = normalize(data)

        t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
        ari_list = []
        # for step in range(10,201,10):
        for step in [12, 24, 38, 52, 64, 76, 90, 102, 115, 128]:
        # for step in [24, 52, 76, 102, 128, 154, 180, 204, 230, 256]:
            # print('window size: %d, step size: %d' %(win_size, step))
            t2s.set_step(step)
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            ari, _, _ = evaluate_clustering(groundtruth, prediction)
            ari_list.append(ari)
        ari_list = np.array(ari_list)
        ari_list_of_all.append(ari_list)
    ari_list_of_all = np.array(ari_list_of_all)
    result = np.mean(ari_list_of_all, axis=0)
    print(result)
    return result

def exp_on_USC_HAD(verbose=False):
    # win_size = 256
    win_size = 512
    params_LSE['in_channels'] = 6
    params_LSE['compared_length'] = win_size
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 40
    params_LSE['kernel_size'] = 3
    params_LSE['compared_length'] = win_size
    
    ari_list_of_all = []
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
            ari_list = []
            # for step in [24, 52, 76, 102, 128, 154, 180, 204, 230, 256]:
            for step in [52, 102, 154, 204, 256, 308, 358, 408, 460, 512]:
                # print('window size: %d, step size: %d' %(win_size, step))
                t2s.set_step(step)
                t2s.predict(data, win_size, step)
                prediction = t2s.state_seq
                ari, _, _ = evaluate_clustering(groundtruth, prediction)
                ari_list.append(ari)
            print('s%dt%d'%(subject, target), ari_list)
            ari_list = np.array(ari_list)
            ari_list_of_all.append(ari_list)
    ari_list_of_all = np.array(ari_list_of_all)
    result = np.mean(ari_list_of_all, axis=0)
    print(result)
    return result

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

def exp_on_PAMAP2(verbose=False):
    win_size = 512
    params_LSE['in_channels'] = 9
    params_LSE['compared_length'] = win_size
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20
    dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(1)+'.dat')
    ari_list_of_all = []
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
        t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
        ari_list = []
        for step in [52, 102, 154, 204, 256, 308, 358, 408, 460, 512]:
            # print('window size: %d, step size: %d' %(win_size, step))
            t2s.set_step(step)
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            ari, _, _ = evaluate_clustering(groundtruth, prediction)
            ari_list.append(ari)
        ari_list = np.array(ari_list)
        ari_list_of_all.append(ari_list)
    ari_list_of_all = np.array(ari_list_of_all)
    result = np.mean(ari_list_of_all, axis=0)
    print(result)
    return result

if __name__ == '__main__':
    run_exp()