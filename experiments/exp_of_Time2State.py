from statistics import mode
import pandas as pd
import sys
import os
import time
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

# def acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#     print(y_pred.size)
#     y_true = y_true.astype(np.int64)
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     ind = linear_assignment(w.max() - w)
#     # accuracy = (TP + TN) / (P + N)
#     converted_true = np.zeros(shape=y_pred.size, dtype=int)
#     for a, b in zip(ind[0], ind[1]):
#         idx = np.argwhere(y_true == a)
#         converted_true[idx] = b
#     print(converted_true)
#     print(y_pred)
#     return metrics.accuracy_score(converted_true, y_pred)

data_path = os.path.join(os.path.dirname(__file__), '../data/')
output_path = os.path.join(os.path.dirname(__file__), '../output/result_of_Time2Seg/')

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
    score_list = []
    params_LSE['in_channels'] = 4
    params_LSE['compared_length'] = win_size
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 4
    params_Triplet['in_channels'] = 4
    params_Triplet['compared_length'] = win_size
    params_TNC['in_channels'] = 4
    params_TNC['win_size'] = win_size
    params_CPC['in_channels'] = 4
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    # time2seg = Time2Seg(win_size, step, CausalConvEncoder(hyperparameters), DPGMM(None))
    for fname in os.listdir(base_path):
        dataset_path = base_path+fname
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        # print(data.shape)
        # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, LSTM_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), KMeansClustering(n_state)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
        # print(t2s.embeddings.shape)

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        # print(acc(groundtruth, prediction))
        # v_list = calculate_scalar_velocity_list(t2s.embeddings)
        # fig, ax = plt.subplots(nrows=2)
        # for i in range(4):
        #     ax[0].plot(data[:,i])
        # ax[1].plot(v_list)
        # plt.show()
        # plot_mulvariate_time_series_and_label_v2(data, label=prediction, groundtruth=groundtruth)
        # embedding_space(t2s.embeddings, show=True, s=5, label=t2s.embedding_label)
        score_list.append(np.array([ari, anmi, nmi]))
         # plot_mulvariate_time_series_and_label(data[0].T, label=prediction, groundtruth=groundtruth)
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_UCR_SEG(win_size, step, verbose=False):
    score_list = []
    params_Triplet['in_channels'] = 1
    params_Triplet['compared_length'] = win_size
    params_LSE['in_channels'] = 1
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 2
    params_LSE['nb_steps'] = 10
    params_LSE['compared_length'] = win_size
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
        df = pd.read_csv(dataset_path+fname)
        data = df.to_numpy()
        data = normalize(data)
        t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        groundtruth = seg_to_label(seg_info)[:-1]
        prediction = t2s.state_seq
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_synthetic(win_size=512, step=100, verbose=False):
    params_LSE['in_channels'] = 4
    params_LSE['compared_length'] = win_size
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 20
    params_LSE['out_channels'] = 4
    params_Triplet['in_channels'] = 4
    params_Triplet['compared_length'] = win_size
    params_TNC['win_size'] = win_size
    params_TNC['in_channels'] = 4
    params_CPC['in_channels'] = 4
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    prefix = os.path.join(data_path, 'synthetic_data_for_segmentation/test')
    score_list = []
    score_list2 = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        # segmentor = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        segmentor = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # segmentor = Time2State(win_size, step, LSTM_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # segmentor = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        prediction = segmentor.state_seq
        segmentor.set_clustering_component(KMeansClustering(5)).predict_without_encode(data, win_size, step)
        prediction2 = segmentor.state_seq
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        ari2, anmi2, nmi2 = evaluate_clustering(groundtruth, prediction2)
        score_list.append(np.array([ari, anmi, nmi]))
        score_list2.append(np.array([ari2, anmi2, nmi2]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari2, anmi2, nmi2))
    score_list = np.vstack(score_list)
    score_list2 = np.vstack(score_list2)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list2[:,0])\
        ,np.mean(score_list2[:,1])
        ,np.mean(score_list2[:,2])))

def exp_on_ActRecTut(win_size, step, verbose=False):
    params_Triplet['in_channels'] = 10
    params_Triplet['compared_length'] = win_size
    params_LSE['in_channels'] = 10
    params_LSE['compared_length'] = win_size
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 50
    params_TNC['in_channels'] = 10
    params_TNC['win_size'] = win_size
    params_CPC['in_channels'] = 10
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    score_list = []

    # train
    if True:
        dataset_path = os.path.join(data_path,'ActRecTut/subject1_gesture/data.mat')
        data = scipy.io.loadmat(dataset_path)
        # print(data)
        groundtruth = data['labels'].flatten()
        groundtruth = reorder_label(groundtruth)
        data = data['data'][:,0:10]
        data = normalize(data, mode='channel')
        # print(set(groundtruth))
        # true state number is 6
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), HDP_HSMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), GHMM(6)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
        # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
    # dir_list = ['subject1_walk', 'subject2_walk']
    dir_list = ['subject1_gesture', 'subject2_gesture']
    for dir_name in dir_list:
        dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
        data = scipy.io.loadmat(dataset_path)
        groundtruth = data['labels'].flatten()
        groundtruth = reorder_label(groundtruth)
        data = data['data'][:,0:10]
        data = normalize(data)
        print(data.shape)
        # print(set(groundtruth))
        # true state number is 6
        t2s.predict(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq+1
        f_cut, p_cut, r_cut = evaluate_cut_point(groundtruth, prediction, 100)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi, f_cut, p_cut, r_cut]))
        # plot_mulvariate_time_series_and_label_v2(data, label=prediction, groundtruth=groundtruth)
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f,  F1: %f, P: %f, R: %f' %(dir_name, ari, anmi, nmi, f_cut, p_cut, r_cut))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f,  F1: %f, P: %f, R: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])
        ,np.mean(score_list[:,3])
        ,np.mean(score_list[:,4])
        ,np.mean(score_list[:,5])))

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

def exp_on_PAMAP2(win_size, step, verbose=False):
    params_LSE['in_channels'] = 9
    params_LSE['compared_length'] = win_size
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 20
    params_LSE['N'] = 6
    params_LSE['nb_steps'] = 40
    params_Triplet['in_channels'] = 9
    params_Triplet['compared_length'] = win_size
    params_TNC['in_channels'] = 9
    params_TNC['win_size'] = win_size
    params_CPC['in_channels'] = 9
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 20
    dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(1)+'.dat')
    df = pd.read_csv(dataset_path, sep=' ', header=None)
    data = df.to_numpy()
    groundtruth = np.array(data[:,1],dtype=int)
    # print(set(groundtruth), len(set(groundtruth)))
    hand_acc = data[:,4:7]
    chest_acc = data[:,21:24]
    ankle_acc = data[:,38:41]
    # hand_gy = data[:,10:13]
    # chest_gy = data[:,27:30]
    # ankle_gy = data[:,44:47]
    # data = np.hstack([hand_acc, chest_acc, ankle_acc, hand_gy, chest_gy, ankle_gy])
    data = np.hstack([hand_acc, chest_acc, ankle_acc])
    data = fill_nan(data)
    data = normalize(data)
    # t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
    score_list = []
    for i in range(1, 9):
        dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(i)+'.dat')
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:,1],dtype=int)
        hand_acc = data[:,4:7]
        chest_acc = data[:,21:24]
        ankle_acc = data[:,38:41]
        # hand_gy = data[:,10:13]
        # chest_gy = data[:,27:30]
        # ankle_gy = data[:,44:47]
        # data = np.hstack([hand_acc, chest_acc, ankle_acc, hand_gy, chest_gy, ankle_gy])
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        data = normalize(data)
        # print(data.shape)
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), HDP_HSMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        t2s.predict(data, win_size, step)
        prediction = t2s.state_seq
        print(groundtruth.shape, prediction.shape)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        # plot_mulvariate_time_series_and_label(data[0].T, label=prediction, groundtruth=groundtruth)
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_USC_HAD(win_size, step, verbose=False):
    score_list = []
    score_list2 = []
    f_list = []
    params_LSE['in_channels'] = 6
    params_LSE['compared_length'] = win_size
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 40
    # params_LSE['kernel_size'] = 5
    # params_LSE['depth'] = 12
    # params_LSE['out_channels'] = 2
    params_Triplet['in_channels'] = 6
    params_Triplet['compared_length'] = win_size
    params_TNC['in_channels'] = 6
    params_TNC['win_size'] = win_size
    params_CPC['in_channels'] = 6
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 10
    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    # t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(train, win_size, step)
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(train, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), GMM_HMM(13)).fit(train, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit_encoder(train)
    # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), HDP_HSMM(None)).fit(train, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit_encoder(train)
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            data2 = data
            # the true num_state is 13
            t2s.predict(data, win_size, step)
            # print(data.shape)
            # t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
            prediction = t2s.state_seq
            # t2s.set_clustering_component(DPGMM(13)).predict_without_encode(data, win_size, step)
            t2s.set_clustering_component(DPGMM(13)).predict_without_encode(data, win_size, step)
            prediction2 = t2s.state_seq
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            ari2, anmi2, nmi2 = evaluate_clustering(groundtruth, prediction2)
            f1, p, r = evaluate_cut_point(groundtruth, prediction2, 500)
            score_list.append(np.array([ari, anmi, nmi]))
            score_list2.append(np.array([ari2, anmi2, nmi2]))
            f_list.append(np.array([f1, p, r]))
            # plot_mulvariate_time_series_and_label_v2(data2, groundtruth, prediction)
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari2, anmi2, nmi2))
                # print('ID: %s, F1: %f, Precision: %f, Recall: %f' %('s'+str(subject)+'t'+str(target), f1, p, r))
    score_list = np.vstack(score_list)
    score_list2 = np.vstack(score_list2)
    f_list = np.vstack(f_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list2[:,0])\
        ,np.mean(score_list2[:,1])
        ,np.mean(score_list2[:,2])))
    # print('AVG ---- F1: %f, Precision: %f, Recall: %f' %(np.mean(f_list[:,0])\
    #     ,np.mean(f_list[:,1])
    #     ,np.mean(f_list[:,2])))

def exp_on_EMD_gesture(win_size, step, verbose=False):
    params_LSE['in_channels'] = 8
    params_LSE['compared_length'] = win_size
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 40
    params_Triplet['in_channels'] = 6
    params_Triplet['compared_length'] = win_size
    params_TNC['in_channels'] = 8
    params_TNC['win_size'] = win_size
    params_CPC['in_channels'] = 8
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 10
    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    data = pd.read_csv(data_path+'EMG_gestures/01/1_raw_data_13-12_22.03.16.txt', index_col=False, header=None, sep="\t",skiprows=1, usecols=range(1,9))
    groundtruth = pd.read_csv(data_path+'EMG_gestures/01/1_raw_data_13-12_22.03.16.txt', index_col=False, header=None, sep="\t",skiprows=1, usecols=[9])
    data = data.to_numpy()
    groundtruth = groundtruth.to_numpy().flatten()
    train = normalize(data)
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
    prediction = t2s.state_seq
    print(groundtruth.shape, prediction.shape)
    print(evaluate_clustering(groundtruth, prediction))

def load_HAPT(dataset_path, id_exp):
    id_user = math.ceil(id_exp/2)
    full_data_path_acc = dataset_path+'/HAPT/RawData/acc_exp%02d_user%02d.txt'%(id_exp, id_user)
    full_data_path_gyro = dataset_path+'/HAPT/RawData/gyro_exp%02d_user%02d.txt'%(id_exp, id_user)
    full_label_path = dataset_path+'/HAPT/RawData/labels.txt'
    data_acc = np.loadtxt(full_data_path_acc)
    # data_gyro = np.loadtxt(full_data_path_gyro)
    # data = np.hstack([data_acc, data_gyro])
    # print(data.shape)

    label = np.loadtxt(full_label_path)
    id_list = label[:,0].flatten()
    label_list = label[:,2].flatten()
    pos_list = label[:,4].flatten()
    idx = np.argwhere(id_list==id_exp)
    y_true = label_list[idx].flatten()
    pos = pos_list[idx].flatten()
    label_json = {}
    for y,p in zip(y_true, pos):
        label_json[int(p)]=int(y)
    groundtruth = seg_to_label(label_json)
    length = len(groundtruth)
    return data_acc[:length], groundtruth

def exp_on_HAPT(win_size, step, verbose=False):
    params_LSE['in_channels'] = 3
    params_LSE['compared_length'] = win_size
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 40
    params_Triplet['in_channels'] = 3
    params_Triplet['compared_length'] = win_size
    params_TNC['in_channels'] = 3
    params_TNC['win_size'] = win_size
    params_CPC['in_channels'] = 3
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 10
    
    score_list = []
    data, groundtruth = load_HAPT(data_path, 1)
    data = normalize(data)
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
    # t2s = Time2State(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
    for i in range(1,21):
        data, groundtruth = load_HAPT(data_path, i)
        data = normalize(data)
        t2s.predict(data, win_size, step)
        prediction = t2s.state_seq
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(str(i), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def run_exp():
    for win_size in [128, 256, 512]:
        for step in [50, 100]:
            print('window size: %d, step size: %d' %(win_size, step))
            time_start=time.time()
            # exp_on_synthetic(win_size, step, verbose=True)
            # exp_on_UCR_SEG(win_size, step, verbose=True)
            # exp_on_MoCap(win_size, step, verbose=True)
            # exp_on_ActRecTut(win_size, step, verbose=True)
            # exp_on_PAMAP(win_size, step, verbose=True)
            # exp_on_PAMAP2(win_size, step, verbose=True)
            # exp_on_USC_HAD(win_size, step, verbose=True)
            # exp_on_PAMAP(beta, lambda_parameter, threshold, verbose=True)
            # exp_on_synthetic(beta, lambda_parameter, threshold, verbose=True)
            time_end=time.time()
            print('time',time_end-time_start)

if __name__ == '__main__':
    # run_exp()
    time_start=time.time()
    # exp_on_UCR_SEG(256, 50, verbose=True)
    # exp_on_MoCap(256, 50, verbose=True)
    # exp_on_PAMAP2(512,100, verbose=True)
    # exp_on_ActRecTut(64, 10, verbose=True)
    # exp_on_synthetic(256, 50, verbose=True)
    exp_on_USC_HAD(256, 50, verbose=True)
    # exp_on_EMD_gesture(128, 50, verbose=False)
    # exp_on_HAPT(256, 100, verbose=True)
    time_end=time.time()
    print('time',time_end-time_start)