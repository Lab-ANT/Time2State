import os
from TSpy.label import *
from TSpy.color import *
import scipy.io
import matplotlib.pyplot as plt

def load_USC_HAD(subject, target, usecols):
    prefix = os.path.join(os.path.dirname(__file__), '../data/USC-HAD/Subject'+str(subject)+'/')
    fname_prefix = 'a'
    fname_postfix = 't'+str(target)+'.mat'
    data_list = []
    label_json = {}
    total_length = 0
    for i in range(1,13):
        data = scipy.io.loadmat(prefix+fname_prefix+str(i)+fname_postfix)
        data = data['sensor_readings']
        data_list.append(data)
        total_length += len(data)
        label_json[total_length]=i
    label = seg_to_label(label_json)
    return np.vstack(data_list), label

def aggregate(X, win_size, mode='mean'):
    channel_list = []
    if mode == 'mean':
        for channel_num in range(X.shape[1]):
            i = 0
            channel = []
            while i < len(X):
                channel.append(np.mean(X[i:i+win_size,channel_num]))
                i += win_size
            channel = np.array(channel)
            channel_list.append(channel)
    return np.vstack(channel_list).T

# print(aggregate(np.ones(shape=(100,5)), 10).shape)

dataset_info = {'amc_86_01.4d':{'n_segs':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02.4d':{'n_segs':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03.4d':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5455:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07.4d':{'n_segs':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08.4d':{'n_segs':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09.4d':{'n_segs':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10.4d':{'n_segs':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11.4d':{'n_segs':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14.4d':{'n_segs':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

def plot_example():
    import os
    import scipy.io
    import pandas as pd
    from TSpy.utils import normalize
    from TSpy.eval import find_cut_points_from_label
    
    # data, groundtruth = load_USC_HAD(1, 3, None)
    # X = data[::10,3:6]
    # groundtruth = groundtruth[::10]
    # # X = normalize(X)
    
    # MoCap
    fname = 'amc_86_03.4d'
    data_path = os.path.join(os.path.dirname(__file__), '../data/MoCap/4d/'+fname)
    data = np.loadtxt(data_path)
    print(data)
    X = data
    groundtruth = seg_to_label(dataset_info[fname]['label'])

    # data_path = os.path.join(os.path.dirname(__file__), '../data/PAMAP2/Protocol/subject101.dat')
    # df = pd.read_csv(data_path, sep=' ', header=None)
    # data = df.to_numpy()
    # X = data[::4,27:30]
    # groundtruth = np.array(data[:,1],dtype=int)[::4]

    # data_path = os.path.join(os.path.dirname(__file__), '../data/ActRecTut/subject2_gesture/data.mat')
    # data = scipy.io.loadmat(data_path)
    # print(data)
    # X = data['data'][22270:23800,:3]
    # label_json = {85:0, 227:1, 344:0, 493:2, 589:0, 768:3, 863:0, 1160:4}
    # groundtruth = seg_to_label(label_json)
    # groundtruth = data['labels'].flatten()[22270:23430]
    # print(set(groundtruth), remove_duplication(groundtruth))

    plt.figure(figsize=(8,4))
    plt.style.use('classic')

    grid = plt.GridSpec(4,1)
    ax1 = plt.subplot(grid[0:3])
    plt.title('Time Series')
    plt.yticks([])

    # for i, l in zip(range(X.shape[1]),['left-arm', 'right-arm', 'left-leg', 'right-leg']):
    #     plt.plot(X[:,i], linewidth=1, alpha=0.9, label=l)
    # plt.legend(ncol=4, fontsize=12)

    # color_list = [air_force_blue, 'orange', 'green']
    color_list = ['#E25945', '#4593C1', '#9F96D6', '#828282']
    label_list = ['left-arm', 'right-arm', 'left-leg', 'right-leg']
    for i, l, c in zip(range(X.shape[1]), label_list, color_list):
        plt.plot(X[:,i], linewidth=1, alpha=0.9, label=l, color=c)
    plt.legend(ncol=4, fontsize=12)

    list_pos_cut = find_cut_points_from_label(groundtruth)
    for pos in list_pos_cut:
        plt.axvline(pos, 0, 300, color="black", lw=1, ls='-')

    list_pos_cut.append(0)
    list_pos_cut.append(len(groundtruth))
    list_pos_cut.sort()
    list_pos_text = []
    for i in range(len(list_pos_cut)-1):
        list_pos_text.append((list_pos_cut[i]+list_pos_cut[i+1])/2)

    plt.style.use('classic')
    plt.subplot(grid[3], sharex=ax1)
    plt.title('State Sequence')
    plt.yticks([])
    plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
          interpolation='nearest')

    text_pos_v = 'bottom'
    postrue_list = ['walking', 'running', 'jumping', 'walking', 'kicking', 'hop\n(left)', 'hop\n(right)', 'stretching', 'walking']
    for pos, text in zip(list_pos_text, postrue_list):
        # print(pos)
        plt.text(pos, 0, text, va='center', ha='center', size=10)

    plt.tight_layout()
    plt.show()

plot_example()