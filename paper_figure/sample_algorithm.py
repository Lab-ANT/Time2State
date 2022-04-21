import matplotlib.pyplot as plt
from TSpy.utils import *

def plot_example():
    import os
    import scipy.io
    import pandas as pd
    from TSpy.utils import normalize
    from TSpy.eval import find_cut_points_from_label
    
    # data, groundtruth = load_USC_HAD(1, 3, None)
    # X = data[::10,:3]
    # groundtruth = groundtruth[::10]
    # X = normalize(X)

    # data_path = os.path.join(os.path.dirname(__file__), '../data/gesture_phase_dataset/a1_va3.csv')
    # data = np.loadtxt(data_path, skiprows=1, delimiter=',', usecols=[1,2,3])
    # X=data
    # groundtruth= np.ones(len(X))
    
    data_path = os.path.join(os.path.dirname(__file__), '../data/ActRecTut/subject2_gesture/data.mat')
    data = scipy.io.loadmat(data_path)
    X = data['data'][22270:23800,:6]
    # X = normalize(X)
    groundtruth = data['labels'].flatten()[22270:23800]

    # data_path = os.path.join(os.path.dirname(__file__), '../data/PAMAP2/Protocol/subject101.dat')
    # df = pd.read_csv(data_path, sep=' ', header=None)
    # data = df.to_numpy()
    # X = data[:,21:24]
    # groundtruth = np.array(data[:,1],dtype=int)

    # plt.figure(figsize=(8,1))
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(8,1.2))

    plt.xticks(color='w')
    plt.yticks(color='w')
    ax.axes.tick_params(size=0)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    normalize(X)

    # plt.plot(X[:,2], linewidth=1, alpha=0.9, color='#9F96D6')
    # plt.plot(X[:,3], linewidth=1, alpha=0.9, color='gray')
    plt.plot(X[:,4], linewidth=1, alpha=0.9)
    # plt.grid()

    # for i in range(X.shape[1]):
    #     plt.plot(X[:,i], linewidth=1, alpha=0.9)

    # list_pos_cut = find_cut_points_from_label(groundtruth)
    # for pos in list_pos_cut:
    #     plt.axvline(pos, 0, 300, color="black", lw=1, ls='-')

    # list_pos_cut.append(0)
    # list_pos_cut.append(len(X))
    # list_pos_cut.sort()
    # list_pos_text = []
    # for i in range(len(list_pos_cut)-1):
    #     list_pos_text.append((list_pos_cut[i]+list_pos_cut[i+1])/2)

    # plt.style.use('classic')
    # plt.subplot(grid[3], sharex=ax1)
    # plt.title('State Sequence')
    # plt.yticks([])
    # plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
    #       interpolation='nearest')

    # text_pos_v = 'bottom'
    # for pos in list_pos_text:
    #     plt.text(pos, 0, 'trans', va='bottom', ha='center')

    plt.tight_layout()
    plt.show()

plot_example()