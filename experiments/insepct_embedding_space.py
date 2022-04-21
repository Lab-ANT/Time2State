import sys

from torch import embedding
sys.path.append('./')
from Time2State.adapers import *
from Time2State.default_params import *
from TSpy.eval import *
import numpy as np
import matplotlib.pyplot as plt
from TSpy.view import *

result_path = os.path.join(os.path.dirname(__file__), '../output/embeddings/')

if not os.path.exists(result_path):
    os.makedirs(result_path)

win_size = 256
step = 10
params_LSE['in_channels'] = 4
params_LSE['out_channels'] = 2
params_LSE['nb_steps'] = 20
params_Triplet['in_channels'] = 4
params_Triplet['out_channels'] = 2
params_Triplet['nb_steps'] = 20
params_TNC['win_size']=256
params_TNC['in_channels'] = 4
params_TNC['out_channels'] = 2
params_TNC['nb_steps'] = 5
params_CPC['win_size']=256
params_CPC['in_channels'] = 4
params_CPC['out_channels'] = 2
params_CPC['nb_steps'] = 5

# convert data to a list of window.
def get_window_label(X, win_size, step):
    # length = X.shape[2]
    length = len(X)
    print(length)
    # exactly [(length-win_size)/step+1] windows.
    if (length-win_size)%step == 0:
        num_of_windows = int((length-win_size)/step+1)
        # list_window = [X[:,:,i:i+win_size] for i in range(0, num_of_windows*step, step)]
        window_label = [X[int((i+i+win_size)/2)] for i in range(0, num_of_windows*step, step)]
    else:
        num_of_windows = int((length-win_size)/step+2)
        # list_window = [X[:,:,i:i+win_size] for i in range(0, num_of_windows*step, step)]
        window_label = [X[int((i+i+win_size)/2)] for i in range(0, num_of_windows*step, step)]
    return window_label


data_path = os.path.join(os.path.dirname(__file__), '../data/')
file_path = data_path+'synthetic_data_for_segmentation/test3.csv'
groundtruth = np.loadtxt(file_path, delimiter=',', usecols=[4])
groundtruth2 = np.loadtxt(file_path, delimiter=',', usecols=[4])
list_pos_cut = find_cut_points_from_label(groundtruth)

# plot raw data
data = np.loadtxt(file_path, delimiter=',', usecols=range(4))

for pos in list_pos_cut:
    groundtruth[pos-128:pos+128] = -1
groundtruth = groundtruth[128::10]

if not os.path.exists(os.path.join(result_path+'/embeddings_Triplet.txt')):
    adaper_Triplet = CausalConv_Triplet_Adaper(params_Triplet)
    data = np.loadtxt(file_path, delimiter=',', usecols=range(4))

    adaper_Triplet.fit(data)
    embeddings_Triplet = adaper_Triplet.encode(data, 256, 10)
    np.savetxt(os.path.join(result_path+'embeddings_Triplet.txt'), embeddings_Triplet)

if not os.path.exists(os.path.join(result_path+'/embeddings_TNC.txt')):
    adaper_TNC = CausalConv_TNC_Adaper(params_TNC)
    data = np.loadtxt(file_path, delimiter=',', usecols=range(4))

    adaper_TNC.fit(data)
    embeddings_TNC = adaper_TNC.encode(data, 256, 10)
    np.savetxt(os.path.join(result_path+'embeddings_TNC.txt'), embeddings_TNC)

if not os.path.exists(os.path.join(result_path+'/embeddings_CPC.txt')):
    adaper_CPC = CausalConv_CPC_Adaper(params_CPC)
    data = np.loadtxt(file_path, delimiter=',', usecols=range(4))

    adaper_CPC.fit(data)
    embeddings_CPC = adaper_CPC.encode(data, 256, 10)
    np.savetxt(os.path.join(result_path+'embeddings_CPC.txt'), embeddings_CPC)

if not os.path.exists(os.path.join(result_path+'/embeddings_LSE.txt')):
    adaper_LSE = CausalConv_LSE_Adaper(params_LSE)
    data = np.loadtxt(file_path, delimiter=',', usecols=range(4))

    adaper_LSE.fit(data)
    embeddings_LSE = adaper_LSE.encode(data, 256, 10)
    np.savetxt(os.path.join(result_path+'embeddings_LSE.txt'), embeddings_LSE)


embeddings_LSE = np.loadtxt(os.path.join(result_path+'/embeddings_LSE.txt'))
embeddings_Triplet = np.loadtxt(os.path.join(result_path+'/embeddings_Triplet.txt'))
embeddings_TNC = np.loadtxt(os.path.join(result_path+'/embeddings_TNC.txt'))
embeddings_CPC = np.loadtxt(os.path.join(result_path+'/embeddings_CPC.txt'))


# plt.style.use('classic')
# embedding_space(embeddings_Triplet, show=False, s=5, label=groundtruth[:len(embeddings_LSE)])
# plt.title('Triplet')
# plt.savefig('Triplet.pdf')
# embedding_space(embeddings_LSE, show=False, s=5, label=groundtruth[:len(embeddings_LSE)])
# plt.title('LSE')
# plt.savefig('LSE.pdf')
# # plt.savefig('LSE.png')
# embedding_space(embeddings_TNC, show=False, s=5, label=groundtruth[:len(embeddings_LSE)])
# plt.title('TNC')
# plt.savefig('TNC.pdf')
# embedding_space(embeddings_CPC, show=False, s=5, label=groundtruth[:len(embeddings_CPC)])
# plt.title('CPC')
# plt.savefig('CPC.pdf')

# plt.style.use('classic')
# fig, ax = plt.subplots(nrows=5)
# for i in range(4):
#     ax[0].plot(data[:,i], lw=0.8)
# plt.yticks([-2, -1, 0, 1, 2])

# ax[1].imshow(groundtruth2.reshape(1, -1), aspect='auto', interpolation='nearest', cmap='tab20')
# ax[2].imshow(embeddings_LSE[::50].T, aspect='auto', interpolation='nearest', cmap='plasma')
# ax[3].imshow(embeddings_Triplet[::50].T, aspect='auto', interpolation='nearest', cmap='plasma')
# ax[4].imshow(embeddings_TNC[::50].T, aspect='auto', interpolation='nearest', cmap='plasma')
# plt.tight_layout()
# plt.show()

def embedding_space22(embeddings, label=None, alpha=0.5, s=0.1, color='blue', show=False):
    color_list = ['b', 'r', 'g', 'purple', 'y', 'gray']
    embeddings = np.array(embeddings)
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.style.use('classic')
    plt.grid()
    i = 0
    if label is not None:
        for l in set(label):
            idx = np.argwhere(label==l)
            plt.scatter(x[idx],y[idx],alpha=alpha,s=s, color=color_list[i])
            i+=1
    else:
        plt.scatter(x,y,alpha=0.5,s=s)
    plt.xticks(size=5)
    plt.yticks(size=5)
    
    
# plt.figure(figsize=(4,5))
# grid_ = plt.GridSpec(11,8, wspace=8, hspace=8)
# # grid_ = plt.GridSpec(11,8)
# plt.style.use('classic')
# plt.subplot(grid_[0:2,:])
# for i in range(4):
#     plt.plot(data[:,i], lw=0.8)
# plt.yticks([-2, -1, 0, 1, 2])

# plt.subplot(grid_[3:7,0:4])
# embedding_space22(embeddings_LSE, show=False, s=1, label=groundtruth[:len(embeddings_LSE)])
# plt.title('LSE',size=10)
# plt.subplot(grid_[3:7,4:8])
# embedding_space22(embeddings_Triplet, show=False, s=1, label=groundtruth[:len(embeddings_Triplet)])
# plt.title('Triplet',size=10)

# plt.subplot(grid_[7:11,0:4])
# embedding_space22(embeddings_TNC, show=False, s=1, label=groundtruth[:len(embeddings_TNC)])
# plt.title('TNC',size=10)

# plt.subplot(grid_[7:11,4:8])
# embedding_space22(embeddings_CPC, show=False, s=1, label=groundtruth[:len(embeddings_CPC)])
# plt.title('CPC',size=10)

# # plt.tight_layout()
# plt.show()


plt.figure(figsize=(4,5))
grid_ = plt.GridSpec(10,8, wspace=8, hspace=8)
# grid_ = plt.GridSpec(11,8)
plt.style.use('classic')
plt.subplot(grid_[0:2,:])
for i in range(4):
    plt.plot(data[:,i], lw=0.8)
plt.yticks([-2, -1, 0, 1, 2],size=5)
plt.xticks(size=5)
plt.title('Raw Data',size=8)

plt.subplot(grid_[2:6,0:4])
embedding_space22(embeddings_LSE, show=False, s=.3, label=groundtruth[:len(embeddings_LSE)])
plt.title('LSE',size=8)
plt.subplot(grid_[2:6,4:8])
embedding_space22(embeddings_Triplet, show=False, s=.3, label=groundtruth[:len(embeddings_Triplet)])
plt.title('Triplet',size=8)

plt.subplot(grid_[6:10,0:4])
embedding_space22(embeddings_TNC, show=False, s=.3, label=groundtruth[:len(embeddings_TNC)])
plt.title('TNC',size=8)

plt.subplot(grid_[6:10,4:8])
embedding_space22(embeddings_CPC, show=False, s=.3, label=groundtruth[:len(embeddings_CPC)])
plt.title('CPC',size=8)

# plt.tight_layout()
plt.show()
