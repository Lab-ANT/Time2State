import sys
sys.path.append('./')
import pandas as pd
import os
from TSpy.eval import *
from TSpy.label import *
from TSpy.utils import *
from TSpy.view import *
from TSpy.dataset import *
from Time2State.time2state import *
from Time2State.adapers import *
from Time2State.clustering import *
import torch
from torch import nn

# transformer_model = nn.Transformer(d_model=32, nhead=16, num_encoder_layers=12, batch_first=True)
# src = torch.rand((1, 4, 32))
# tgt = torch.rand((1, 4, 32))
# out = transformer_model(src, tgt)
# print(out.shape)


# data_path = os.path.join(os.path.dirname(__file__), '../data/synthetic_data_for_segmentation/test0.csv')

# params_TNC = {
#     'win_size' : 128,
#     'in_channels' : 4,
#     'nb_steps' : 5,
#     'out_channels' : 4,
# }

# df = pd.read_csv(data_path, sep=',', header=None)
# X = df.to_numpy()[:,:4]
# t2s = Time2State(128, 10, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(X, 128, 10)
# print(t2s.embeddings.shape)

# a = [1,2,3,4,5,6,7,8,9,10]

# # convert data to a list of window.
# def time2window(X, win_size, step):
#     # length = X.shape[2]
#     length = len(X)
#     print(length)
#     # exactly [(length-win_size)/step+1] windows.
#     if (length-win_size)%step == 0:
#         num_of_windows = int((length-win_size)/step+1)
#         # list_window = [X[:,:,i:i+win_size] for i in range(0, num_of_windows*step, step)]
#         list_window = [X[i:i+win_size] for i in range(0, num_of_windows*step, step)]
#     else:
#         num_of_windows = int((length-win_size)/step+2)
#         # list_window = [X[:,:,i:i+win_size] for i in range(0, num_of_windows*step, step)]
#         list_window = [X[i:i+win_size] for i in range(0, num_of_windows*step, step)]
#     return list_window
# print(time2window(a,3,2))

# data = torch.rand(1,2,3)
# print(data)
# print(data.squeeze(2))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f(x):
    return 1-sigmoid(-x)
    # return sigmoid(x)-1

print(f(np.arange(-1,1,0.01)))
plt.plot(f(np.arange(-1,1,0.01)))
plt.show()