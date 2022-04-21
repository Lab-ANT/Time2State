import pandas as pd
import numpy as np
import sys
sys.path.append('./')
from Time2State.adapers import *
from Time2State.default_params import *
from TSpy.view import *
import os
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.dirname(__file__), '../data/IHEPC/household_power_consumption.txt')
# Dataset as a dataframe
df = pd.read_csv(data_path, sep=';', decimal=',')
# Replace missing values by the last seen values
dataset = np.transpose(np.array(df))[2].reshape(1, 1, -1)
for i in range(np.shape(dataset)[2]):
    if dataset[0, 0, i] == '?':
        dataset[0, 0, i] = dataset[0, 0, i-1]
dataset = dataset.astype(float)
print(dataset.shape)
dataset = dataset.reshape(1,-1).T
print(dataset.shape)

dataset = dataset[:500000,:]
# plot_mulvariate_time_series(dataset, show=False)
# # plt.savefig('./fig.png')
# plt.show()
params_LSE['in_channels'] = 1
params_LSE['out_channels'] = 2
params_LSE['M'] = 20
params_LSE['N'] = 4
params_LSE['nb_steps'] = 20
params_LSE['compared_length'] = 512
encoder = CausalConv_LSE_Adaper(params_LSE)
encoder.fit(dataset)
embeddings = encoder.encode(dataset, 512, 10)
embedding_space(embeddings, show=False, s=3)
# plt.savefig('./fig.png')
plt.show()
density_map(embeddings, show=False)
plt.show()
# plt.savefig('./density.png')
