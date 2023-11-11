from Time2State.time2state import Time2State
from Time2State.adapers import CausalConv_LSE_Adaper
from Time2State.clustering import DPGMM
from Time2State.default_params import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from TSpy.view import plot_mts
import matplotlib.pyplot as plt

win_size = 512
step = 50

# Load data
data = pd.read_csv('data/MoCap/4d/amc_86_02.4d', sep=' ').to_numpy()
data = StandardScaler().fit_transform(data)

# model
params_LSE['in_channels'] = data.shape[1]
params_LSE['out_channels'] = 4
params_LSE['nb_steps'] = 30
params_LSE['win_size'] = win_size
params_LSE['win_type'] = 'hanning' # {rect, hanning}

t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None), params_LSE)
t2s.fit(data, win_size, step)

plt.style.use('classic')
plot_mts(data, t2s.state_seq)
plt.savefig('demo.png')
