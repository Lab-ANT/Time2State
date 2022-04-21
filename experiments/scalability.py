import numpy as np
import time
import sys
sys.path.append('./')
import os
from TSpy.eval import *
from TSpy.label import *
from TSpy.utils import *
from TSpy.view import *
from TSpy.dataset import *
from Time2State.time2state import *
from Time2State.default_params import *
from Time2State.adapers import *
from Time2State.clustering import *

data_path = os.path.join(os.path.dirname(__file__), '../data')
def warm_up():
    path = os.path.join(data_path, 'synthetic_data_for_segmentation/test0.csv')
    data = np.loadtxt(path, delimiter=',')
    data = np.concatenate([data[:,:4] for x in range(15)])

    params_LSE['in_channels'] = 4
    params_LSE['compared_length'] = 512
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 5

    Time2State(512, 100, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data[:10000,:], 512, 100)

def effect_of_length():
    time_list = []
    path = os.path.join(data_path, 'synthetic_data_for_segmentation/test0.csv')
    data = np.loadtxt(path, delimiter=',')
    data = np.concatenate([data[:,:4] for x in range(15)])
    params_LSE['in_channels'] = 4
    params_LSE['compared_length'] = 512
    params_LSE['M'] = 20
    params_LSE['N'] = 10
    params_LSE['nb_steps'] = 20

    warm_up()
    for length in range(1,21):
        time_start=time.time()
        Time2State(512, 50, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data[:length*10000,:], 512, 50)
        time_end=time.time()
        print('length: %d, time: %f'%(int(length), round(time_end-time_start,2)))
        time_list.append(time_end-time_start)
    time_list = np.array(time_list)
    print(time_list.round(2))

def effect_of_dimension():
    time_list = []
    data = np.loadtxt(os.path.join(data_path, 'synthetic_data_for_segmentation/test0.csv'), delimiter=',')
    data = data[:,:4]
    data = np.hstack([data, data, data, data, data])
    data = np.vstack([data, data])
    data = data[:30000,:]
    # print(data.shape,data)
    params_LSE['compared_length'] = 512
    params_LSE['M'] = 20
    params_LSE['N'] = 10
    params_LSE['nb_steps'] = 20
    warm_up()
    for i in range(1,21):
        params_LSE['in_channels'] = i
        print(data[:,:i].shape)
        testdata = data[:,:i]
        time_start=time.time()
        Time2State(512, 50, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(testdata[:,:i], 512, 50)
        time_end=time.time()
        print(i,time_end-time_start)
        time_list.append(time_end-time_start)
    time_list = np.array(time_list)
    print(time_list.round(2))

effect_of_length()
# effect_of_dimension()