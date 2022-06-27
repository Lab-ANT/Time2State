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

import os, sys, time
sys.path.append("Baselines/ClaSP/")

import numpy as np
# np.random.seed(1379)

import pandas as pd
import scipy

from TSpy.utils import len_of_file
from TSpy.label import *
from TSpy.eval import evaluate_clustering
from TSpy.utils import *
from TSpy.dataset import *

from src.clasp import extract_clasp_cps, extract_clasp_cps_from_multivariate_ts

import os

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../../../data')

def effect_of_length():
    time_list = []
    path = os.path.join(data_path, 'synthetic_data_for_segmentation/test0.csv')
    data = np.loadtxt(path, delimiter=',')
    data = np.concatenate([data[:,:4] for x in range(15)])

    for length in range(1,21):
        time_start=time.time()
        extract_clasp_cps_from_multivariate_ts(data[:length*10000], 50, 20)
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
    for i in range(1,21):
        testdata = data[:,:i]
        time_start=time.time()
        extract_clasp_cps_from_multivariate_ts(testdata, 50, 20)
        time_end=time.time()
        print('dim: %d, time: %f'%(int(i), round(time_end-time_start,2)))
        time_list.append(time_end-time_start)
    time_list = np.array(time_list)
    print(time_list.round(2))

# effect_of_length()
effect_of_dimension()