# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

def read_result_dir(path, original_file_path):
    results = os.listdir(path)
    # TODO: fix this but.
    results = [s if re.match('segment.\d', s) != None else None for s in results]
    results = np.array(results)
    idx = np.argwhere(results!=None)
    results = results[idx].flatten()

    label_list = []
    dta = pd.read_csv(original_file_path, names=['col0','col1','col2','col3'], index_col=False, header=None, sep=" ")
    length = len(dta)

    l = 0
    for r in results:
        labels = np.zeros(length)
        data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
        start = data.col0
        end = data.col1
        for s, e in zip(start,end):
            labels[s:e+1]=l
        l+=1
        label_list.append(labels)
    return label_list

def convert_result():
    out_path = '_dat/state_seq'
    for i in range(1,5):
        for j in range(1,5):
            label_list = read_result_dir('_out/dat_tmp/dat'+str(i*j)+'/', '_dat/t'+str(i)+'_'+str(j)+'.1d')

# convert_result()

# print(results)
# path = '_out/dat_tmp/'
# output_path = '_dat/'
# results = os.listdir('_out/dat_tmp/dat16')
# # print(re.match('segment.*', results))

# results = [s if re.match('segment.\d', s) != None else None for s in results]
# results = np.array(results)
# idx = np.argwhere(results!=None)
# results = results[idx].flatten()
# print(results)

# data1 = pd.read_csv('_dat2/86_03.amc.4d', names=['col0','col1','col2','col3'], index_col=False, header=None, sep=" ")
# length = len(data1)
# labels = np.zeros(length)
# print(length)

# l = 0
# for r in results:
#     data = pd.read_csv(path+'/dat16/'+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
#     start = data.col0
#     end = data.col1
#     for s, e in zip(start,end):
#         labels[s:e+1]=l
#     l+=1
#     print(data)

# print(labels)

# sub1 = plt.subplot(241)
# sub2 = plt.subplot(242)
# sub3 = plt.subplot(243)
# sub4 = plt.subplot(244)
# sub5 = plt.subplot(245)
# sub6 = plt.subplot(246)
# sub7 = plt.subplot(247)
# sub8 = plt.subplot(248)
# sub1.plot(data1.col0)
# sub5.step(np.arange(len(labels)),labels)
# plt.show()
