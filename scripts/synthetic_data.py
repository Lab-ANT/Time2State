import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shape import RMDF
import tqdm

# channel number
channel_num = 4
# segment number
seg_num = 30
seg_len = 1000
# must be lower than seg_num.
state_num = 5
ts_num = 100
# shape_len = 200
num_shape = 5

# Generate segment json
def generate_seg_json(seg_num, state_num, random_state=None):
    # Config seed to generate determinally.
    if random_state is not None:
        np.random.seed(random_state)
    seg_json = {}
    state_list = np.random.randint(state_num, size=seg_num)
    total_len = 1000
    for i, state in zip(range(seg_num), state_list):
        seg_len = 1000
        seg_json[total_len]=state
        total_len += seg_len
    return seg_json

def gen_channel_from_json(seg_json, shape_len_list):
    state_list = [seg_json[seg] for seg in seg_json]
    # print(state_list)
    true_state_num = len(set(state_list))
    # This is an object list.
    rmdf_list = [RMDF.RMDF(depth=5) for i in range(true_state_num)]
    for rmdf in rmdf_list:
        rmdf.gen_anchor()
    seg_list = []
    for state in state_list:
        seg = [rmdf_list[state].gen(forking_depth=2, length=shape_len_list[state]) for i in range(num_shape)]
        seg_list.append(np.concatenate(seg))
    result = np.concatenate(seg_list)
    return result

def gen_from_json(seg_json):
    # generate channel respectively.
    shape_len_list = np.random.randint(20,200,size=state_num)
    # print(shape_len_list)
    channel_list = [gen_channel_from_json(seg_json, shape_len_list) for i in range(channel_num)]
    return np.stack(channel_list).T, shape_len_list

seg_json_list = []
for i in range(ts_num):
    seg_json =generate_seg_json(seg_num, state_num)
    while len(set([seg_json[key] for key in list(seg_json)])) != state_num:
        seg_json = generate_seg_json(seg_num, state_num)
    seg_json_list.append(seg_json)

def gen_new_seg_json(seg_json, shape_len_list):
    state_list = [seg_json[seg] for seg in seg_json]
    new_seg_json = {}
    total_length = 0
    for state in state_list:
        total_length+=shape_len_list[state]*num_shape
        new_seg_json[total_length] = state
    return new_seg_json

from TSpy.label import seg_to_label, adjust_label
import os
script_path = os.path.dirname(__file__)
path = os.path.join(script_path, '../data/synthetic_data_for_segmentation3')

for i, seg_json in enumerate(seg_json_list):
    data,shape_len_list = gen_from_json(seg_json)
    seg_json = gen_new_seg_json(seg_json, shape_len_list)
    label = adjust_label(seg_to_label(seg_json))
    len_of_data = len(data)
    len_of_label = len(label)
    min_len = len_of_data if len_of_data<=len_of_label else len_of_label
    data=data[:min_len]
    label=label[:min_len]
    df = pd.DataFrame(data)
    df.insert(4, 'label', label)
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_csv(path+'/test'+str(i)+'.csv', header=False, index=False)

# path = '../data/synthetic_data_for_Autoplait'
# if not os.path.exists(path):
#     os.mkdir(path)
# with open(path+'/list', 'w') as f:
#     for i in range(ts_num):
#         f.writelines('../data/synthetic_data_for_Autoplait/test'+str(i)+'.csv\n')
# for i, seg_json in enumerate(seg_json_list):
#     data = gen_from_json(seg_json)
#     # plot_mulvariate_time_series(data)
#     label = adjust_label(seg_to_label(seg_json))
#     df = pd.DataFrame(data).round(4)
#     df.insert(4, 'label', label)
#     if not os.path.exists(path):
#         os.mkdir(path)
#     df.to_csv(path+'/test'+str(i)+'.csv', header=False, index=False, sep=' ')