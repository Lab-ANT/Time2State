import os
import json
from miniutils import cluster_segs, create_path
import pandas as pd
import numpy as np
import time

dataset_list = ['MoCap', 'UCR-SEG', 'ActRecTut', 'synthetic_data', 'PAMAP2', 'USC-HAD']

metric = 'euclidean'

script_path = os.path.dirname(__file__)
dataset_path = os.path.join(script_path, 'lastCode/data/UCR-SEG/UCR_datasets_seg')
seg_result_path = os.path.join(script_path, 'extracted_seg_pos/UCR-SEG_result.json')
clustering_result_save_path = os.path.join(script_path, 'output_FLOSS-%s/clustering'%(metric))

def cluster_on(dataset):
    create_path(os.path.join(clustering_result_save_path, dataset))
    print(dataset)

    t1 = time.time()
    cps_save_path = os.path.join(script_path, 'extracted_seg_pos/%s_result.json'%(dataset))
    with open(cps_save_path, 'r') as f:
        cps_json = json.load(f)
    for file_name in list(cps_json):
        print(file_name)
        data = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s'%(dataset, file_name)), header=None).to_numpy()
        groundtruth = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s.label'%(dataset, file_name[:-4])), header=None).to_numpy()
        cps_list = cps_json[file_name]
        num_states = len(set(list(groundtruth.flatten())))

        # SimpleSynthetic_125_3000_5000.txt in the UCR-SEG dataset is a special case.
        if file_name == 'SimpleSynthetic_125_3000_5000.txt':
            num_states -= 1

        # Sort boundaries by their temporal order.
        cps_list.sort()
        clustering_result = cluster_segs(data, found_cps=cps_json[file_name], n_states=num_states, metric=metric)
        np.save(os.path.join(clustering_result_save_path, '%s/%s'%(dataset,file_name[:-4])), clustering_result)
    t2 = time.time()
    print('Time consumption: %s seconds'%((t2-t1)))

for dataset in dataset_list:
    cluster_on(dataset)
