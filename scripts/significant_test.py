import scipy
import os
from TSpy.eval import evaluate_clustering
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# methods = ['Time2State', 'TICC', 'HDP_HSMM', 'AutoPlait', 'ClaSP']
methods = ['Time2State', 'TICC', 'HDP_HSMM', 'AutoPlait', 'ClaSP', 'HVGH']
datasets = ['synthetic', 'MoCap', 'USC-HAD', 'UCR-SEG', 'ActRecTut', 'PAMAP2']

script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../results/output_')

def is_dataset_exist(method, dataset):
    return os.path.exists(os.path.join(output_path+method,dataset))

def evaluate(dataset, method, verbose=True):
    out_path = os.path.join(output_path+method,dataset)
    ari_list = []
    nmi_list = []
    f_list = os.listdir(out_path)
    f_list.sort()
    for fname in f_list:
        fpath = os.path.join(out_path, fname)
        result = np.load(fpath)
        groundtruth = result[0,:].flatten()
        prediction = result[1,:].flatten()
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        ari_list.append(np.array(ari))
        nmi_list.append(np.array(nmi))
        ari_mean = np.mean(ari_list)
        nmi_mean = np.mean(nmi_list)
    return ari_list, nmi_list, ari_mean, nmi_mean

for dataset in datasets:
    print('==========%s========='%(dataset))
    ari_list_of_methods = []
    nmi_list_of_methods = []

    for method in methods:
        if not is_dataset_exist(method, dataset):
            continue
        ari, nmi, ari_mean, nmi_mean = evaluate(dataset, method, verbose=True)
        ari_list_of_methods.append(ari)
        nmi_list_of_methods.append(nmi)
        print(dataset, method, ari_mean, nmi_mean)
    
    num_methods = len(methods)
    print('-------- ARI --------')
    for i in range(num_methods):
        group = []
        for j in range(num_methods):
            if i==j:
                continue
            _, p = scipy.stats.ttest_ind(ari_list_of_methods[i], ari_list_of_methods[j])
            # print(methods[i], methods[j], p)
            if p >= 0.05:
                group.append(j)
        print(methods[i],[methods[k] for k in group])

    print('-------- NMI --------')
    for i in range(num_methods):
        group = []
        for j in range(num_methods):
            if i==j:
                continue
            _, p = scipy.stats.ttest_ind(nmi_list_of_methods[i], nmi_list_of_methods[j])
            # print(methods[i], methods[j], p)
            if p >= 0.05:
                group.append(j)
        print(methods[i],[methods[k] for k in group])