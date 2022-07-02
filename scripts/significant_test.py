import scipy
import os
from TSpy.eval import evaluate_clustering
import numpy as np

methods = ['Time2State', 'TICC', 'HDP_HSMM']
datasets = ['synthetic', 'MoCap', 'USC-HAD']

script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../results/output_')

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
        ari, nmi, ari_mean, nmi_mean = evaluate(dataset, method, verbose=True)
        ari_list_of_methods.append(ari)
        nmi_list_of_methods.append(nmi)
        print(dataset, method, ari_mean, nmi_mean)
    
    for i in range(3):
        group = []
        for j in range(3):
            if i==j:
                continue
            _, p = scipy.stats.ttest_ind(ari_list_of_methods[i], ari_list_of_methods[j])
            # print(methods[i], methods[j], p)
            if p >= 0.05:
                group.append(j)
        print(methods[i],[methods[k] for k in group])