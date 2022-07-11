import os
from TSpy.eval import evaluate_clustering
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd


methods = ['Time2State', 'Triplet', 'TNC', 'CPC', 'TS2Vec']
# methods = ['Time2State', 'TICC', 'HDP_HSMM', 'AutoPlait', 'ClaSP', 'HVGH']
datasets = ['MoCap', 'USC-HAD', 'UCR-SEG', 'ActRecTut', 'PAMAP2']

script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../results/output_')
target_path = os.path.join(script_path, '../results')

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

results = []
for method in methods:
    print('==========%s========='%(method))
    ari_list_of_datasets = []
    nmi_list_of_datasets = []
    for dataset in datasets:
        ari_list, nmi_list, ari_mean, nmi_mean = evaluate(dataset, method, verbose=True)
        ari_list_of_datasets.append(ari_list)
        nmi_list_of_datasets.append(nmi_list)
        print(dataset, method, ari_mean, nmi_mean)
    ari_list_of_datasets = np.concatenate(ari_list_of_datasets)
    print(len(ari_list_of_datasets))
    results.append(ari_list_of_datasets)
results = np.concatenate(results)

list_method_name = []
list_dataset_name = []

for method in methods:
    list_dataset_name += ['dataset'+str(i) for i in range(139)]

for method in methods:
    list_method_name += [method for i in range(139)]

data = {'classifier_name':list_method_name, 'dataset_name':list_dataset_name, 'accuracy':results}
# print(len(list_method_name), len(list_dataset_name), len(results))
df = pd.DataFrame(data)
# print(df)
df.to_csv(os.path.join(target_path, 'CD.csv'))