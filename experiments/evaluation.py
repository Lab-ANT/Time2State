import numpy as np
import os
from TSpy.eval import evaluate_clustering
from TSpy.utils import *
import warnings
warnings.filterwarnings("ignore")

method = 'AutoPlait'

script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../results/output_'+method)

def evaluate(dataset='MoCap', verbose=True):
    out_path = os.path.join(output_path,dataset)
    score_list = []
    f_list = os.listdir(out_path)
    f_list.sort()
    for fname in f_list:
        fpath = os.path.join(out_path, fname)
        result = np.load(fpath)
        groundtruth = result[0,:].flatten()
        prediction = result[1,:].flatten()
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

datasets = ['synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
for dataset in datasets:
    evaluate(dataset=dataset,verbose=False)