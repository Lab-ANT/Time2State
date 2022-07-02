import numpy as np
import os
from TSpy.eval import evaluate_clustering
from TSpy.utils import *

script_path = os.path.dirname(__file__)
# Path for saving resutls.
output_path = os.path.join(script_path, '../results/output_HDP_HSMM')

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
        # print("TICC,dataset%d,%f"%(int(fname[:-4])+1,ari))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- , ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

# evaluate(dataset='MoCap',verbose=True)
# evaluate(dataset='synthetic',verbose=False)
# evaluate(dataset='UCR-SEG',verbose=True)
# evaluate(dataset='ActRecTut',verbose=True)
# evaluate(dataset='PAMAP2',verbose=True)
evaluate(dataset='USC-HAD',verbose=True)