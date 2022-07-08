import numpy as np
import os
from TSpy.utils import *
from TSpy.view import *
from TSpy.label import *
import warnings
warnings.filterwarnings("ignore")

method = 'ClaSP'
script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../results/output_'+method)
figure_out_put_path = os.path.join(script_path, '../results/figures/')

def evaluate(dataset):
    out_path = os.path.join(output_path,dataset)
    f_list = os.listdir(out_path)
    f_list.sort()
    for fname in f_list:
        fpath = os.path.join(out_path, fname)
        result = np.load(fpath)
        groundtruth = adjust_label(result[0,:].flatten())
        prediction = adjust_label(result[1,:].flatten())
        plt.step(np.arange(len(groundtruth)), groundtruth)
        plt.step(np.arange(len(groundtruth)), prediction)
        plt.savefig(figure_out_put_path + 'result'+fname+'.png')
        plt.close()

# datasets = ['synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD2', 'UCR-SEG']
datasets = ['USC-HAD2']
for dataset in datasets:
    evaluate(dataset=dataset)