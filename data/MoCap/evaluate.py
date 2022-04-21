from numpy.lib.function_base import average
import steval.segmentation as ss
import steval.label as sl
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import steval.segmentation as ss

# MoCap = {'amc_86_01':{'n_segs':4, 'label':{588:1,1200:2,2006:1,2530:3,3282:1,4048:4,4579:3}},
#          'amc_86_02':{'n_segs':8, 'label':{1009:1,1882:2,2677:3,3158:4,4688:5,5963:1,7327:6,8887:7,9632:8,10617:1}},
#          'amc_86_03':{'n_segs':7, 'label':{872:1, 1938:2, 2448:3, 3470:1, 4632:4, 5372:5, 6182:6, 7089:7, 8401:1}},
#          'amc_86_07':{'n_segs':6, 'label':{1060:1,1897:2,2564:3,3665:2,4405:3,5169:4,5804:5,6962:1,7806:6,8702:1}},
#          'amc_86_08':{'n_segs':9, 'label':{1062:1,1904:2,2661:3,3282:4,3963:5,4754:6,5673:7,6362:5,7144:8,8139:9,9206:1}},
#          'amc_86_09':{'n_segs':5, 'label':{921:1,1275:2,2139:3,2887:4,3667:5,4794:1}},
#          'amc_86_10':{'n_segs':4, 'label':{2003:1,3720:2,4981:1,5646:3,6641:4,7583:1}},
#          'amc_86_11':{'n_segs':4, 'label':{1231:1,1693:2,2332:3,2762:2,3386:4,4015:3,4665:2,5674:1}},
#          'amc_86_14':{'n_segs':3, 'label':{671:1,1913:2,2931:1,4134:3,5051:1,5628:2,6055:3}},}

MoCap = {'amc_86_01':{'n_segs':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02':{'n_segs':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07':{'n_segs':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08':{'n_segs':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09':{'n_segs':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10':{'n_segs':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11':{'n_segs':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14':{'n_segs':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

# TICC
path = 'result_TICC/'
ari_list = []
ami_list = []
accuracy_list = []
macrof1_list = []
for f in os.listdir(path):
    # print(f)
    prediction = pd.read_csv(path+f, header=None)
    prediction = ss.adjust_label(np.array(prediction).flatten())
    groundtruth = ss.adjust_label(sl.seg_to_label(MoCap[f[:-4]]['label'])[2:])
    ari = ss.ARI(groundtruth, prediction)
    ami = ss.AMI(groundtruth, prediction)
    accuracy = metrics.accuracy_score(groundtruth,prediction)
    macrof1 = metrics.f1_score(groundtruth,prediction,average='macro')
    ari_list.append(ari)
    ami_list.append(ami)
    accuracy_list.append(accuracy)
    macrof1_list.append(macrof1)
#     print(f+' ARI: %f, AMI: %f, ACC: %f' %(ari, ami, accuracy))
# print(np.mean(ari_list),np.mean(ami_list),np.mean(accuracy_list))
    print(f+' ARI: %f, AMI: %f, ACC: %f, F1: %f' %(ari, ami, accuracy, macrof1))
print(np.mean(ari_list),np.mean(ami_list),np.mean(accuracy_list), np.mean(macrof1))

print('StateCorr with spectral2')
path = 'result_StateCorr_spectral/'
offset=64
ari_list = []
ami_list = []
accuracy_list = []
macrof1_list = []
for f in os.listdir(path):
    # print(f)
    prediction = pd.read_csv(path+f, header=None)
    prediction = ss.adjust_label(np.array(prediction).flatten())
    groundtruth = ss.adjust_label(sl.seg_to_label(MoCap[f[:-4]]['label'])[offset:-offset+1])
    ari = ss.ARI(groundtruth, prediction)
    ami = ss.AMI(groundtruth, prediction)
    accuracy = metrics.accuracy_score(groundtruth,prediction)
    macrof1 = metrics.f1_score(groundtruth,prediction,average='macro')
    ari_list.append(ari)
    ami_list.append(ami)
    accuracy_list.append(accuracy)
    macrof1_list.append(macrof1)
    print(f+' ARI: %f, AMI: %f, ACC: %f, F1: %f' %(ari, ami, accuracy, macrof1))
print(np.mean(ari_list),np.mean(ami_list),np.mean(accuracy_list), np.mean(macrof1))

# print('Autoplait')
# base_dir = '_out2/dat_tmp/'
# for dir_name in os.listdir(base_dir):
#     print(dir_name)
#     print(os.listdir(base_dir+dir_name))

# def read_seg(path):
#     with open(path) as f:
#         print(f.readlines())

# read_seg('_out2/dat_tmp/dat1/segment.0')