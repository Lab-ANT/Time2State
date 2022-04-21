import matplotlib.pyplot as plt
import numpy as np
import stview.colors as sc

def label_to_seg(label):
    label_set = set(label)
    for l in label_set:
        pass

label = [0,0,0,1,1,1,2,2,2,0,0,0]
seg = {3:0,6:1,9:2,12:0}

# pre = 0
# for i in seg:
#     print(np.arange(pre,i+1))
#     plt.step(np.arange(pre,i+1),np.ones(i+1-pre),color=sc.associated_colors[seg[i]],linewidth=10)
#     pre = i

# plt.show()

dataset = {'amc_86_01.4d':{'n_segs':4, 'label':{588:1,1200:2,2006:1,2530:3,3282:1,4048:4,4579:3}},
        'amc_86_02.4d':{'n_segs':8, 'label':{1009:1,1882:2,2677:3,3158:4,4688:5,5963:1,7327:6,8887:7,9632:8,10617:1}},
        'amc_86_03.4d':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07.4d':{'n_segs':6, 'label':{1060:1,1897:2,2564:3,3665:2,4405:3,5169:4,5804:5,6962:1,7806:6,8702:1}},
        'amc_86_08.4d':{'n_segs':9, 'label':{1062:1,1904:2,2661:3,3282:4,3963:5,4754:6,5673:7,6362:5,7144:8,8139:9,9206:1}},
        'amc_86_09.4d':{'n_segs':5, 'label':{921:1,1275:2,2139:3,2887:4,3667:5,4794:1}},
        'amc_86_10.4d':{'n_segs':4, 'label':{2003:1,3720:2,4981:1,5646:3,6641:4,7583:1}},
        'amc_86_11.4d':{'n_segs':4, 'label':{1231:1,1693:2,2332:3,2762:2,3386:4,4015:3,4665:2,5674:1}},
        'amc_86_14.4d':{'n_segs':3, 'label':{671:1,1913:2,2931:1,4134:3,5051:1,5628:2,6055:3}},
}
import steval.label as sl
import stview.colors as sc
label = sl.seg_to_label({588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2})
segments = sc.get_segment_label(label)
print(seg)
start = 0
plt.step(np.arange(len(label)),label,color='black',linestyle='-.')
for seg in segments:
    end = start + len(seg)
    plt.step(np.arange(start,end),label[start:end],color=sc.associated_colors[seg[0]],linewidth=5.0)
    start += len(seg)
plt.show()
