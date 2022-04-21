import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def ARI():
    plt.figure(figsize=(8, 3))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
    Time2State =  [0.8503, 0.7529, 0.7670, 0.3135, 0.6522]
    GHMM       =  [0.3285, 0.3650, 0.6063, 0.2096, 0.4198]
    TICC_      =  [0.6242, 0.7218, 0.7839, 0.3008, 0.3947]
    Autoplait_ =  [0.0713, 0.8057, 0.0586, 0.0001, 0.2948]
    HVGH_      =  [0.0809, 0.0500, 0.0881, 0.0032, 0.0788]
    KMeans     =  [0.0429, 0.0228, 0.0273, 0.0200, 0.0200]
    HDP_HSMM   =  [0.6619, 0.5509, 0.6644, 0.2882, 0.4678]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.14  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 2*width, Time2State, width, label='Time2Seg',hatch='//.')
    plt.bar(x - 1*width, TICC_, width, label='TICC', hatch='/.')
    # plt.bar(x - 0.5*width, GHMM, width, label='GHMM',hatch='\\\\.')
    plt.bar(x + 0*width, HDP_HSMM, width, label='HDP-HSMM', hatch='\\.')
    plt.bar(x + 1*width, Autoplait_, width, label='AutoPlait', hatch='///.')
    plt.bar(x + 2*width, HVGH_, width, label='HVGH', hatch='\\\\\\.')
    plt.scatter([3.14],[0.05], marker='x', label='refuse\nto work', color='red')
    plt.ylabel('ARI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=6, fontsize=10, loc='upper center')
    plt.tight_layout()
    plt.show()

def NMI():
    plt.figure(figsize=(8, 3))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
    Time2State  =  [0.8407, 0.7670, 0.7170, 0.5509, 0.8028]
    TICC_ =       [0.7489, 0.7524, 0.7466, 0.5262, 0.7028]
    GHMM       =  [0.4661, 0.4816, 0.6456, 0.4001, 0.5524]
    Autoplait_ =  [0.1307, 0.8289, 0.1418, 0.0000, 0.5413]
    HVGH_ =       [0.1606, 0.1523, 0.2088, 0.0374, 0.1883]
    HDP_HSMM   =  [0.8105, 0.7168, 0.6796, 0.5477, 0.7219]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.14  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 2*width, Time2State, width, label='Time2Seg',hatch='//.')
    plt.bar(x - 1*width, TICC_, width, label='TICC', hatch='/.')
    plt.bar(x + 0*width, HDP_HSMM, width, label='HDP-HSMM', hatch='\\.')
    plt.bar(x + 1*width, Autoplait_, width, label='AutoPlait', hatch='///.')
    plt.bar(x + 2*width, HVGH_, width, label='HVGH', hatch='\\\\\\.')
    plt.scatter([3.14],[0.05], marker='x', label='refuse\nto work', color='red')
    plt.ylabel('NMI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=6, fontsize=10, loc='upper center')
    plt.tight_layout()
    plt.show()

ARI()
# NMI()