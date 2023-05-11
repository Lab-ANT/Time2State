import matplotlib.pyplot as plt
import numpy as np

# Time2State
# PAMAP2: Average ARI: 0.345196, NMI: 0.635574
# MoCap: Average ARI: 0.781271, NMI: 0.794367
# USC-HAD: Average ARI: 0.604856, NMI: 0.795647
# UCR-SEG: Average ARI: 0.432147, NMI: 0.484898
# synthetic_data: Average ARI: 0.817618, NMI: 0.826886
# ActRecTut: Average ARI: 0.811926, NMI: 0.763406

# FLOSS-euclidean
# UCR-SEG: Average ARI is 0.916917, Average NMI is 0.905667
# ActRecTut: Average ARI is 0.192174, Average NMI is 0.283981
# synthetic_data: Average ARI is 0.123059, Average NMI is 0.244610
# MoCap: Average ARI is 0.393796, Average NMI is 0.560234
# USC-HAD: Average ARI is 0.198402, Average NMI is 0.542815
# PAMAP2: Average ARI is 0.034317, Average NMI is 0.229006

# FLOSS-dtw
# ActRecTut: Average ARI is 0.237908, Average NMI is 0.346167
# synthetic_data: Average ARI is 0.105247, Average NMI is 0.275834
# MoCap: Average ARI is 0.404689, Average NMI is 0.561100
# USC-HAD: Average ARI is 0.196724, Average NMI is 0.528533
# UCR-SEG: Average ARI is 0.944433, Average NMI is 0.924668

def ARI():
    plt.figure(figsize=(10, 3.5))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
    Time2State =  [0.8176, 0.7812, 0.8119, 0.3236, 0.6048, 0.4321]
    FLOSS_EUCLIDEAN      =  [0.1230, 0.3937, 0.1921, 0.0343, 0.1984, 0.9169]
    FLOSS_DTW      =  [0.1052, 0.4046, 0.2379, 0, 0.1967, 0.9444]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.2  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - width, Time2State, width, label='Time2State', hatch='\\.')
    plt.bar(x , FLOSS_EUCLIDEAN, width, label='FLOSS+TSKMeans-euclidean', hatch='\\\.')
    plt.bar(x + width, FLOSS_DTW, width, label='FLOSS+TSKMeans-dtw', hatch='///.')

    plt.ylabel('ARI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()

def NMI():
    plt.figure(figsize=(10, 3.5))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
    Time2State =  [0.8268, 0.7943, 0.7634, 0.6355, 0.7956, 0.4848]
    FLOSS_EUCLIDEAN      =  [0.2446, 0.5602, 0.2839, 0.2290,  0.5428, 0.9056]
    FLOSS_DTW =  [0.2758, 0.5611, 0.3461, 0, 0.5285, 0.9246]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.25  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - width, Time2State, width, label='Time2State', hatch='\\.')
    plt.bar(x , FLOSS_EUCLIDEAN, width, label='FLOSS+TSKMeans-euclidean', hatch='\\\.')
    plt.bar(x + width, FLOSS_DTW, width, label='FLOSS+TSKMeans-dtw', hatch='///.')

    plt.ylabel('NMI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()

ARI()
NMI()