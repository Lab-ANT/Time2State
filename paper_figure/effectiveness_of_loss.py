import matplotlib.pyplot as plt
import numpy as np

def NMI():
    plt.figure(figsize=(8, 3))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'USC-HAD', 'PAMAP2']
    LSE      = [0.8407, 0.7734, 0.7170, 0.8028, 0.6172]
    Triplet  = [0.7634, 0.7065, 0.6690, 0.6881, 0.5606]
    TNC      = [0.7677, 0.6997, 0.6839, 0.6897, 0.5261]
    CPC      = [0.6836, 0.6421, 0.5641, 0.6241, 0.4814]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.15  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 1.5*width, LSE, width, label='LSE',hatch='//.')
    plt.bar(x - 0.5*width, Triplet, width, label='Triplet',hatch='\\\\.')
    plt.bar(x + 0.5*width, TNC, width, label='TNC', hatch='/.')
    plt.bar(x + 1.5*width, CPC, width, label='CPC', hatch='\\.')
    plt.ylabel('NMI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()

def ARI():
    plt.figure(figsize=(8, 3))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'USC-HAD', 'PAMAP2']
    LSE      = [0.8503, 0.7696, 0.7383, 0.6522, 0.3188]
    Triplet  = [0.7098, 0.6632, 0.7138, 0.4784, 0.2714]
    TNC      = [0.7229, 0.6845, 0.7239, 0.4553, 0.2717]
    CPC      = [0.5204, 0.5845, 0.4967, 0.4435, 0.2625]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.15  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 1.5*width, LSE, width, label='LSE',hatch='//.')
    plt.bar(x - 0.5*width, Triplet, width, label='Triplet',hatch='\\\\.')
    plt.bar(x + 0.5*width, TNC, width, label='TNC', hatch='/.')
    plt.bar(x + 1.5*width, CPC, width, label='CPC', hatch='\\.')
    plt.ylabel('ARI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()

# ARI()
NMI()