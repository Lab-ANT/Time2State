import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.patches import PathPatch
# from matplotlib.path import Path

def effect_of_length():
    plt.style.use('ggplot')
    # plt.style.use('classic')
    num_x_ticks = 20
    fig = plt.figure(figsize=(8,8))
    consumption_Time2Seg_C = [7.24,8.23,9.24,10.15,11.29,12.26,13.21,14.12,15.2,16.08,17.02,18.21,19.23,20.65,21.73,22.2,22.93,24.11,25.67,26.48]
    consumption_Time2Seg_G = [3.04,3.28,3.43,3.61,3.86,3.98,4.33,4.37,4.54,4.74,5.02,5.31,5.40,5.59,5.89,6.27,6.26,6.48,6.69,6.75]
    consumption_TICC = [30.13,35.86,43.57,49.06,58.85,60.85,69.79,72.85,47.73,94.83,45.58,106.52,51.11,122.86,122.04,132.45,132.84,138.25,151.69,155.94]
    consumption_AutoPlait  = [43.8, 119.76, 179.16]
    # consumption_HVGH = [14, 14.22, 30.88, 43.96, 59.17, 60.97, 90.11, 71.66, 73.99, 85.32, 146.52, 127.93, 117.48, 179.51]
    consumption_HVGH = [47.76, 78.03, 106.98, 147.82, 205.52]
    consumption_HDP_HSMM = [69.86, 139.91, 211.39]
    plt.plot(np.arange(1,num_x_ticks+1,1), np.array(consumption_Time2Seg_C), marker='o', label='Time2Seg-CPU',linewidth=3, markersize=10)
    plt.plot(np.arange(1,num_x_ticks+1,1), np.array(consumption_Time2Seg_G), marker='^', label='Time2Seg-GPU',linewidth=3, markersize=10)
    plt.plot(np.arange(1,num_x_ticks+1,1), np.array(consumption_TICC), marker='s', label='TICC',linewidth=3, markersize=10)
    plt.plot(np.arange(1,4,1), np.array(consumption_AutoPlait), marker='D', label='AutoPlait',linewidth=3, markersize=10)
    plt.plot(np.arange(1,6,1), np.array(consumption_HVGH), marker='<', label='HVGH',linewidth=3, markersize=10)
    plt.plot(np.arange(1,4,1), np.array(consumption_HDP_HSMM)+10, marker='h', label='HDP-HSMM',linewidth=3, markersize=10)

    # add annotation box
    plt.gca().add_patch(plt.Rectangle((8.5,40),5,18, fill=False, edgecolor='black', linewidth=2))
    # add annotation
    plt.text(14, 47, 'Early Stopping', fontsize=20)
    # plt.quiver(13.5, 50, 15.5, 0, color='black', width=0.003)

    plt.xlabel('Length (10k)', size=20)
    plt.ylabel('Time Consumption (s)', size=20)
    plt.xticks(range(2,num_x_ticks+1,2),size=20)
    plt.yticks(size=20)
    plt.legend(ncol=2, fontsize=16, loc='upper right')
    # plt.legend(bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=11, loc='upper center')
    plt.tight_layout()
    plt.tight_layout()
    plt.show()

def effect_of_dimension():
    plt.style.use('ggplot')
    # plt.style.use('classic')
    num_x_ticks = 20
    fig = plt.figure(figsize=(8,8))
    consumption_Time2Seg_C = [9.27,9.21,9.39,9.22,9.89,9.88,9.16,9.76,9.24,9.19,9.2,9.2,9.83,9.19,9.28,9.18,9.96,9.83,9.96,10.25]
    consumption_Time2Seg_G = [3.41,3.41,3.45,3.39,3.42,3.51,3.43,3.43,3.54,3.41,3.44,3.47,3.45,3.45,3.52,3.49,3.48,3.43,3.51,3.5]
    # consumption_AutoPlait  = [10.88, 32.83, 30.781, 71.443,]
    # consumption_AutoPlait  = [11.30,33.03,31.65,175.86,70.80,324.64,330.40,280.52,478.24,472.02,17.86,14.04,12.62,9.12,9.07,9.53,9.36,9.97,10.38,12.21]
    consumption_AutoPlait  = [11.30,33.03,31.65,175.86,70.80,324.64,330.40,280.52,478.24,472.02]#,17.86]
    consumption_TICC = [14.08,16.03,20.02,23.35,20.55,24.5,25.88,33.71,41.97,49.12,68.55,77.62,77.14,85.81,118.13,106.93,126.12,114.18,167.61,191.82]
    # consumption_HVGH = [26.98, 33.77,26.83,27.28,33.64,31.8,39.31,24.46,29.5,27.56,34.07,29.99,34.66,38.26,30.37,34.21,43.64,30.28,29.34,47.3]
    consumption_HVGH = [113.465,123.265,120.865,107.695,115.93,95.8,112.55,116.455,98.93,92.21,101.05,107.07,106.955,98.13,110.62,107.75,97.055,114.49,103.035,109.285]
    consumption_HDP_HSMM = [216.98,213.77,216.83,217.218,223.64,231.8,239.31,224.46,229.5,227.56,234.07,229.99,234.66,238.226,230.37,234.21,243.64,230.28,229.34,247.3]
    plt.plot(np.arange(1,num_x_ticks+1,1), np.array(consumption_Time2Seg_C), marker='o', label='Time2Seg-CPU',linewidth=3, markersize=10)
    plt.plot(np.arange(1,num_x_ticks+1,1), np.array(consumption_Time2Seg_G), marker='^', label='Time2Seg-GPU',linewidth=3, markersize=10)
    plt.plot(np.arange(1,num_x_ticks+1,1), consumption_TICC, marker='s', label='TICC',linewidth=3, markersize=10)
    plt.plot(np.arange(1,11,1), consumption_AutoPlait, marker='D', label='AutoPlait',linewidth=3, markersize=10)
    plt.plot([10], [472.02], marker='x', color='red',linewidth=3, markersize=15)
    plt.plot(np.arange(1,num_x_ticks+1,1), consumption_HVGH, marker='<', label='HVGH',linewidth=3, markersize=10)
    plt.plot(np.arange(1,num_x_ticks+1,1), np.array(consumption_HDP_HSMM)+10, marker='h', label='HDP-HSMM',linewidth=3, markersize=10)

    plt.text(10.5, 450, 'Refused to work', fontsize=20)

    plt.xlabel('Dimension', size=20)
    plt.ylabel('Time Consumption (s)', size=20)
    plt.xticks(range(2,num_x_ticks+1,2),size=20)
    plt.yticks(size=20)
    plt.legend(ncol=1, fontsize=14, loc='upper left')
    plt.tight_layout()
    plt.show()

def total_time_consumption():
    plt.figure(figsize=(8, 3))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
    Time2Seg_  =  [200, 60, 20, 300,  300]
    Autoplait_ =  [0.2089, 0.4543, 0.8009, 0.1039, 0.0573]
    TICC_ =       [3755, 283.66, 125.93, 3061.67, 10333.20]
    HVGH_ =       [0.2579, 0.4171, 0.3628, 0.0573, 0.1361]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.15  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 1.5*width, Time2Seg_, width, label='Time2Seg',hatch='//.')
    plt.bar(x - 0.5*width, Autoplait_, width, label='AutoPlait',hatch='\\\\.')
    plt.bar(x + 0.5*width, TICC_, width, label='TICC', hatch='/.')
    plt.bar(x + 1.5*width, HVGH_, width, label='HVGH', hatch='\\.')
    plt.ylabel('Time Consumption (s)', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()

# effect_of_length()
effect_of_dimension()
# total_time_consumption()