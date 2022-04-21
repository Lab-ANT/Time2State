# Created by Chengyu on 8/14/2021.
# Retrieve channels from the MoCap Dataset.
# MoCap is a motion capture dataset created by CMU.
# For more details please refer to http://mocap.cs.cmu.edu
#
# A data frame in MoCap is of the following form.

'''
root 9.37216 17.8693 -17.3198 -2.01677 -7.59696 -3.23164
lowerback 2.30193 -0.395121 1.17299
upperback 0.0030495 -0.462657 2.70388
thorax -1.27453 -0.231833 2.13151
lowerneck -9.32819 -3.76531 -6.70788
upperneck 27.8377 -3.2335 -3.01318
head 10.556 -2.55728 -0.318388
rclavicle 3.64024e-015 -6.75868e-015
rhumerus -29.5133 -11.7797 -80.4307
rradius 21.1829
rwrist -7.55893
rhand -17.4806 -21.0413
rfingers 7.12502
rthumb 8.77158 -50.8391
lclavicle 3.64024e-015 -6.75868e-015
lhumerus 17.2039 -14.515 62.7889
lradius 136.231
lwrist 10.1195
lhand -37.631 -17.4438
lfingers 7.12502
lthumb -10.6834 12.2646
rfemur -0.629535 4.65229 22.5467
rtibia 26.4457
rfoot -15.2124 -9.97437
rtoes 3.93605
lfemur 4.00236 1.20472 -13.8412
ltibia 20.088
lfoot -16.1868 6.57726
ltoes -4.61789
'''

# Specify the channel that you want to retrieve by the usecols parameter.
# e.g., usecols = ['lhumerus','rhumerus','lfemur','rfemur']
# The output is of csv format.

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd

def retrieve(in_path, out_path, usecols, show=False):
    cols = {}
    cols_numeric = {}
    for col_name in usecols:
        cols[col_name]=[]
    with open(in_path) as f:
        lines = f.readlines()
        for line in lines:
            for col_name in usecols:
                if re.match(col_name, line):
                    cols[col_name].append(line)
    for col_name in usecols:
        cols_numeric[col_name]=np.array([float(line.split(' ')[1]) for line in cols[col_name]])
    
    if show:
        for col_name in usecols:
            plt.plot(cols_numeric[col_name])
            plt.title(in_path+str(len(cols_numeric[usecols[1]])))
        plt.show()
    
    df_array = []
    for col_name in usecols:
        df_array.append(cols_numeric[col_name])
    df = pd.DataFrame(df_array).T.round(4)
    df.to_csv(out_path, header=None, index=None, sep=' ')

usecols = ['lhumerus','rhumerus','lfemur','rfemur']
out_dir = './4d/'
in_dir = './raw/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# for f in os.listdir(in_dir):
#     in_fpath = in_dir+f
#     out_fpath = out_dir+f[:-4]+'.4d'
#     print(in_fpath)
#     retrieve(in_fpath, out_fpath, usecols, show=True)

retrieve('./raw/amc_86_10.txt', './4d/amc_02_01.4d', usecols, show=True)