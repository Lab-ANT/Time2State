import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def solar(dataset_name):
    data_path = os.path.join(os.path.dirname(__file__), '../data/multivariate-time-series-data/solar_raw/'+dataset_name)
    target_path = os.path.join(os.path.dirname(__file__), '../data/multivariate-time-series-data/solar/'+dataset_name)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    data = []
    fname_list = list(os.listdir(data_path))
    fname_list.sort()

    for fname in fname_list[50:60]:
        col = np.loadtxt(os.path.join(data_path, fname), skiprows=1, usecols=[1], delimiter=',')
        print(fname, col.shape)
        data.append(col)
        print(col)
    
    data = np.vstack(data)
    print(data.shape)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=10, sharex=True)

    for col, sub_ax in zip(data, ax):
        sub_ax.plot(col)
    plt.show()

solar('Alabama')

