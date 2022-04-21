import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io

def info(path):
    pass

def plot(path):
    data = scipy.io.loadmat('subject1_walk/data.mat')
    print(data['data'].shape)
    print(data['labels'].shape)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    for i in range(0,10):
        ax[0].plot(data['data'][:,i],label=str(i))
        ax[0].legend()
    # sub1.plot(data['data'][:,4])
    ax[1].plot(data['labels'])
    plt.show()

def prepare_data_for_effect_of_len(path):
    data = scipy.io.loadmat(path)
    data = data['data'][:,:4]
    data = np.concatenate([data for x in range(10)])
    for i in range(1,21):
        df = pd.DataFrame(data[:i*10000,:])
        df.to_csv('effect_of_len/walk'+str(i)+'.csv', header=None, index=None, sep=' ')

def convert(path):
    data = scipy.io.loadmat(path)
    print(data['data'].shape)
    print(data['labels'].shape)
    labels = data['labels']
    data = data['data']
    df = pd.DataFrame(data)
    df.loc[:,df.shape[1]-1] = labels
    print(df)
    # df.to_csv('subject1_walk/data_labeled.csv', index=None)

# def prepare_data_for_autoplait():
#     data = scipy.io.loadmat('subject1_walk/data.mat')
#     # print(data['data'].shape)
#     data = data['data']
#     for i in range(1,21):
#         df = pd.DataFrame(data[:,0:i]).round(2)
#         df.to_csv('data_for_autoplait/ActRecTut'+str(i)+'.csv', index=None, header=None, sep=' ')

# def prepare_data_for_autoplait():
#     data = scipy.io.loadmat('subject1_walk/data.mat')
#     # print(data['data'].shape)
#     data = data['data']
#     for i in range(1,21):
#         df = pd.DataFrame(data[:,0:i]).round(2)
#         df.to_csv('data_for_autoplait/ActRecTut'+str(i)+'.csv', index=None, header=None, sep=' ')

prepare_data_for_effect_of_len('subject1_walk/data.mat')