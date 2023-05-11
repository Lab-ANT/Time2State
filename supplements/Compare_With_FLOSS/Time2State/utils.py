# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import torch.utils.data
from statsmodels.tsa.stattools import adfuller
import math

class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    @param dataset Numpy array representing the dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


class LabelledDataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset and its associated labels.

    @param dataset Numpy array representing the dataset.
    @param labels One-dimensional array of the same length as dataset with
           non-negative int values.
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

class TNCDataset(torch.utils.data.Dataset):
    def __init__(self, x, mc_sample_size, window_size, augmentation, epsilon=3, state=None, adf=False):
        super(TNCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.window_size = window_size
        self.sliding_gap = int(window_size*25.2)
        self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)

        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size,4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        # print(f, max(0,t - w_t), min(x.shape[-1], t + w_t))
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n
