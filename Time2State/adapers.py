'''
Created by Chengyu on 2022/2/26.
'''

import numpy as np
import sys
import os
from TSpy.utils import all_normalize
sys.path.append(os.path.dirname(__file__))
import encoders
from Time2State.abstractions import *
import tnc
import cpc

class LSTM_LSE_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        self.encoder = encoders.LSTM_LSE(**self.hyperparameters)

    def fit(self, X):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        self.encoder.fit(X, save_memory=True, verbose=False)

    def encode(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings

class CausalConv_CPC_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        win_size = params['win_size']
        in_channels = params['in_channels']
        out_channels = params['out_channels']
        self.nb_steps = params['nb_steps']
        self.encoder = cpc.CausalConv_CPC(win_size, out_channels, in_channels)
        # self.encoder = tnc.CausalConv_TNC(window_size=win_size, out_channels=out_channels, in_channels=in_channels)

    def fit(self, X):
        self.encoder.fit_encoder(X, self.nb_steps)

    def encode(self, X, win_size, step):
        return self.encoder.encode(X, win_size, step)

class CausalConv_TNC_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        win_size = params['win_size']
        in_channels = params['in_channels']
        out_channels = params['out_channels']
        self.nb_steps = params['nb_steps']
        self.encoder = tnc.CausalConv_TNC(window_size=win_size, out_channels=out_channels, in_channels=in_channels)

    def fit(self, X):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        self.encoder.fit_encoder(X, self.nb_steps, 0.05)

    def encode(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        return self.encoder.encode(X, win_size=win_size, step=step)

class CausalConv_LSE_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        self.encoder = encoders.CausalConv_LSE(**self.hyperparameters)

    def fit(self, X):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        self.encoder.fit(X, save_memory=True, verbose=False)
        # self.loss_list = self.encoder.loss_list

    def encode(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings

class CausalConv_Triplet_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        self.encoder = encoders.CausalConv_Triplet(**self.hyperparameters)

    def fit(self, X):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        self.encoder.fit(X, save_memory=True, verbose=False)
        self.loss_list = self.encoder.loss_list

    def encode(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings