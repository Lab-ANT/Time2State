'''
Created by Chengyu on 2022/2/26.
'''

import numpy as np
from TSpy.label import reorder_label

class Time2State:
    def __init__(self, win_size, step, encoder_class, clustering_class, verbose=False):
        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size/2)
        self.__encoder = encoder_class
        self.__clustering_component = clustering_class

    def set_step(self, step):
        self.__step = step

    def set_clustering_component(self, clustering_obj):
        self.__clustering_component = clustering_obj
        return self

    def fit_encoder(self, X):
        self.__encoder.fit(X)
        return self
        # self.loss_list = self.__encoder.loss_list

    def predict_without_encode(self, X, win_size, step):
        self.__cluster()
        self.__assign_label()
        return self

    def predict(self, X, win_size, step):
        self.__length = X.shape[0]
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def fit(self, X, win_size, step):
        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def __encode(self, X, win_size, step):
        self.__embeddings = self.__encoder.encode(X, win_size, step)

    def __cluster(self):
        self.__embedding_label = reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __assign_label(self):
        hight = len(set(self.__embedding_label))
        # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
        weight_vector = np.ones(shape=(2*self.__offset)).flatten()
        self.__state_seq = self.__embedding_label
        vote_matrix = np.zeros((self.__length,hight))
        i = 0
        for l in self.__embedding_label:
            vote_matrix[i:i+self.__win_size,l]+= weight_vector
            i+=self.__step
        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])

    # def __smooth(self):
    #     self.__state_seq

    # def __assign_label(self):
    #     hight = len(set(self.__embedding_label))
    #     weight_vector = np.ones(shape=(2*self.__offset)).flatten()
    #     self.__state_seq = self.__embedding_label
    #     fake_len = (len(self.__embedding_label)-1)*self.__step+self.__win_size
    #     vote_matrix = np.zeros((fake_len,hight))
    #     i = 0
    #     for l in self.__embedding_label:
    #         vote_matrix[i:i+self.__win_size,l]+= weight_vector
    #         i+=self.__step
    #     self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])[:self.__length]

    def save_encoder(self):
        pass

    def load_encoder(self):
        pass

    def save_result(self, path):
        pass

    def load_result(self, path):
        pass

    def plot(self, path):
        pass

    @property
    def embeddings(self):
        return self.__embeddings

    @property
    def state_seq(self):
        return self.__state_seq
    
    @property
    def embedding_label(self):
        return self.__embedding_label