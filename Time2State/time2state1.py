'''
Created by Chengyu on 2022/2/26.
'''

import numpy as np
from TSpy.label import reorder_label
from TSpy.utils import calculate_scalar_velocity_list

class Time2State:
    def __init__(self, win_size, step, encoder, clustering_component, verbose=False):
        """
        Initialize Time2State.

        Parameters
        ----------
        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        encoder_class : object.
            The instance of encoder.

        clustering_class: object.
            The instance of clustering component.
        """

        # The window size must be an even number.
        if win_size%2 != 0:
            raise ValueError('Window size must be even.')

        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size/2)
        self.__encoder = encoder
        self.__clustering_component = clustering_component

    def fit(self, X, win_size, step):
        """
        Fit time2state.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted time2state.
        """
        
        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        # self.__use_cps()
        return self

    def predict(self, X, win_size, step):
        """
        Find state sequence for X.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted time2state.
        """
        self.__length = X.shape[0]
        self.__step = step
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        # self.__use_cps()
        return self

    def set_step(self, step):
        self.__step = step

    def set_clustering_component(self, clustering_obj):
        self.__clustering_component = clustering_obj
        return self

    def fit_encoder(self, X):
        self.__encoder.fit(X)
        return self

    def predict_without_encode(self, X, win_size, step):
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

    def __calculate_velocity(self):
        self.__velocity = calculate_scalar_velocity_list(self.__embeddings, interval=1)

    def map_cut_list(self, X):
        return np.array(X, dtype=int)*self.__step

    # def __use_cps(self):
    #     self.__calculate_velocity()
    #     cut_list = self.__find_potential_cp()
    #     self.__embedding_label = self.bucket(self.__embedding_label, cut_list)
    def __use_cps(self):
        self.__calculate_velocity()
        cut_list = self.__find_potential_cp()
        cut_list = self.map_cut_list(cut_list)+self.__offset
        self.__state_seq = self.bucket(self.__state_seq, cut_list)
    
    def __find_potential_cp(self):
        # # threshold = np.mean(self.__velocity)*3
        # threshold = np.percentile(self.__velocity, 95)
        # idx = self.__velocity>=threshold
        # pre = idx[0]
        # cut_list = []
        # for i, e in enumerate(idx):
        #     if e == pre:
        #         continue
        #     else:
        #         cut_list.append(i)
        #         pre = e
        # self.__change_points = cut_list
        # return cut_list

        # threshold = np.mean(self.__velocity)*3
        threshold = np.percentile(self.__velocity, 95)
        idx = np.argwhere(self.__velocity>=threshold)
        return [e[0] for e in idx]

    def bucket(self, X, cut_points):
        result = np.zeros(X.shape, dtype=int)
        print(len(cut_points), cut_points)
        pre = cut_points[0]
        for cut in cut_points[1:]:
            sub_seq = X[pre:cut]
            label_set = list(set(sub_seq))
            vote_list = []
            for label in label_set:
                vote_list.append(len(np.argwhere(sub_seq==label)))
            max_idx = np.argmax(vote_list)
            # print(max_idx, len(vote_list), label_set)
            result[pre:cut]=label_set[max_idx]
            pre = cut
        print(result.shape, X.shape, len(cut_points), set(result))
        return reorder_label(result)

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

    @property
    def velocity(self):
        return self.__velocity

    @property
    def change_points(self):
        return self.__change_points