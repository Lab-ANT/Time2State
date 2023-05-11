import abc

'''
Created by Chengyu on 2022/2/26.
'''

class BasicClusteringClass:
    def __init__(self, params):
        pass

    @abc.abstractmethod
    def fit(self, X):
        pass

class BasicEncoderClass:
    def __init__(self, params):
        self._set_parmas(params)

    @abc.abstractmethod
    def _set_parmas(self, params):
        pass

    @abc.abstractmethod
    def fit(self, X):
        pass

    @abc.abstractmethod
    def encode(self, X, win_size, step):
        pass