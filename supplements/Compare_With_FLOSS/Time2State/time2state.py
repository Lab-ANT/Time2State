'''
Created by Chengyu on 2022/2/26.
'''

import numpy as np
from TSpy.label import reorder_label
from TSpy.utils import calculate_scalar_velocity_list

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
        self.__smooth()
        return self

    def predict(self, X, win_size, step):
        self.__length = X.shape[0]
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        self.__smooth()
        return self

    def fit(self, X, win_size, step):
        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        self.__smooth()
        self.__calculate_velocity()
        self.use_cps()
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

    # def find_change_points_by_velocity(self):
    #     self.__velocity = calculate_scalar_velocity_list(self.__embeddings, interval=1)
    #     self.__acceleration = calculate_scalar_velocity_list(self.__velocity, interval=1)
    #     for i in range(10):
    #         self.__acceleration = calculate_scalar_velocity_list(self.__acceleration, interval=1)

    # def __calculate_change_points(self):
    #     change_points = []
    #     change_points.append(0)

    #     pre = self.__state_seq[0]
    #     for idx, e in enumerate(self.__state_seq):
    #         if e != pre:
    #             change_points.append(idx-1)
    #             pre = e
    #     length = len(self.__state_seq)
    #     change_points.append(length-1)
    #     return change_points

    # def __judge_trival_segs(self, cps):
    #     length = len(self.__state_seq)
    #     start = cps[0]
    #     for end in cps[1:]:
    #         if end-start < .01*length:
    #             return True
    #         start = end
    #     return False
        
    # def __merge_trival_segs(self, change_points):
    #     length = len(self.__state_seq)
    #     start = change_points[0]
    #     for end in change_points[1:]:
    #         if (end-start) < .01*length:
    #             mid = int((end+start)/2)
    #             self.__state_seq[start:mid] = self.__state_seq[start-2]
    #             self.__state_seq[mid-1:end] = self.__state_seq[end] 
    #         start = end

    # def __majority_voting_smooth(self):
    #     pass

    # def __simple_smooth(self):
    #     change_points = []
    #     change_points.append(0)

    #     pre = self.__state_seq[0]
    #     for idx, e in enumerate(self.__state_seq):
    #         if e != pre:
    #             change_points.append(idx-1)
    #             pre = e
    #     length = len(self.__state_seq)
    #     change_points.append(length-1)

    #     length = len(self.__state_seq)
    #     start = change_points[0]
    #     for end in change_points[1:]:
    #         if (end-start) < .02*length:
    #             mid = int((end+start)/2)
    #             self.__state_seq[start:mid] = self.__state_seq[start-2]
    #             self.__state_seq[mid-1:end] = self.__state_seq[end] 
    #         start = end

    def use_cps(self):
        cut_list = self.find_potentional_cp()
        self.__embedding_label = self.bucket(self.__embedding_label, cut_list)
    
    def find_potentional_cp(self):
        threshold = np.mean(self.__velocity)
        idx = self.__velocity>=threshold
        pre = idx[0]
        cut_list = []
        for i, e in enumerate(idx):
            if e == pre:
                continue
            else:
                cut_list.append(i)
                pre = e
        self.__change_points = cut_list
        return cut_list

    def bucket(self, X, cut_points):
        result = np.array(X.shape, dtype=int)
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
        return result

    def __smooth(self):
        return
        # self.__simple_smooth()
        # change_points = self.__calculate_change_points()
        # while self.__judge_trival_segs(change_points):
        #     self.__merge_trival_segs(change_points)
        #     change_points = self.__calculate_change_points()
        #     print(change_points)

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

    @property
    def velocity(self):
        return self.__velocity

    @property
    def change_points(self):
        return self.__change_points