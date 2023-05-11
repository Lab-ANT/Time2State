import os
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Find cut point from a state sequence.
# The state sequence can be predicted or ground truth.
def find_cp_from_state_seq(X):
    pre = X[0]
    cp_list = []
    for i, e in enumerate(X):
        if e == pre:
            continue
        else:
            cp_list.append(i)
            pre = e
    return cp_list

def txt_filter(file_list):
    filtered_list = []
    for file_name in file_list:
        if '.txt' in file_name:
            filtered_list.append(file_name)
    return filtered_list

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def padding(X, max_length):
    if len(X.shape) == 1:
        dim = 1
        X = np.expand_dims(X, axis=1)
    else:
        _, dim = X.shape
    data = np.zeros((max_length, dim))
    length = len(X)
    data[:length] = X
    return data

def padding_and_stack(seg_list):
    max_length = 0
    for seg in seg_list:
        length = len(seg)
        if length > max_length:
            max_length = length
    new_seg_list = []
    for seg in seg_list:
        new_seg_list.append(padding(seg, max_length))
    result = np.stack(new_seg_list)
    return result

def calculate_seg_len_list(found_cps,length):
    length_list = []
    prev = 0
    for cp in found_cps:
        length_list.append(int(cp-prev))
        prev=cp
    length_list.append(length-prev)
    return length_list

def cluster_segs(X, found_cps, n_states, metric='euclidean'):
    seg_list = []
    length = len(X)
    start = 0
    for cp in found_cps:
        seg_list.append(X[start:cp])
        start = cp
    seg_list.append(X[start:length])
    segments = padding_and_stack(seg_list)
    # Using TSKmeans
    # dtw, euclidean, softdtw
    segments = TimeSeriesScalerMeanVariance().fit_transform(segments)
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_states,metric=metric).fit(segments)
    labels = ts_kmeans.labels_

    # Using KShape
    # KShape seems does not work, the following is the error information.
    # Resumed because of empty cluster
    # For this method to operate properly, prior scaling is required
    # segments = TimeSeriesScalerMeanVariance().fit_transform(segments)

    # kShape clustering
    # ks = KShape(n_clusters=n_states, verbose=True)
    # labels = ks.fit_predict(segments)

    seg_label_list = []
    length_list = calculate_seg_len_list(found_cps,length)

    for label, length in zip(labels, length_list):
        seg_label_list.append(label*np.ones(length))
    result = np.hstack(seg_label_list)
    return result

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data