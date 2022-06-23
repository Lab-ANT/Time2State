from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
import numpy as np

X = random_walks(n_ts=50, sz=32, d=1)
print(X.shape)

data1 = np.array([1,2,3,4,5,6])
data2 = np.array([1,2,3,4,5,6])
data3 = np.array([1,2,3,4,5,6])
data4 = np.array([1,2,3,4,5,6])
data5 = np.array([1,2,3,4,5,6])
data6 = np.array([1,2,3,4,5,6])

# def padding(X, max_length):
#     data = np.zeros((max_length))
#     length = len(X)
#     data[:length] = X
#     return data

# print(padding(data1, 20))

data1 = np.ones((100,))
data = np.hstack([data1, data1, data1, data1])
print(data.shape)

# data = np.array([data1,data2,data3,data4,data5,data6])

# km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5,
#                           max_iter_barycenter=5,
#                           random_state=0).fit(np.squeeze(X))
# print(km_dba.labels_, km_dba.labels_.shape)