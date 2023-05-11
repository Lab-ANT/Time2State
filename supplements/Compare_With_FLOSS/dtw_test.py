# This script is used to validate if padding 0
# will affect the output of DTW.

from tslearn.metrics import dtw
import numpy as np
import matplotlib.pyplot as plt

ts1 = np.sin(np.linspace(1,6*np.pi,200))
ts2 = np.cos(np.linspace(1,6*np.pi,100))
pad = np.zeros((100,))
ts3 = np.concatenate([ts2, pad])

plt.plot(ts1)
plt.plot(ts3)
plt.plot(ts2)
plt.show()

dtw_score1 = dtw(ts1,ts2)
dtw_score2 = dtw(ts1,ts3)

print(dtw_score1, dtw_score2)