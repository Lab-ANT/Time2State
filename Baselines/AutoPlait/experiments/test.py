# Created by Chengyu on 2021/5/9
# This is a test script.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Decompose state sequence into single state sequences.
# Input: state sequence
# Output: list of single state sequences
def decompose_state_sequence(state_seq):
    result = []
    # state_seq = np.array(state_seq)
    seq_len = len(state_seq)
    state_num = len(set(state_seq))
    for index in range(state_num):
        single_state_seq = np.zeros(seq_len)
        single_state_seq[np.argwhere(state_seq==index)] = 1
        result.append(single_state_seq)
    return result

# JMeasure
def JMeasure(seq1, seq2, lag):
    result_NMI=metrics.normalized_mutual_info_score(seq1, seq2)
    return result_NMI

# positive JMeasure
def positive_JMeasure(seq1, seq2, lag):
    pass

s1 = [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,2,2,2,1,1,1,2,2,2,0,0,0]
s1 = np.array(s1)

result = decompose_state_sequence(s1)

for i in range(len(result)):
    plt.step(np.arange(len(result[i])), result[i]+i*2)
    pass
plt.step(np.arange(len(result[i])), s1+3*2)
plt.show()

# print(JMeasure(result[1],result[1]))

# print(JMeasure([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0],[0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],0))