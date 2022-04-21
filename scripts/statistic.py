from unittest import skip
import numpy as np
import os

path = os.path.dirname(__file__)
data_path = os.path.join(path, 'result2.txt')

with open(data_path, 'r') as f:
    lines = f.readlines()
    sum = 0
    for line in lines:
        if line[0] == 'w':
            print(sum/10)
            sum = 0
        else:
            sum+=float(line[14:21])
