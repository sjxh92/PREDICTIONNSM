import numpy as np
import math
import pandas as pd
import networkx as nx
import torch
from torch.nn import functional as F
from itertools import count

n = 5  #类别数
indices = torch.randint(0, n, size=(3,3))  #生成数组元素0~5的二维数组（15*15）
print(indices)
one_hot = torch.nn.functional.one_hot(indices, n)  #size=(15, 15, n)
print(one_hot)

for i in count(1):
    print(i)

    if i > 1000:
        break
for t in range(1):
    print(t)

l = [1,2,3,4,5,6,7,8]
l = np.array(l)
print(l)
a = np.amax(l)
print(a)

a = []
a.append(1)
a.append(1)
a.append(1)
print(a)