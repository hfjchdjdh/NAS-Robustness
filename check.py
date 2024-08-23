import numpy as np
from dataset import load_nasbench101_graphs


_,_,_,adjs = load_nasbench101_graphs(30000)
print(adjs)