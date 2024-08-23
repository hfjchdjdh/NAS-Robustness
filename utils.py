import copy
from nasbench import api
from nas_201_api import NASBench201API as API201
import numpy as np
import math
from torch import nn
import os
# import pygraphviz as pgv
import igraph

# basic matrix for nas_bench 201
BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]]

MAX_NUMBER = 15625
NULL = 'null'
CONV1X1 = 'nor_conv_1x1'
CONV3X3 = 'nor_conv_3x3'
AP3X3 = 'avg_pool_3x3'
def delete_useless_node(ops):
    # delete the skip connections nodes and the none nodes
    # output the pruned metrics
    # start to change matrix
    matrix = copy.deepcopy(BASIC_MATRIX)
    for i, op in enumerate(ops, start=1):
        m = []
        n = []

        if op == 'skip_connect':
            for m_index in range(8):
                ele = matrix[m_index][i]
                if ele == 1:
                    # set element to 0
                    matrix[m_index][i] = 0
                    m.append(m_index)

            for n_index in range(8):
                ele = matrix[i][n_index]
                if ele == 1:
                    # set element to 0
                    matrix[i][n_index] = 0
                    n.append(n_index)

            for m_index in m:
                for n_index in n:
                    matrix[m_index][n_index] = 1

        elif op == 'none':
            for m_index in range(8):
                matrix[m_index][i] = 0
            for n_index in range(8):
                matrix[i][n_index] = 0

    ops_copy = copy.deepcopy(ops)
    ops_copy.insert(0, 'input')
    ops_copy.append('output')

    # start pruning
    model_spec = api.ModelSpec(matrix=matrix, ops=ops_copy)
    return model_spec.matrix, model_spec.ops

def save_arch_str2op_list(save_arch_str):
    op_list = []
    save_arch_str_list = API201.str2lists(save_arch_str)
    op_list.append(save_arch_str_list[0][0][0])
    op_list.append(save_arch_str_list[1][0][0])
    op_list.append(save_arch_str_list[1][1][0])
    op_list.append(save_arch_str_list[2][0][0])
    op_list.append(save_arch_str_list[2][1][0])
    op_list.append(save_arch_str_list[2][2][0])
    return op_list

def padding_zeros(matrix, op_list):
    assert len(op_list) == len(matrix)
    padding_matrix = matrix
    len_operations = len(op_list)
    if not len_operations == 8:
        for j in range(len_operations, 8):
            op_list.insert(j - 1, NULL)
        adjecent_matrix = copy.deepcopy(matrix)
        padding_matrix = np.insert(adjecent_matrix, len_operations - 1, np.zeros([8 - len_operations, len_operations]),
                                   axis=0)
        padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([8, 8 - len_operations]), axis=1)

    return padding_matrix, op_list


class custom_DataParallel(nn.parallel.DataParallel):
# define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G)-1:
                scattered_G.append((mini_batch, ))
                mini_batch = []
        return tuple(scattered_G), tuple([{}]*len(scattered_G))


def is_same_DAG(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):
            return False
    return True


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label == 0:  
        label = 'input'
        color = 'skyblue'
    elif label == 1:
        label = 'output'
        color = 'pink'
    elif label == 2:
        label = 'conv3'
        color = 'yellow'
    elif label == 3:
        label = 'sep3'
        color = 'orange'
    elif label == 4:
        label = 'conv5'
        color = 'greenyellow'
    elif label == 5:
        label = 'sep5'
        color = 'seagreen3'
    elif label == 6:
        label = 'avg3'
        color = 'azure'
    elif label == 7:
        label = 'max3'
        color = 'beige'
    else:
        label = ''
        color = 'aliceblue'
    #label = f"{label}\n({node_id})"
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)









