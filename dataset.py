from cProfile import label
from nasbench.lib import model_metrics_pb2
import igraph
import tensorflow as tf
import json
import base64
import numpy as np
import random
import pickle
import argparse
from utils import *
import torch
# from torch.utils.data import Dataset
from dgl.data import DGLDataset
import dgl
from scipy import sparse



NAS_BENCH_101 = "dataset/nasbench_only108.tfrecord"
DARTS = "dataset/darts_dataset.pth.tar"
NASBENCH_101_dict_op= {"input": 0, "output": 1, "conv1x1-bn-relu": 2, "conv3x3-bn-relu": 3, "maxpool3x3": 4}

def load_nasbench101_graphs(num_data,n_types=3, fmt="dgl", all=False, regurized=True, rand_seed=0, graph_args=argparse.ArgumentParser().parse_known_args()[0]):
        #load NASBENCH format NNs to igraphs or tensors
    g_list = []
    max_n = 7
    i = 0
    acc_list = []
    adjs = []
    ops = []
    metrs = []
    for serialized_row in tf.compat.v1.python_io.tf_record_iterator(NAS_BENCH_101):
        # print(serialized_row)
        acc_l = []
        module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (
                json.loads(serialized_row.decode('utf-8')))
        dim = int(np.sqrt(len(raw_adjacency)))
        if(dim!=7):
            continue
        adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
        adjacency = np.reshape(adjacency, (dim, dim))
        adjs.append(adjacency)
        operations = raw_operations.split(',')

        ops.append(operations)

        metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))

        metrs.append(metrics)

        final_evaluation = metrics.evaluation_data[2]
        y = final_evaluation.test_accuracy
        acc_l.append(y)
        if i % 3 == 2:
            # print(operations)
            mean_acc = np.mean(np.array(acc_l))
            acc_list.append(mean_acc)
            if(fmt == 'igraph'):
                g = decode_NASBENCH_to_igraph(adjacency,operations)
            elif(fmt == 'dgl'):
                g = decode_NASBENCH_to_dgl(adjacency,operations)
            g_list.append((g,mean_acc))
            acc_l = []
        i+=1
        if i>int(num_data *3) and all==False:
            break
    if regurized:
        # mean_value = 0.902434
        # std_value = 0.058647
        acc_list = np.array(acc_list)
        mean_value = np.mean(acc_list)
        std_value = np.std(acc_list)
        g_list_copy = []
        for g,y in g_list:
            g_list_copy.append((g,(y - mean_value) / std_value))
        g_list = g_list_copy
        del g_list_copy
    graph_args.num_vertex_type = 5
    graph_args.max_n = max_n
    graph_args.START_TYPE = 0
    graph_args.END_TYPE = 1
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    # adjacency：需要返回，从而发动攻击
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args, adjs,ops, metrs





def denormalize_nasbench101(x):
    mean_value = 0.902434
    std_value = 0.058647
    return x * std_value + mean_value

def decode_NASBENCH_to_dgl(adjacency,operations, dataset="101"):
    MAX_FEATURES_Darts = 6
    if dataset == "101":
        NASBENCH_dict_op = NASBENCH_101_dict_op
    # print(adjacency, operations)
    nonzero = sparse.coo_matrix(adjacency).nonzero()
    src = nonzero[0].tolist()
    dst = nonzero[1].tolist()
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    g = dgl.graph((src, dst))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    num_verticles= adjacency.shape[0]
    g_features = np.zeros((num_verticles, MAX_FEATURES_Darts), dtype=float)
    #g_features[-1, :] = dict_feat['global']
    for i,op in enumerate(operations):
        # print("op:",op)
        g_features[i, NASBENCH_dict_op[op]] = 1
    # print(g_features)
    g.ndata['attr'] = torch.tensor(g_features, dtype=torch.float32)
    return g
    
    


def decode_NASBENCH_to_igraph(adjacency,operations, dataset="101"):
    #convert NASBENCH adjacency matrix to ENAS format which is list of lists
    if dataset == "101":
        NASBENCH_dict_op = NASBENCH_101_dict_op
    g = igraph.Graph(directed=True)
    g.add_vertices(len(operations))
    for i,op in enumerate(operations):
        g.vs[i]['type'] = NASBENCH_dict_op[op]
    for i, node in enumerate(adjacency):
        for j, edge in enumerate(node):
            if(edge==1):
                g.add_edge(i,j)
    return g

def decode_DARTS_to_igraph(adjacency,operations):
    #convert NASBENCH adjacency matrix to ENAS format which is list of lists
    g = igraph.Graph(directed=True)
    g.add_vertices(len(operations))
    for i,op in enumerate(operations):
        g.vs[i]['type'] = op
    for i, node in enumerate(adjacency):
        for j,edge in enumerate(node):
            if(edge==1):
                g.add_edge(i,j)
    return g

def decode_DARTS_to_dgl(adjacency,operations):
    MAX_FEATURES_Darts = 6
    nonzero = sparse.coo_matrix(adjacency).nonzero()
    src = nonzero[0].tolist()
    dst = nonzero[1].tolist()
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    g = dgl.graph((src, dst))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    num_verticles= adjacency.shape[0]
    g_features = np.zeros((num_verticles, MAX_FEATURES_Darts), dtype=float)
    #g_features[-1, :] = dict_feat['global']
    for i,op in enumerate(operations):
        g_features[i, op] = 1
    g.ndata['attr'] = torch.tensor(g_features, dtype=torch.float32)
    return g
    


class ArchDarts:
    def __init__(self, arch):
        self.arch = arch

    @classmethod
    def random_arch(cls):
        # output a uniformly random architecture spec
        # from the DARTS repository
        # https://github.com/quark0/darts
        NUM_VERTICES = 4
        OPS = ['none',
               'sep_conv_3x3',
               'dil_conv_3x3',
               'sep_conv_5x5',
               'dil_conv_5x5',
               'max_pool_3x3',
               'avg_pool_3x3',
               'skip_connect'
               ]
        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(1, len(OPS)), NUM_VERTICES)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
        return (normal, reduction)


class DataSetDarts:
    def __init__(self, dataset_num=int(1e6), dataset=None):
        self.dataset = 'darts'
        self.INPUT_1 = 'c_k-2'  # num 0
        self.INPUT_2 = 'c_k-1'  # num 1
        self.BASIC_MATRIX = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],  #5
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  #6
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  #7
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #8
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  #9
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  #10
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #11
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  #12
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #13
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #14
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # a mapping between genotype and op_list
        self.mapping_intermediate_node_ops = [{'input_0': 1, 'input_1': 5}, #2
                                              {'input_0': 2, 'input_1': 6, 2: 9}, #3
                                              {'input_0': 3, 'input_1': 7, 2: 10, 3: 12}, # 4
                                              {'input_0': 4, 'input_1': 8, 2: 11, 3: 13, 4: 14}] #5
        self.op_integer = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: -1}
        if dataset is not None:
            self.dataset = dataset
            print('Generate DARTS dataset, the size is :{}'.format(dataset_num))
        else:
            if dataset_num > 0:
                self.dataset = self.generate_random_dataset(dataset_num)
                print('Generate DARTS dataset, the size is :{}'.format(dataset_num))

    def generate_random_dataset(self, num):
        """
        create a dataset of randomly sampled architectures where may exist duplicates
        """
        data = []
        while len(data) < num:
            archtuple = ArchDarts.random_arch()
            data.append(archtuple)
        return data

    def get_ops(self, cell_tuple):
        all_ops = []
        mapping = self.mapping_intermediate_node_ops
        # assign op list
        # initial ops are all zeros, i.e. all types are None
        ops = np.zeros(16, dtype='int8')
        # 'input' -2, 'output' -3
        input_output_integer = {'input': -2, 'output': -3}
        ops[0], ops[-1] = input_output_integer['input'], input_output_integer['output']
        for position, op in enumerate(cell_tuple):
            intermediate_node = position // 2 
            prev_node = op[0]
            if prev_node == 0:
                prev_node = 'input_0'
            elif prev_node == 1:
                prev_node = 'input_1'

            # determine the position in the ops
            ops_position = mapping[intermediate_node][prev_node]
            op_type = op[1]
            ops[ops_position] = op_type
        return ops

    def delete_useless_nodes(self, cell_tuple):
        '''
        This function would not change the op integers (1-6)
        The skip connection is 7, the none is 0
        '''
        ops= self.get_ops(cell_tuple)

        BASICMATRIX_LENGTH = len(self.BASIC_MATRIX)
        matrix = copy.deepcopy(self.BASIC_MATRIX)
        for i, op in enumerate(ops):
            if op == 7:  # skip connection
                m, n = [], []
                for m_index in range(BASICMATRIX_LENGTH):
                    ele = matrix[m_index][i]
                    if ele == 1:
                        # set element to 0
                        matrix[m_index][i] = 0
                        m.append(m_index)

                for n_index in range(BASICMATRIX_LENGTH):
                    ele = matrix[i][n_index]
                    if ele == 1:
                        # set element to 0
                        matrix[i][n_index] = 0
                        n.append(n_index)

                for m_index in m:
                    for n_index in n:
                        matrix[m_index][n_index] = 1

            elif op == 0:  # none op type
                for m_index in range(BASICMATRIX_LENGTH):
                    matrix[m_index][i] = 0
                for n_index in range(BASICMATRIX_LENGTH):
                    matrix[i][n_index] = 0

            # start pruning
        model_spec = api.ModelSpec(matrix=matrix, ops=list(ops))
        return model_spec.matrix, model_spec.ops

    def transfer_ops(self, ops):
        '''
        op_dict = {
                0: 'none',
                1: 'sep_conv_5x5',
                2: 'dil_conv_5x5',
                3: 'sep_conv_3x3',
                4: 'dil_conv_3x3',
                5: 'max_pool_3x3',
                6: 'avg_pool_3x3',
                7: 'skip_connect'
            }
        transfer_ops:
        op_dict = {
                0: 'input',
                1: 'output',
                2: 'conv_1x1',
                3: 'conv_3x3',
                4: 'pool',
                5: 'conv5x5',
            }
        '''
        trans_op = copy.deepcopy(ops)
        for index, op_value in enumerate(trans_op):
            if op_value == -2:
                trans_op[index] = 0
            elif op_value == -3:
                trans_op[index] = 1
            elif op_value == 3 or op_value == 4:
                trans_op[index] = 3
            elif op_value == 5 or op_value == 6:
                trans_op[index] = 4
            elif op_value == 1 or op_value == 2:
                trans_op[index] = 5
            else:
                raise ValueError("Error: unknown ops: %d"%(op_value))
        return trans_op


    def load_DARTS_graphs(self, transfer_ops=True, type="dgl"):
        g_list = []
        for index, tuple_arch in enumerate(self.dataset):
            norm_matrixes, norm_ops = self.delete_useless_nodes(tuple_arch[0])
            reduc_matrixes, reduc_ops = self.delete_useless_nodes(tuple_arch[1])
            if transfer_ops:
                norm_ops = self.transfer_ops(norm_ops)
                reduc_ops = self.transfer_ops(reduc_ops)
            norm_adj = np.array(norm_matrixes, dtype=np.int8)
            reduc_adj = np.array(reduc_matrixes, dtype=np.int8)
            if type == "igraph":
                norm_g = decode_DARTS_to_igraph(norm_adj, norm_ops)
                reduc_g = decode_DARTS_to_igraph(reduc_adj, reduc_ops)
            elif type == "dgl":
                norm_g = decode_DARTS_to_dgl(norm_adj, norm_ops)
                reduc_g = decode_DARTS_to_dgl(reduc_adj, reduc_ops)
            g_list.append(norm_g)
            g_list.append(reduc_g)
        return g_list

    def transfer_tuple_to_graph(self, tuple, transfer_ops=True, type = 'dgl'):
        matrixes, ops = self.delete_useless_nodes(tuple)
        if transfer_ops:
            ops = self.transfer_ops(ops)
        adj = np.array(matrixes, dtype=np.int8)
        # print(adj)
        # print("len:"+str(len(ops)))
        if type == "igraph":
            g = decode_DARTS_to_igraph(adj, ops)
        elif type == "dgl":
            g = decode_DARTS_to_dgl(adj, ops)

        return g


def read_darts_dataset(regurized=True):
    data = torch.load(DARTS)
    dataset = data['dataset']
    d = DataSetDarts(0)
    DARTS_norm_g_y = []
    DARTS_reduc_g_y = []
    acc = data['best_acc_list']
    acc_1 = []
    for i in acc:
        acc_1.append(i/100.)
    acc_np = np.array(acc_1)
    mean = np.mean(acc_np)
    std = np.std(acc_np)

    for index, (norm_tuple, reduc_tuple) in enumerate(dataset):
        norm_tuple_g = d.transfer_tuple_to_graph(norm_tuple)
        reduc_tuple_g = d.transfer_tuple_to_graph(reduc_tuple)
        acc_value = acc_1[index]
        if regurized:
            acc_value = (acc_value - mean) / std
        DARTS_norm_g_y.append((norm_tuple_g, acc_value))
        DARTS_reduc_g_y.append((reduc_tuple_g, acc_value))
    return DARTS_norm_g_y, DARTS_reduc_g_y, mean, std


class NASBench101Dataset(DGLDataset): #将Bench101数据转化为dgl库能处理的形式
    def __init__(self,NUM_EXAMPLES = 30000, all=False):
        self.NUM_EXAMPLES=NUM_EXAMPLES
        self.all = all
        super().__init__(name='NASBench')

    def process(self):
        print("Building dataset NASBench_101...")
        # NAS_Bench_101
        self.graphs = []
        self.labels = []
        g_train, g_test, _, self.adjs ,self.ops, self.metrs = load_nasbench101_graphs(self.NUM_EXAMPLES, all=self.all)
        all_graphs = g_train + g_test
        for (g,y) in all_graphs:
            self.graphs.append(g)
            self.labels.append(y)
        # Convert the label list to tensor for saving.
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        #self.features = np.array(self.features, dtype=object)
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    def __len__(self):
        return len(self.graphs)



class DartsDataset(DGLDataset):
    def __init__(self,NUM_EXAMPLES_Darts = 30000,labeled=False):
        self.labeled=labeled
        if labeled:
            self.NUM_EXAMPLES_Darts=100
            print("Labeled data...")
        else:
            self.NUM_EXAMPLES_Darts = NUM_EXAMPLES_Darts
        super().__init__(name='Darts')

    def process(self):
        print("Building dataset Darts...")
        self.norm_graphs = []
        self.reduc_graphs = []
        self.norm_labels = []
        self.reduc_labels = []
        self.graphs = []
        if self.labeled:
            norm_DARTS_test, reduc_DARTS_test, mean, std = read_darts_dataset()
            for (g,y) in norm_DARTS_test:
                self.norm_graphs.append(g)
                self.norm_labels.append(y)
            self.norm_labels = torch.tensor(self.norm_labels, dtype=torch.float32)
            for (g,y) in reduc_DARTS_test:
                self.reduc_graphs.append(g)
                self.reduc_labels.append(y)
            self.reduc_labels = torch.tensor(self.reduc_labels, dtype=torch.float32)
        else:
            self.graphs = DataSetDarts(int(self.NUM_EXAMPLES_Darts/2)).load_DARTS_graphs()

    def __getitem__(self, i):
        if self.labeled:
            return self.norm_graphs[i], self.reduc_graphs[i] , self.norm_labels[i]
        else:
            return self.graphs[i]

    def __len__(self):
        if self.labeled:
            return len(self.norm_graphs)
        else:
            return len(self.graphs)
