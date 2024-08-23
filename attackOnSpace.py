import base64

import dgl

from dataset import *
# from netattack.nettack.nettack.nettack import Nettack
from dgl.dataloading import GraphDataLoader
import numpy as np
import random
import torch
import os
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import network
import loss
import time
import torch.nn.functional as F
from scipy.stats import kendalltau
import yaml
from SearchSpaceAttack import *

import tensorflow as tf

from attackUtils import *

def MSE(predict, target):
    return F.mse_loss(predict, target)

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict


def image_classification_test(epoch, step, r = 0.8):
    model.train(False)
    Ktau = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for i, (g_norm_batch, g_reduc_batch, y_batch) in enumerate(tgt_dataloader_test):
            g_norm_batch = g_norm_batch.to(device)
            g_reduc_batch = g_reduc_batch.to(device)
            y_batch = y_batch.to(device)
            _, norm_outputs ,_,_  = model(g_norm_batch, g_norm_batch.ndata['attr'], bridge=config['net']["bridge"])
            _, reduc_outputs ,_,_  = model(g_reduc_batch, g_reduc_batch.ndata['attr'], bridge=config['net']["bridge"])
            all_output = (norm_outputs * r + reduc_outputs * (1-r))
            all_label = torch.tensor(y_batch).float()
            predict = all_output.squeeze(1).to(device)
            label = all_label
            accuracy = eval(config['train']['loss_function'])(predict, label)
            d = predict
            d_label = label
            predict = predict * std + mean
            label = label * std + mean
            denormal_accuracy = F.mse_loss(predict, label)
            Ktau = (kendalltau(predict.cpu().numpy(), label.cpu().numpy())[0])
            log_str = "{}: Epoch: {:d}, MSE: {:.5f}, demo_MSE: {:.5f}, mean KTau:{:.5f}".format(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()), epoch, accuracy, denormal_accuracy, Ktau)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
    return accuracy, Ktau




class NASBench101DatasetAfterPerturb(DGLDataset): #将Bench101数据转化为dgl库能处理的形式
    def __init__(self,NUM_EXAMPLES = 30000,all=False):
        self.NUM_EXAMPLES=NUM_EXAMPLES
        self.all = all
        super().__init__(name='NASBench')

    def process(self):
        print("Building dataset NASBench_101...")
        # NAS_Bench_101
        self.graphs = []
        self.labels = []
        g_train, g_test = load_preprocessed_data(adjs,ops,metrs)
        all_graphs = g_train + g_test
        for (g,y) in all_graphs:
            self.graphs.append(g)
            self.labels.append(y)
        # Convert the label list to tensor for saving.
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        #self.features = np.array(self.features, dtype=object)

    def getAdjsAndOps(self):
        return self.adjs,self.ops, self.metrs

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def load_preprocessed_data(adjs,ops,metrs):
    regurized = True
    g_list = []
    acc_list = []
    i = 0
    for adj,op,metrics in zip(adjs,ops,metrs):
        # print("adj:",adj)
        # print("op",op)
        # print("metrics",metrics)
        acc_l = []
        final_evaluation = metrics.evaluation_data[2]
        y = final_evaluation.test_accuracy
        acc_l.append(y)
        if i % 3 == 2:
                    # print(operations)
                mean_acc = np.mean(np.array(acc_l))
                acc_list.append(mean_acc)
                g = decode_NASBENCH_to_dgl(adj,op)
                g_list.append((g,mean_acc))

        i += 1
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
            
    random.Random(0).shuffle(g_list)

    ng = len(g_list)

    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):]




def train(config):
    best_ktau = 0.0
    weight_decay = config['train']['weight_decay']
    lr = config['train']['lr']
    iter_num = 0
    best_model_path = "results/trainmodel-2024-08-17-18-50-59/"
    for i in range(config['train']['epoch']):
        data_zip = enumerate(zip(src_dataloader_train, tgt_dataloader_train))
        for step, ((src_batch, src_acc_batch), tgt_batch) in data_zip:
            model.train(True)
            ad_net.train(True)

            # src_batch.add_edges(0, 99)
            src_attacker = NasBench101SearchSpaceAttack(src_batch, src_acc_batch, 0)
            src_batch = src_attacker.start_attacking_by_modify_edge()



            src_batch = src_batch.to(device)  # _X_obs
            src_acc_batch = src_acc_batch.to(device)  # _y_obs

            tgt_batch = tgt_batch.to(device)
            lr = lr * (1 + 0.001 * i) ** (-0.75)
            optimizer = optim.Adam(model.get_parameters() + ad_net.get_parameters(), lr=lr, weight_decay=weight_decay)
            optimizer.zero_grad()

            # 获取到对应的权重
            W1, W2 = model.get_weight()

            features_source, outputs_source, vec_source, focal_source = model(src_batch, src_batch.ndata['attr'],
                                                                              bridge=config['net']["bridge"])
            features_target, outputs_target, vec_target, focal_target = model(tgt_batch, tgt_batch.ndata['attr'],
                                                                              bridge=config['net']["bridge"])
            features = torch.cat((features_source, features_target), dim=0)
            vec = torch.cat((vec_source, vec_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            focals = torch.cat((focal_source, focal_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)


            transfer_loss, bridge = loss.my_loss(features, [softmax_out, focals], ad_net, config['device']['device'],
                                                 network.calc_coeff(i, max_iter=config['train']['epoch']))
            regression_loss = eval(config['train']['loss_function'])(outputs_source.squeeze(1),
                                                                     torch.tensor(src_acc_batch).to(
                                                                         config['device']['device']))
            total_loss = transfer_loss + config['train']['trade_off'] * regression_loss + config['net'][
                'bridge'] * bridge
            log_str = "{}: Epoch: {:d}, total_loss: {:05f}, transferloss: {:.5f}, regression_loss: {:.5f}, bridge:{:.5f}".format(
                time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()), i, total_loss, transfer_loss, regression_loss,
                bridge)

            print(log_str)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            total_loss.backward()
            optimizer.step()

            iter_num += 1
            acc, ktau = image_classification_test(i, step)

            best_ktau = ktau
            torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'ad_net_state_dict': ad_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_ktau': best_ktau
                }, best_model_path+f"best_model_{i}.pth.tar")


            print(f"Best model saved at epoch {i} with ktau {best_ktau}")


if __name__ == "__main__":

    train_data_before = NASBench101Dataset(1000)
    adjs, ops, metrs = train_data_before.adjs, train_data_before.ops, train_data_before.metrs
    for i,adj in enumerate(adjs):
        adj = adj + adj.T
        adj[adj >= 1] = 1
        adjs[i] = adj

    # print("ops_prepared_for_nas",ops)
    train_data_before = GraphDataLoader(train_data_before, batch_size=1000, drop_last=True)
    train_data_after = NASBench101DatasetAfterPerturb(1000)

    path = './config.yaml'
    config = read_yaml(path)
    cuda = config['device']['use_cuda'] and torch.cuda.is_available()
    # if cuda:
    #     device = torch.device("cuda:" + config['device']['gpu_id'])
    # else:
    #     device = torch.device("cpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config['device']['device'] = device

    config["output_path"] = os.path.join(os.path.dirname(os.path.realpath('__file__')), \
                                         'results\\trainmodel-{}'.format(
                                             time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    config["best_model"] = osp.join(config["output_path"], "best_model.pth.tar")


    #模型所在
    model = network.model(embedding_len=config['net']['embedding_len'], inter_l=config['encoder']['inter_l'],
                          feature_l=config['encoder']['feature_l'], use_embed=config['net']['use_embed'])
    model = model.to(device)
    ad_net = network.AdversarialNetwork(config['encoder']['feature_l'], config['net']['adnet_hidden_size'])
    ad_net = ad_net.to(device)
    
    #权重所在
    W1, W2 = model.get_weight()
    

    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    if config['data']['source_dataset_name'] == "NASBENCH101":
        train_data = NASBench101Dataset(config['data']["source_101_dataset_num"], all=config['data']["all_101"])
        adjs = train_data.adjs

    if config['data']['target_dataset_name'] == "DARTS":
        tgt_dataset = DartsDataset(NUM_EXAMPLES_Darts=config['data']["target_dataset_num"], labeled=False)
        DARTS_test = DartsDataset(NUM_EXAMPLES_Darts=config['data']["target_dataset_num"], labeled=True)

        _, _, mean, std = read_darts_dataset()

    src_dataloader_train = GraphDataLoader(train_data_after, batch_size=config['train']['train_batch_size'], drop_last=True)

    tgt_dataloader_train = GraphDataLoader(tgt_dataset, batch_size=config['train']['train_batch_size'], drop_last=True)
    tgt_dataloader_test = GraphDataLoader(DARTS_test, batch_size=config['train']['infer_batch_size'], drop_last=True)

    seed = config['train']['seed']
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config["out_file"].write(str(config))
    config["out_file"].flush()
    
    print(config)

    # 这里可以进行攻击操作
    net_attack = None

    # 这里可以进行攻击操作
    # for src_batch, src_acc_batch in src_dataloader_train:
    #     attacker =  NasBench101SearchSpaceAttack(src_batch, src_acc_batch, 0)
    #     print(f"graph before perturbation:{src_batch.adjacency_matrix().to_dense()[0][1]}")
    #     src_batch = attacker.start_attacking_by_modify_edge()
    #     print(f"graph after perturbation:{src_batch.adjacency_matrix().to_dense()[0][1]}")

    train(config)







