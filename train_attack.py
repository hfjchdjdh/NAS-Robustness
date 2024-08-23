import os
import os.path as osp
from dgl.dataloading import GraphDataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import random
import time
from dataset import *
import torch.nn.functional as F
from scipy.stats import kendalltau
import yaml
from matplotlib import pyplot as plt 
plt.switch_backend('agg')

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

def MSE(predict, target):
    return F.mse_loss(predict, target)

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

def test(model, archtuple, config, r = 0.8):
    norm_tuple, reduc_tuple = archtuple
    d = DataSetDarts(0)
    norm_tuple_g = d.transfer_tuple_to_graph(norm_tuple)
    reduc_tuple_g = d.transfer_tuple_to_graph(reduc_tuple)
    g_tup = [norm_tuple_g, reduc_tuple_g]
    dataloader = GraphDataLoader(g_tup, batch_size= len(g_tup))
    for i, g_batch in enumerate(dataloader):
        g_batch = g_batch.to("cuda:1")
        _, outputs ,_  = model(g_batch, g_batch.ndata['attr'], bridge=config['net']["bridge"])
    outputs = outputs.squeeze(1)
    DARTS_test, _, config['DARTS_mean'], config['DARTS_std'] = read_darts_dataset()
    outputs = outputs * config['DARTS_std'] + config['DARTS_mean']
    return torch.mean(outputs)

def test_batch(model, tuples, config, device, r = 0.3):
    d = DataSetDarts(0)
    norm_gs, reduc_gs = [], []
    for (i, j) in tuples:
        norm_gs.append(d.transfer_tuple_to_graph(i))
        reduc_gs.append(d.transfer_tuple_to_graph(j))
    norm_dataloader = GraphDataLoader(norm_gs, batch_size= len(norm_gs))
    reduc_dataloader = GraphDataLoader(reduc_gs, batch_size= len(reduc_gs))
    for i, (norm_g_batch, reduc_g_batch) in enumerate(zip(norm_dataloader, reduc_dataloader)):
        norm_g_batch = norm_g_batch.to(device)
        reduc_g_batch = reduc_g_batch.to(device)
        _, norm_outputs,_ ,_  = model(norm_g_batch, norm_g_batch.ndata['attr'], bridge=config['net']["bridge"])
        _, reduc_outputs,_ ,_  = model(reduc_g_batch, reduc_g_batch.ndata['attr'], bridge=config['net']["bridge"])
    norm_outputs = norm_outputs.squeeze(1)
    reduc_outputs = reduc_outputs.squeeze(1)
    _, _, config['DARTS_mean'], config['DARTS_std'] = read_darts_dataset()
    norm_outputs = norm_outputs * config['DARTS_std'] + config['DARTS_mean']
    reduc_outputs = reduc_outputs * config['DARTS_std'] + config['DARTS_mean']
    acc = (norm_outputs * r + reduc_outputs * (1-r))
    idx = torch.argmax(acc)
    return torch.max(acc), tuples[idx]

def train(config):
    best_ktau = 0.0
    weight_decay = config['train']['weight_decay']
    lr = config['train']['lr']
    iter_num = 0

    

    for i in range(config['train']['epoch']):
        data_zip = enumerate(zip(src_dataloader_train, tgt_dataloader_train))
        for step, ((src_batch, src_acc_batch), tgt_batch) in data_zip:
            model.train(True)
            ad_net.train(True)
            src_batch = src_batch.to(device)
            src_acc_batch = src_acc_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            lr = lr * (1 + 0.001 * i) ** (-0.75)
            optimizer = optim.Adam(model.get_parameters() + ad_net.get_parameters(), lr=lr,weight_decay=weight_decay)
            optimizer.zero_grad()
            features_source, outputs_source, vec_source, focal_source = model(src_batch, src_batch.ndata['attr'], bridge=config['net']["bridge"])# 打印出来这些变量
            features_target, outputs_target, vec_target, focal_target = model(tgt_batch, tgt_batch.ndata['attr'],  bridge=config['net']["bridge"])

            print(f"src_input_ndata:{src_batch.ndata['h'].shape}")
            print(f"tgt_batch_ndata:{tgt_batch.ndata['h'].shape}")
            
            features = torch.cat((features_source, features_target), dim=0)
            vec = torch.cat((vec_source, vec_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            focals = torch.cat((focal_source,focal_target),dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            
            
            transfer_loss, bridge = loss.my_loss(features, [softmax_out,focals], ad_net, config['device']['device'], network.calc_coeff(i,max_iter = config['train']['epoch']))
            regression_loss = eval(config['train']['loss_function'])(outputs_source.squeeze(1), torch.tensor(src_acc_batch).to(config['device']['device']))
            total_loss = transfer_loss +config['train']['trade_off'] * regression_loss + config['net']['bridge'] * bridge
            log_str = "{}: Epoch: {:d}, total_loss: {:05f}, transferloss: {:.5f}, regression_loss: {:.5f}, bridge:{:.5f}".format(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()), i, total_loss , transfer_loss, regression_loss, bridge)


            print(log_str)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            total_loss.backward()
            optimizer.step()
            iter_num += 1
            acc, ktau = image_classification_test(i, step)

if __name__ == "__main__":
    path = './config.yaml'
    config = read_yaml(path)
    cuda = config['device']['use_cuda'] and torch.cuda.is_available()
    # if cuda:
    #     device = torch.device("cuda:"+config['device']['gpu_id'])
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config['device']['device'] = device

    config["output_path"] = os.path.join(os.path.dirname(os.path.realpath('__file__')), \
        'results/trainmodel-{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))


    config["best_model"] = osp.join(config["output_path"], "best_model.pth.tar")

    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    if config['data']['source_dataset_name'] == "NASBENCH101":
        train_data = NASBench101Dataset(config['data']["source_101_dataset_num"], all = config['data']["all_101"])
    
    if config['data']['target_dataset_name'] == "DARTS":
        tgt_dataset = DartsDataset(NUM_EXAMPLES_Darts=config['data']["target_dataset_num"], labeled=False)
        DARTS_test = DartsDataset(NUM_EXAMPLES_Darts=config['data']["target_dataset_num"], labeled=True)
        _, _, mean, std = read_darts_dataset()


    src_dataloader_train = GraphDataLoader(train_data, batch_size= config['train']['train_batch_size'], drop_last=True)
    tgt_dataloader_train = GraphDataLoader(tgt_dataset, batch_size= config['train']['train_batch_size'], drop_last=True)
    tgt_dataloader_test = GraphDataLoader(DARTS_test, batch_size= config['train']['infer_batch_size'], drop_last=True)

    

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
    model = network.model(embedding_len=config['net']['embedding_len'],inter_l=config['encoder']['inter_l'], feature_l=config['encoder']['feature_l'], use_embed=config['net']['use_embed'])

    model = model.to(device)
    ad_net = network.AdversarialNetwork(config['encoder']['feature_l'], config['net']['adnet_hidden_size'])
    ad_net = ad_net.to(device)
    print(config)
    train(config)




