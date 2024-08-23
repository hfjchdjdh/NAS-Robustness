import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import igraph
import dgl
from dgl.nn.pytorch import GraphConv

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

def one_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=1)
        nn.init.zeros_(m.bias)

def hun_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=100)
        nn.init.zeros_(m.bias)


class model(nn.Module):
  def __init__(self, embedding_len=3, inter_l=64, feature_l=50, new_cls=True, bridge_num=1, bridge_initial=True, use_embed=True):
    super(model, self).__init__()
    self.model = Encoder(embedding_len=embedding_len ,inter_len=inter_l, feature_len=feature_l, use_embed=use_embed)
    self.model_2 = Encoder(embedding_len=embedding_len ,inter_len=inter_l, feature_len=feature_l, use_embed=use_embed)
    self.fc = nn.Linear(feature_l, 1)
    self.gvbg = nn.Linear(feature_l, 1)
    nz = feature_l

    self.sigmoid = nn.Sigmoid()
    self.new_cls = new_cls
    self.bridge_num = bridge_num
    num_f = 50
    if new_cls:
        self.fc = nn.Linear(nz, num_f)
        self.fc2 = nn.Linear(num_f, 1)
        if bridge_initial:
            self.fc.apply(hun_weights)
        else:
            self.fc.apply(init_weights)
        self.bridge = nn.Linear(nz, num_f)
        self.bridge.apply(init_weights)
        self.__in_features = nz
    else:
        self.fc = nn.Linear(nz, 1)
        self.__in_features = nz

    self.W1, self.W2 = self.model.get_weight()

  def forward(self, g, input_data, bridge=True):
    x = self.model(g, input_data)
    x = x.view(x.size(0), -1)
    geuristic = self.bridge(x)
    y = self.fc(x)
    if bridge:
        y = y - geuristic
    final =  self.fc2(y)
    return x, final, y, geuristic

  def get_parameters(self):
    parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list

  def get_weight(self):
      W1,W2 = self.model.get_weight()
      return W1,W2


MAX_FEATURES_Darts = 6
class Encoder(nn.Module):
    def __init__(self, embedding_len=3, inter_len=64, feature_len=50, use_embed=True):
        super(Encoder, self).__init__()
        if use_embed:
            self.conv1= GraphConv(embedding_len, inter_len)
        else:
            self.conv1= GraphConv(MAX_FEATURES_Darts, inter_len)
        self.bn1=nn.BatchNorm1d(inter_len)
        self.relu= nn.ReLU(True)
        self.conv2= GraphConv(inter_len, feature_len)
        self.bn2=nn.BatchNorm1d(feature_len)
        self.dropout=nn.Dropout()
        self.use_embed = use_embed
        self.embeddings = nn.Embedding(MAX_FEATURES_Darts, embedding_len)
    

    def _get_embeddings(self,X):
        #get one hot vector as input and returns the embeddings
        X_emb=[]
        dev = X.device
        X_l = X.data.tolist()
        for label in X_l:
            for i in range(0, len(label)):
                if label[i] == 1:
                    X_emb.append(i)
                    break
        X_emb = torch.LongTensor(X_emb).to(str(dev))
        X_emb=self.embeddings(X_emb)
        X_emb=torch.tensor(X_emb).float()
        return X_emb

    def forward(self, g, input_data):
        if self.use_embed:
            input_data = self._get_embeddings(input_data)

        h = self.conv1(g, input_data)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(g,h)
        h = self.bn2(h)
        h = self.dropout(h)
        g.ndata['h'] = h
        # print(f"h.shape:{h.shape}")
        # print(f"h:{h}")
        feature = dgl.readout_nodes(g, 'h', op='mean')
        return feature
    def get_weight(self):
        W1 = self.conv1.weight
        W2 = self.conv2.weight
        return W1,W2

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.gvbd = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.dropout3 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    return y

  def get_parameters(self):
    parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list
