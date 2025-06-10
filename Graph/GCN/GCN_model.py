# create_basic_graph apply_MEFG create_graph_list_V create_graph_list_A create_graph_list_EarlyFusion GCN train_epoch vaid_epoch 
import os
import pdb
import copy
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold,LeaveOneOut

import scipy.io
from scipy.signal import resample
from scipy.fftpack import fft
from scipy.io import loadmat
from collections import OrderedDict

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import sparse, to_torch_coo_tensor
import torch_geometric.transforms as T
from torch_geometric.nn import GATv2Conv,GATConv, global_mean_pool, GCNConv
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.init as init
import optuna
from optuna.samplers import TPESampler

from preprocessing import cut_videos, flat_data, fourier_transform_resample, set_seed, generate_spectral_heatmap, getVideoFeature, graph_visualization, plot_heatmap, preprocess, set_seed 
from Layers import MEFG, TTP, CrossAttention, CrossTransformer, CrossTransformerEncoder, FullAttention, GatedGCNLayer, LinearAttention, elu_feature_map 

def create_basic_graph(df_spdata, num_nodes):
  nx_G = nx.complete_graph(num_nodes)
  g = dgl.from_networkx(nx_G)
  # Adding node fatures to each node from the dataframe containing spectral representaion of each feature
  node_features = torch.tensor(df_spdata.values, dtype=torch.float32)
  # Set node features in the DGL graph
  g.ndata['feature'] = node_features
  g.edata['feature'] = torch.rand(g.num_edges(), 1)
  batch_graphs = g
  batch_x = batch_graphs.ndata['feature']
  batch_e = batch_graphs.edata['feature']
  # print('Graph before TTP',batch_graphs)
  ttp = TTP(in_dim = 120, hidden_dim = 240, edge_thresh=0.2)
  lr_g = ttp(batch_graphs, batch_x, batch_e)
  return lr_g

def apply_MEFG(g, num_nodes):
  batch_graphs = g
  batch_x = batch_graphs.ndata['feature']
  batch_e = batch_graphs.edata['feature']
  # print("batch_e shape:", batch_e.shape)
  # batch_x = increase_dim(batch_graphs,batch_x)
  mefg=MEFG(in_dim=120,hidden_dim=240, max_node_num= num_nodes, global_layer_num = 1, dropout = 0.1)
  # e_afterMEFG = mefg(batch_graphs, batch_x, batch_e)
  # return e_afterMEFG
  graph_afterMEFG = mefg(batch_graphs, batch_x, batch_e)
  return graph_afterMEFG

def create_graph_list_V(task_dir,csv_files, n):
  i=0
  G_list_MEFG_V=[]
  num_nodes = 31
  for i in range(n):
    if i< 50:
      df_file=generate_spectral_heatmap(task_dir,csv_files[i], Primitive_num = num_nodes, K = 60, fre_resolution = 128)
      graph_afterTTP= create_basic_graph(df_file, num_nodes)
      graph_afterMEFG=apply_MEFG(graph_afterTTP, num_nodes)
      G_list_MEFG_V.append(graph_afterMEFG)
      i+=1
      print(i)
  return G_list_MEFG_V
  

def create_graph_list_A(task_dir,csv_files, n):
  i=0
  G_list_MEFG_A=[]
  num_nodes = 91
  for i in range(n):
    if i< 50:
      df_file=generate_spectral_heatmap(task_dir,csv_files[i], Primitive_num = num_nodes, K = 60, fre_resolution = 128)
      graph_afterTTP= create_basic_graph(df_file, num_nodes)
      #print("TTP module output before MEFG:",graph_afterTTP)
      graph_afterMEFG=apply_MEFG(graph_afterTTP, num_nodes)
      #print("TTP module output after MEFG:",graph_afterMEFG)
      G_list_MEFG_A.append(graph_afterMEFG)
      i+=1
      print(i)
  return G_list_MEFG_A  


def create_graph_list_EarlyFusion(task_dir_V,task_csv_files_V, task_dir_A,task_csv_files_A, n):
  i=0
  G_list_MEFG_Early=[]
  num_nodes = 122
  for i in range (n):
      df_file_V=generate_spectral_heatmap(task_dir_V,task_csv_files_V[i], Primitive_num = 31, K = 60, fre_resolution = 128)
      df_file_A=generate_spectral_heatmap(task_dir_A,task_csv_files_A[i], Primitive_num = 91, K = 60, fre_resolution = 128)
      df_concat = pd.concat([df_file_V, df_file_A], axis=0)
      graph_afterTTP= create_basic_graph(df_concat, num_nodes)
      graph_afterMEFG=apply_MEFG(graph_afterTTP, num_nodes)
      G_list_MEFG_Early.append(graph_afterMEFG)
      i+=1
      print(i)
  return G_list_MEFG_Early

class GCN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, num_classes=1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                init.xavier_uniform_(m.lin.weight)
                if m.lin.bias is not None:
                    init.zeros_(m.lin.bias)
            elif isinstance(m, torch.nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, data, dropout_rate):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch=data.batch)
        x = self.lin(x)

        return x

def train_epoch(train_loader, dr, optimizer, criterion, model_GCN, scheduler):
    loss_train_per = []
    model_GCN.train()
    for data in train_loader:
        data = data.to(next(model_GCN.parameters()).device)
        optimizer.zero_grad()
        output_train = model_GCN(data, dropout_rate=dr)
        target_train = data.y
        output_train = output_train.view(-1, 1)
        target_train = target_train.view(-1, 1)
        loss = criterion(output_train, target_train)
        loss.backward()
        optimizer.step()
        loss_train_per.append(loss.item())
    loss_train_total = np.mean(loss_train_per)
    if scheduler is not None:
        scheduler.step()
    return loss_train_total, model_GCN, optimizer

def valid_epoch(test_loader, dr, criterion, model_GCN):
    val_loss_per = []
    model_GCN.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(next(model_GCN.parameters()).device)
            output_test = model_GCN(data, dropout_rate=dr)
            target_test = data.y
            output_test = output_test.view(-1, 1)
            target_test = target_test.view(-1, 1)
            loss = criterion(output_test, target_test)
            val_loss_per.append(loss.item())
    val_loss_total = np.mean(val_loss_per)
    return val_loss_total

def train_epoch_late(train_loader1, dr1, optimizer1, model_GCN1, scheduler1, train_loader2, dr2, optimizer2, model_GCN2, scheduler2,criterion):
    loss_train_per = []
    model_GCN1.train()
    model_GCN2.train()
    for data1, data2 in zip(train_loader1, train_loader2):
        optimizer1.zero_grad()
        output_train1 = model_GCN1(data1, dropout_rate= dr1)
        target_train1 = data1.y
        output_train1 = output_train1.view(-1,1)
        target_train1 = target_train1.view(-1,1)
        loss1 = criterion(output_train1, target_train1)
        loss1.backward(retain_graph= True)
        optimizer1.step()

        optimizer2.zero_grad()
        output_train2 = model_GCN2(data2, dropout_rate= dr2)
        target_train2 = data2.y
        output_train2 = output_train2.view(-1,1)
        target_train2 = target_train2.view(-1,1)
        loss2 = criterion(output_train2, target_train2)
        loss2.backward(retain_graph= True)
        optimizer2.step()
        final_output = (output_train1+output_train2)/2.0
        #predicted_labels = torch.argmax(final_output, dim=1)
        #target_fused = torch.max(target_train1, target_train2)
        loss = (loss1.item()+ loss2.item())/2.0
        loss_train_per.append(loss)
    loss_train_total= np.mean(loss_train_per)
    if scheduler1 is not None:
        scheduler1.step()  # Update learning rate
    if scheduler2 is not None:
        scheduler2.step()  # Update learning rate

    return loss_train_total, model_GCN1, optimizer1, model_GCN2, optimizer2

def valid_epoch_late(test_loader1, dr1, model_GCN1, test_loader2, dr2, model_GCN2, criterion):
  val_loss_per=[]
  model_GCN1.eval()
  model_GCN2.eval()
  with torch.no_grad():
      for data1, data2 in zip(test_loader1, test_loader2):
        output_test1 = model_GCN1(data1, dropout_rate= dr1)
        target_test1 = data1.y
        output_test1 = output_test1.view(-1,1)
        target_test1 = target_test1.view(-1,1)
        output_test2 = model_GCN2(data2, dropout_rate= dr2)
        target_test2 = data2.y
        output_test2 = output_test2.view(-1,1)
        target_test2 = target_test2.view(-1,1)
        output_test_fused = (output_test1+output_test2)/2.0
        target_fused = (target_test1+target_test2)/2.0
        loss = criterion(output_test_fused, target_fused)
        val_loss_per.append(loss.item())
  val_loss_total= np.mean(val_loss_per)
  return val_loss_total