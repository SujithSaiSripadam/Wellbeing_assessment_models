# set_seed preprocess getVideoFeature flat_data cut_videos plot_heatmap fourier_transform_resample generate_spectral_heatmap graph_visualization
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

import optuna
from optuna.samplers import TPESampler



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def preprocess(raw_data):
    num, length = raw_data.shape
    out_data = np.zeros((num, length))
    for i in range(num):
        out_data[i, :] = raw_data.iloc[i] - np.median(raw_data.iloc[i])
    return out_data

def getVideoFeature(feaVec):
    feaNum = feaVec.shape[0]
    videoFea = np.zeros((1, feaNum*20-6*6))
    # print(videoFea.shape)
    for i in range(feaNum):
        videoFea[0, (i-1)*12 + 1] = np.mean(feaVec[i,:])
        videoFea[0, (i-1)*12 + 2] = np.std(feaVec[i,:])
        videoFea[0, (i-1)*12 + 3] = np.max(feaVec[i,:])
        videoFea[0, (i-1)*12 + 4] = np.min(feaVec[i,:])
        videoFea[0, (i-1)*12 + 5] = np.mean(np.diff(feaVec[i,:]))
        videoFea[0, (i-1)*12 + 6] = np.std(np.diff(feaVec[i,:]))
        videoFea[0, (i-1)*12 + 7] = np.max(np.diff(feaVec[i,:]))
        videoFea[0, (i-1)*12 + 8] = np.min(np.diff(feaVec[i,:]))
        a = feaVec[i,0:-2]
        b = feaVec[i,2:]
        videoFea[0, (i-1)*12 + 9] = np.mean(a-b)
        videoFea[0, (i-1)*12 + 10] = np.std(a-b)
        videoFea[0, (i-1)*12 + 11] = np.max(a-b)
        videoFea[0, (i-1)*12 + 12] = np.min(a-b)
    return videoFea

def flat_data(data, t_length, N):
    flat_data = np.zeros(t_length)
    row_num = data.shape[0]
    for i in range(row_num):
        flat_data[i * N:(i + 1) * N] = data[i, :]
    return flat_data

def cut_videos(raw_data, fre_resolution):
    num_frames = raw_data.shape[1]
    raw_num_keep_frame = np.floor(num_frames /fre_resolution)
    num_keep_frame = raw_num_keep_frame * fre_resolution
    num_delete = num_frames - num_keep_frame
    num_delete_start = num_delete // 2
    num_delete_end = num_delete - num_delete_start
    # Remove the specified number of frames from the beginning and end
    raw_data = np.delete(raw_data, np.r_[:num_delete_start, num_frames - num_delete_end:].astype(int), axis=1)
    # raw_data = np.delete(raw_data, np.concatenate((np.arange(0, num_delete_start), np.arange(num_frames - num_delete_end, num_frames))).astype(int), axis=1)
    return raw_data, num_keep_frame, raw_num_keep_frame

def plot_heatmap(amp_map):
  print(amp_map.shape)
  print(amp_map)
  dataplot=sns.heatmap(amp_map, cmap='Reds')
  plt.show()
    
def fourier_transform_resample(all_data, N, num_fre):
    channel_num, length = all_data.shape
    amp_map = np.zeros((channel_num, num_fre))
    phase_map = np.zeros((channel_num, num_fre))
    # print(all_data.shape, amp_map.shape, phase_map.shape)
    for i in range(channel_num):
        temp_contain = np.fft.fft(all_data[i,:])
        if length % 2 == 0:
            temp_contain = temp_contain[:length//2+1]
        else:
            temp_contain = temp_contain[:(length+1)//2]

        temp_resample_data = resample(temp_contain, num_fre)
        amp_map[i,:] = np.abs(temp_resample_data) / length
        phase_map[i,:] = np.angle(temp_resample_data)
        
    amp_map_return = amp_map[:, :N]
    phase_map_return = phase_map[:, :N]
    return amp_map_return, phase_map_return
    
def generate_spectral_heatmap(dir_path,name, Primitive_num, K = 60, fre_resolution = 128):
  t_length = K * Primitive_num
  filename=dir_path+'/'+ name
  raw_data=pd.read_csv(filename, index_col=[0])
  processed_data = preprocess(raw_data)
  processed_data = processed_data.T
  # feature extraction
  sta_fea = getVideoFeature(processed_data)
  amp_map, phase_map = fourier_transform_resample(processed_data,K, fre_resolution)
  spectral_data= np.concatenate((amp_map, phase_map), axis=1)
  df_spectral_data=pd.DataFrame(spectral_data)
  return df_spectral_data  
    
def graph_visualization(G):
  pos = nx.spring_layout(G)
  plt.title("DGL Graph Visualization")
  nx.draw(G,pos, with_labels=True, node_color = 'green',node_size = 1500, font_size=10)   
    
   
    
    
    
    
    
    
    
    
    
    