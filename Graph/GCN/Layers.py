# GatedGCNLayer CrossAttention elu_feature_map LinearAttention FullAttention CrossTransformerEncoder CrossTransformer TTP MEFG
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

from preprocessing import cut_videos, flat_data, fourier_transform_resample, generate_spectral_heatmap, getVideoFeature, graph_visualization, plot_heatmap, preprocess, set_seed 

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    # input_dim= 120, output_dim= 240
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False
        #  self.residual = False
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        self.C = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, g, h, e):

        h_in = h # for residual connection
        e_in = e # for residual connection
        # print("node matrix shape :", h.shape,"edge matrix shape:", e.shape)
        g.ndata['h']  = h
        # h=h.T
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e']  = e
        # g.edata['Ce'] = self.C(e)

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        # print("Size of DEh :", g.edata['DEh'].shape)
        # g.edata['e'] = g.edata['DEh'] + g.edata['Ce'] # Original code
        g.edata['e'] = g.edata['DEh'] + e # Changed code excluding g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func)
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        # print("h shape:", h.shape,"e shape :", e.shape)
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization
            e = self.bn_node_e(e) # batch normalization

        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            # self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None


        self.window_size = None
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None, k=None, v=None):
        B, N, C = x.shape

        if k is None:
            k = x
            v = x
            N_k = N
            N_v = N
        else:
            N_k = k.shape[1]
            N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)                        # (B, N_q, dim)
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)                        # (B, N_k, dim)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)

        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, num_heads, N_q, dim)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, num_heads, N_k, dim)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, num_heads, N_v, dim)


        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()

class CrossTransformerEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 drop = 0.1):
        super(CrossTransformerEncoder, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            #nn.Dropout(drop),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
            nn.Dropout(drop),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        # print("shape of x:", x.shape," shape of source:",source.shape)
        query, key, value = x, source, source
        # print(" shape of query:", query.shape," shape of key:", key.shape," shape of value:", value.shape)
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        # print(" query shape: [N, L, (H, D)]", query.shape)
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        # print("shape of key should be [N, S, (H, D)] :", key.shape)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        # print("message shape should be [N, L, (H, D)] :", message.shape)
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        # print("message shape should be [N, L, C] :", message.shape)
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class CrossTransformer(nn.Module):

    def __init__(self, d_model, nhead=1, layer_nums=1, attention_type='linear'):
        super().__init__()

        encoder_layer = CrossTransformerEncoder(d_model, nhead, attention_type)
        self.VCR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.VVR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, qfea, kfea, mask0=None, mask1=None):
        """
        Args:
            qfea (torch.Tensor): [B, N, D]
            kfea (torch.Tensor): [B, D]
            mask0 (torch.Tensor): [B, N] (optional)
            mask1 (torch.Tensor): [B, N] (optional)
        """
        #assert self.d_model == qfea.size(2), "the feature number of src and transformer must be equal"

        B,N,D = qfea.shape
        kfea = kfea.unsqueeze(1).repeat(1, N, 1) #[B,N,D]
        # print(" shape of kfea should be [B,N,D] :", kfea.shape)

        mask1 = torch.ones([B,N]).to(qfea.device)
        for layer in self.VCR_layers:
            qfea = layer(qfea, kfea, mask0, mask1) #[B,N,D]
            # print(" shape of qfea should be [B,N,D] :", qfea.shape)
            #kfea = layer(kfea, qfea, mask1, mask0)

        qfea_end = qfea.repeat(1,1,N).view(B,-1,D) #[B,N*N,D]
        # print(" shape of qfea end should be [B,N*N,D] :", qfea_end.shape)
        qfea_start = qfea.repeat(1,N,1).view(B,-1,D) #[B,N*N,D]
        # print(" shape of qfea start should be [B,N*N,D] :", qfea_start.shape)
        #mask2 = mask0.repeat([1,N])
        for layer in self.VVR_layers:
            #qfea_start = layer(qfea_start, qfea_end, mask2, mask2)
            qfea_start = layer(qfea_start, qfea_end)#[B,N*N,D]
            # print(" shape of qfea start should be [B,N*N,D] :", qfea_start.shape)

        return qfea_start.view([B,N,N,D])
    
# The TTP code
class TTP(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_thresh):
        super().__init__()
        self.proj_g1 = nn.Linear(in_dim,hidden_dim**2)
        self.bn_node_lr_g1 = nn.BatchNorm1d(hidden_dim**2)
        self.proj_g2 = nn.Linear(in_dim,hidden_dim)
        self.bn_node_lr_g2 = nn.BatchNorm1d(hidden_dim)
        self.hidden_dim = hidden_dim #lr_g
        self.proj_g = nn.Linear(hidden_dim, 1)
        self.edge_thresh = edge_thresh
    def forward(self, g, h, e):
        lr_gs = []
        gs = dgl.unbatch(g)
        for g in gs:
            N = g.number_of_nodes()
            h_single = g.ndata['feature'].to(h.device)
            h_proj1 = F.dropout(F.relu(self.bn_node_lr_g1(self.proj_g1(h_single))), 0.1, training=self.training).view(-1,self.hidden_dim)
            h_proj2 = F.dropout(F.relu(self.bn_node_lr_g2(self.proj_g2(h_single))), 0.1, training=self.training).permute(1,0)
            mm = torch.mm(h_proj1,h_proj2)
            mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]
            # print("Golbal Contextual Representation shape:", mm.shape)
            mm = self.proj_g(mm).squeeze(-1)
            # print("Xh matrix shape:", mm.shape)
            diag_mm = torch.diag(mm)
            diag_mm = torch.diag_embed(diag_mm)
            mm -= diag_mm
            #matrix = torch.sigmoid(mm)
            #matrix = F.softmax(mm, dim=0)
            matrix = F.softmax(mm, dim=0) * F.softmax(mm, dim=1)
            # print("Adjacency probability matrix",matrix.shape)
            #binarized = BinarizedF()
            #matrix = binarized.apply(matrix) #(0/1)
            lr_connetion = torch.where(matrix>self.edge_thresh)
            # print(lr_connetion[0], lr_connetion[1])
            g.add_edges(lr_connetion[0], lr_connetion[1])
            # print("Learned TTP Graph", g)
            lr_gs.append(g)
        g = dgl.batch(lr_gs).to(h.device)

        return g
    
'''
M =num_edges
Din= input_dim = 120
hidden_num= hidden dim =64
Nmax= num_nodes = 31
D= node_features =120
B = Batch
'''
class MEFG(nn.Module):
    def __init__(self, in_dim,hidden_dim, max_node_num, global_layer_num = 1, dropout = 0.1):
        super().__init__()
        self.edge_proj = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.edge_proj2 = nn.Linear(in_dim,hidden_dim) #baseline4
        self.edge_proj3 = nn.Linear(in_dim,hidden_dim)
        self.edge_proj4 = nn.Linear(hidden_dim,hidden_dim)
        self.hidden_dim = hidden_dim #baseline4
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)

        self.max_node_num = max_node_num
        self.global_layers = nn.ModuleList([GatedGCNLayer(in_dim, hidden_dim, dropout,
                                    True, True) for _ in range(global_layer_num -1) ])
        self.global_layers.append(GatedGCNLayer(in_dim, hidden_dim, dropout, True, True))
        #self.global_layers = nn.ModuleList([ ResidualAttentionBlock( d_model = hidden_dim, n_head = 1)
        #                                    for _ in range(global_layer_num) ])
        self.CrossT = CrossTransformer(hidden_dim, nhead=1, layer_nums=1, attention_type='linear')

    def forward(self, g, h, e):

        g.apply_edges(lambda edges: {'src' : edges.src['feature']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        # print("src shape:", src.shape)
        g.apply_edges(lambda edges: {'dst' : edges.dst['feature']})
        dst = g.edata['dst'].unsqueeze(1) #[M,1,D]
        # print("dst shape:", dst.shape)
        edge = torch.cat((src,dst),1).to(h.device) #[M,2,D] [num_edges,2,node_features]
        # print("edge shape:", edge.shape)
        lr_e_local = self.edge_proj(edge).squeeze(1)#[M,D]
        # print('lr_e_local shape:', lr_e_local.shape)
        lr_e_local = self.edge_proj2(lr_e_local)  #[num_edges,hidden_dim]
        # print('lr_e_local shape:', lr_e_local.shape)

        hs = []
        gs = dgl.unbatch(g)
        # print("Graph g :", g)
        # print("Graph gs:", gs)
        mask0 = torch.zeros([len(gs),self.max_node_num]).to(h.device) #[1, num_nodes]
        for i,g0 in enumerate(gs):
            Ng = g0.number_of_nodes()
            padding = nn.ConstantPad1d((0,self.max_node_num - Ng),0)
            pad_h = padding(g0.ndata['feature'].T).T #[Nmax, D]
            hs.append(pad_h.unsqueeze(0))
            mask0[i,:Ng] = 1
        hs = torch.cat(hs,0).to(h.device) #[B,Nmax,Din]
        # print("vertex feature shape:", hs.shape)
        hs = self.edge_proj3(hs) #[B,Nmax,hidden_num]
        # print("vertex feature shape 2:", hs.shape)

        if e is None:
            e = torch.ones([g.number_of_edges() ,h.shape[-1]]).to(h.device)
        # Gated-GCN for extract global feature
        hs2g = h
        # print("hs2g shape :", hs2g.shape)
        # print("e :",e.shape)
        # print("global layers :", self.global_layers)
        # print("length of global layers:", len(self.global_layers))
        for conv in self.global_layers:
            # print("conv :", conv)
            hs2g, _ = conv(g, hs2g, e)
        g.ndata['hs2g'] = hs2g
        g.ndata['feature']=hs2g
        # print("hs2g shape:", hs2g.shape)
        global_g = dgl.mean_nodes(g, 'hs2g') #[B,hidden_num]
        # print("shape of global layer:", global_g.shape)

        '''
        # Transformer for extract global feature
        mask_t = mask0.unsqueeze(1)*mask0.unsqueeze(2)
        mask_t = (mask_t==0)
        #mask_t = None

        hs2g = hs.permute((1,0,2))
        for conv in self.global_layers:
            hs2g = conv(hs2g, mask_t)
        global_g = hs2g.permute((1,0,2)).mean(1) #[B,D]
        '''
        # hs ([B, MaxnumNode, Hidden_Num])
        # global_g ([B, Hidden_Num])
        # print("hs shape:", hs.shape)
        edge = self.CrossT(hs, global_g, mask0) #[B,N,N,D]
        # print("edge shape :", edge.shape)

        index_edge = []
        for i,g0 in enumerate(gs):
            index_edge.append(edge[i, g0.all_edges()[0],g0.all_edges()[1],:])
        index_edge = torch.cat(index_edge,0)

        lr_e_global = self.edge_proj4(index_edge)

        if e is not None:
            e = e + lr_e_local + lr_e_global
        else:
            e = lr_e_local + lr_e_global
#        lr_e = lr_e_local + lr_e_global
        # bn=>relu=>dropout
        # print("edge matrix shape :", e.shape)
        e = self.bn_node_lr_e(e)
        e = F.relu(e)
        e = F.dropout(e, 0.1, training=self.training)
        # print("edge matrix shape 2 :", e.shape)
        g.edata['feature']=e
        # print("graph in MEFG module:", g)
        # return e
        return g   
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    