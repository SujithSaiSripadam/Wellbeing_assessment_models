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

from preprocessing import cut_videos, flat_data, fourier_transform_resample, generate_spectral_heatmap, getVideoFeature, graph_visualization, plot_heatmap, preprocess, set_seed 
from Layers import MEFG, TTP, CrossAttention, CrossTransformer, CrossTransformerEncoder, FullAttention, GatedGCNLayer, LinearAttention, elu_feature_map
from GCN_model import GCN, apply_MEFG, create_basic_graph, create_graph_list_A, create_graph_list_EarlyFusion, create_graph_list_V, train_epoch, train_epoch_late, valid_epoch, valid_epoch_late

labels=[13,4,19,14,3,3,2,4,3,7,1,5,4,0,7,0,3,4,2,13,8,0,0,2,2,1,6,3,2,3,4,2,4,2,0,7,1,13,2]
values=[5,5,13,4,19,14,3,3,2,4,3,7,1,5,4,0,7,0,3,4,2,13,8,0,0,2,2,1,6,3,2,3,4,2,4,2,0,7,1,13,2]

def normalize_target(target,mean_target, std_target):
    return (target - mean_target) / std_target

def denormalize_output(output, mean, std):
    if isinstance(mean, torch.Tensor):
        mean = mean.item()
    if isinstance(std, torch.Tensor):
        std = std.item()
    return (output * std) + mean

seed_value = 42
set_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device', device)

import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Train GCN model for Audio-Visual regression")

parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
parser.add_argument("--n_trials", type=int, required=True, help="Number of Optuna trials")
parser.add_argument("--task_name", type=str, required=True, help="Task name for audio (e.g., Task2h_A)")

# Parse the arguments
args = parser.parse_args()

# Extract individual arguments
batch_size = args.batch_size
n_trials = args.n_trials
task_name = args.task_name
taskv_name = task_name.replace("Task", "Task_").replace("_A", "")

# Print them out
print(f"[INFO] Batch Size      : {batch_size}")
print(f"[INFO] Num Trials      : {n_trials}")
print(f"[INFO] Task Name (A)   : {task_name}")
print(f"[INFO] Task Name (V)   : {taskv_name}")

#########################################################################################------> Changes for audio/Video
task_dir_V=f"/rds/user/sss77/hpc-work/Graph_Data/Modify_Data/{taskv_name}"
task_csv_files_V=os.listdir(task_dir_V)
task_csv_files_V.sort()
n = len(task_csv_files_V)
print(f"n0 = {n}")

task_dir_A=f"/rds/user/sss77/hpc-work/Graph_Data/Modify_Audio_Data/{task_name}"
task_csv_files_A=os.listdir(task_dir_A)
task_csv_files_A.sort()
n = len(task_csv_files_A)
print(f"n00 = {n}")

G_list_MEFG_A = create_graph_list_A(task_dir_A,task_csv_files_A,n)
print(f"n1 = {len(G_list_MEFG_A)}")

G_list_MEFG_V = create_graph_list_V(task_dir_V,task_csv_files_V,n)
print(f"n2 = {len(G_list_MEFG_V)}")

if n == 39:
  y_train_i = torch.tensor(labels, dtype=torch.float32)
else :
  print("check1")
  y_train_i = torch.tensor(values, dtype=torch.float32)
  
global mean_target 
global std_target

mean_target = y_train_i.mean()
std_target = y_train_i.std()

y_train = normalize_target(y_train_i, mean_target, std_target)

full_dataset_V = []
for idx in range(0 ,n):
  data=Data(x=G_list_MEFG_V[idx].ndata['feature'],  edge_index=torch.stack([G_list_MEFG_V[idx].all_edges()[0], G_list_MEFG_V[idx].all_edges()[1]], dim=0) ,edge_attr=G_list_MEFG_V[idx].edata['feature'],y=y_train[idx])
  full_dataset_V.append(data)

full_dataset_A = []
for idx in range(0 ,n):
  data=Data(x=G_list_MEFG_A[idx].ndata['feature'],  edge_index=torch.stack([G_list_MEFG_A[idx].all_edges()[0], G_list_MEFG_A[idx].all_edges()[1]], dim=0) ,edge_attr=G_list_MEFG_A[idx].edata['feature'],y=y_train[idx])
  full_dataset_A.append(data)

#====================================================================================================================================
##################################################  Adam Train ######################################################################
#====================================================================================================================================

def train_valid_loop_ADAM_late(lr1, dr1, hidden_classes1, lr2, dr2, hidden_classes2, trial_subdir, plot_subdir, task_name, batch_size):
    num_epochs = 50
    best_epoch = -1
    best_stop_epoch=-1
    early_stop_count = 0
    best_val_loss = float('inf')
    patience = 7
    all_train_losses = [[] for _ in range(n)]
    avg_train_losses=[]
    all_val_losses = []

    model_GCN1 = [GCN(node_features= 240,hidden_channels=hidden_classes1, num_classes=1).to(device) for _ in range(n)]
    optimizer1 = [torch.optim.Adam(model_GCN1[_].parameters(), lr= lr1) for _ in range(n)]
    scheduler1 = [StepLR(optimizer1[_], step_size=20, gamma=0.9) for _ in range(n)] # decrease LR by a factor of 0.9 every epoch

    
    model_GCN2 = [GCN(node_features= 240,hidden_channels=hidden_classes2, num_classes=1).to(device) for _ in range(n)]
    optimizer2 = [torch.optim.Adam(model_GCN2[_].parameters(), lr= lr2) for _ in range(n)]
    scheduler2 = [StepLR(optimizer2[_], step_size=20, gamma=1) for _ in range(n)] # decrease LR by a factor of 0.9 every epoch

    criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)
    
    for epoch in range(num_epochs):
        epoch_subdir = os.path.join(trial_subdir, f"epoch_{epoch}")
        os.makedirs(epoch_subdir, exist_ok=True)
        val_losses=[]
        for idx in range(n):
            train_indices = list(range(n))
            del train_indices[idx]
            train_dataset1 = [full_dataset_V[i].to(device) for i in train_indices]
            test_dataset1 = [full_dataset_V[idx].to(device)]
            train_dataset2 = [full_dataset_A[i].to(device) for i in train_indices]
            test_dataset2 = [full_dataset_A[idx].to(device)]

            train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
            test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)
            train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
            test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

            # Training loop
            train_loss, model_GCN1[idx], optimizer1[idx], model_GCN2[idx], optimizer2[idx] = train_epoch_late(train_loader1, dr1, optimizer1[idx], model_GCN1[idx], scheduler1[idx], train_loader2, dr2, optimizer2[idx], model_GCN2[idx], scheduler2[idx],criterion)
            all_train_losses[idx].append(train_loss)

            # Validation loop
            val_loss = valid_epoch_late(test_loader1, dr1, model_GCN1[idx], test_loader2, dr2, model_GCN2[idx], criterion)
            val_losses.append(val_loss)
            
            model_filename1 = os.path.join(epoch_subdir, f"Model1_subject_{idx}_model.pt")
            torch.save({
                        'model_state_dict1': model_GCN1[idx].state_dict(),
                        'learning rate1': lr1,
                        'dropout1':dr1,
                        'hidden_classes1':hidden_classes1,
                        }, model_filename1)
            model_filename2 = os.path.join(epoch_subdir, f"Model2_subject_{idx}_model.pt")
            torch.save({
                        'model_state_dict2': model_GCN2[idx].state_dict(),
                        'learning rate2': lr2,
                        'dropout2':dr2,
                        'hidden_classes2':hidden_classes2,
                        }, model_filename2)

        all_val_losses.append(np.mean(val_losses))
        print(f" For epoch {epoch}, Validation Loss: {np.mean(val_losses):.4f}")
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            early_stop_count = 0
            best_stop_epoch= epoch
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print(f'Early stopped at epoch: {epoch}')
            best_stop_epoch= epoch
            break

    for i in range(best_stop_epoch+1):
      avg_train_losses.append(np.average([x[i] for x in all_train_losses]))

    rows = (n // 6) + (1 if n % 6 != 0 else 0)  # ceil(n / 6)
    cols = min(n, 6)
    fig_loss, axs_loss = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    if rows == 1 and cols == 1:
        axs_loss = np.array([axs_loss])
    elif rows == 1 or cols == 1:
        axs_loss = np.ravel(axs_loss)
    else:
        axs_loss = axs_loss.flatten()

    for idx in range(n):
        ax_loss = axs_loss[idx]  # Ensure index is valid
        ax_loss.plot(range(len(all_train_losses[idx])), all_train_losses[idx], label='Train', color='red')
        ax_loss.set_title(f"Loss - Subject {idx}-{task_name}")
        ax_loss.legend()
    for i in range(n, len(axs_loss)):
        fig_loss.delaxes(axs_loss[i])

    fig_loss.tight_layout()
    fig, ax = plt.subplots(figsize=(10, 6))
    if all_val_losses and avg_train_losses:
        ax.plot(range(len(all_val_losses)), all_val_losses, label='Validation Loss', color='orange')
        ax.plot(range(len(avg_train_losses)), avg_train_losses, label='Training Loss', color='blue')

    ax.set_title('Train + Val Loss vs. Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.tight_layout()
    fig_loss.savefig(os.path.join(plot_subdir+"/"+f"training_loss.png"))
    fig.savefig(os.path.join(plot_subdir+"/"+f"validation_plots.png"))

    print("Final validation loss:", best_val_loss)
    print("Epoch at which Best model with lowest validation loss is obtained:", best_epoch)
    print("Epochs at which Early stopping is done:", best_stop_epoch)
    return best_val_loss, best_stop_epoch

#====================================================================================================================================
################################################## Test Epoch #######################################################################
#====================================================================================================================================

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Function to compute Concordance Correlation Coefficient (CCC)

def evaluate_model_late(task_name, full_dataset_V, full_dataset_A, n, device, trial, epoch, mean_target, std_target, batch_size):
    def concordance_corr_coef(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) < 2: return np.nan
        mean_true, mean_pred = y_true.mean(), y_pred.mean()
        var_true, var_pred = y_true.var(), y_pred.var()
        corr, _ = pearsonr(y_true, y_pred)
        return (2 * corr * np.sqrt(var_true) * np.sqrt(var_pred)) / (var_true + var_pred + (mean_true - mean_pred) ** 2)

    save_dir = f"/GCN_checkpoints_{task_name}/trial_{trial}/epoch_{epoch}/"
    print(f"Evaluating from checkpoint: {save_dir}")
    
    all_preds, all_targets, total_rmse_list = [], [], []

    for idx in range(n):
        # Load both model checkpoints
        ckpt1_pth = os.path.join(save_dir, f"Model1_subject_{idx}_model.pt")
        checkpoint1 = torch.load(ckpt1_pth, map_location=device)
        model_state_dict1, lr1, dr1, hidden_classes1 = checkpoint1['model_state_dict1'], checkpoint1['learning rate1'], checkpoint1['dropout1'], checkpoint1['hidden_classes1']
        print(f"learning rate: {lr1}, dropout: {dr1}, hidden_classes: {hidden_classes1}")
        model_GCN1 = GCN(node_features=240, hidden_channels=hidden_classes1, num_classes=1).to(device)
        model_GCN1.load_state_dict(model_state_dict1)
        model_GCN1.eval()
        
        ckpt2_pth = os.path.join(save_dir, f"Model2_subject_{idx}_model.pt")
        checkpoint2 = torch.load(ckpt2_pth, map_location=device)
        model_state_dict2, lr2, dr2, hidden_classes2 = checkpoint2['model_state_dict2'], checkpoint2['learning rate2'], checkpoint2['dropout2'], checkpoint2['hidden_classes2']
        print(f"learning rate: {lr2}, dropout: {dr2}, hidden_classes: {hidden_classes2}")
        model_GCN2 = GCN(node_features=240, hidden_channels=hidden_classes2, num_classes=1).to(device)
        model_GCN2.load_state_dict(model_state_dict2)
        model_GCN2.eval()

        # Prepare test loaders
        test_dataset1 = [full_dataset_V[idx].to(device)]
        test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)
        test_dataset2 = [full_dataset_A[idx].to(device)]
        test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

        model_GCN1.eval()
        model_GCN2.eval()
        with torch.no_grad():
            for data_V, data_A in zip(test_loader1, test_loader2):
                out1 = model_GCN1(data_V, dropout_rate=dr1).cpu().numpy().flatten()
                target_test1 = data_V.y.cpu().numpy().flatten()
                out2 = model_GCN2(data_A, dropout_rate=dr2).cpu().numpy().flatten()
                target_test2 = data_A.y.cpu().numpy().flatten()
                prediction1 = denormalize_output(out1, mean_target, std_target)
                prediction2 = denormalize_output(out2, mean_target, std_target)
                
                prediction = (prediction1 + prediction2) / 2.0
                
                target1 = denormalize_output(target_test1, mean_target, std_target)
                target2 = denormalize_output(target_test2, mean_target, std_target)
                
                target = (target1 + target2)/2.0
                all_preds.extend(prediction.tolist())
                all_targets.extend(target.tolist())
                
        mse = mean_squared_error(target, prediction)
        rmse = np.sqrt(mse)
        total_rmse_list.append(rmse)

    # Metrics
    final_rmse = np.mean(total_rmse_list)
    pcc, _ = pearsonr(all_preds, all_targets)
    ccc = concordance_corr_coef(all_targets, all_preds)

    # Save results
    output_file = f"/rds/user/sss77/hpc-work/GCN_Late/{task_name}/{task_name}_results_{trial}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"Task Name: {task_name}\n")
        f.write(f"Final Test RMSE: {final_rmse:.4f}\n")
        f.write(f"PCC: {pcc:.4f}\n")
        f.write(f"CCC: {ccc:.4f}\n")
    print(f"RMSE: {final_rmse:.4f}, PCC: {pcc:.4f}, CCC: {ccc:.4f}\nResults saved to: {output_file}")

#====================================================================================================================================
################################################## Epoch Train ######################################################################
#====================================================================================================================================

best_trial_dir = None
best_trial_loss = float('-inf')
def objective_ADAM_late(trial):
    global best_trial_dir, best_trial_loss
    lr1 = trial.suggest_float("lr1", 1e-5, 1e-2)
    lr2 = trial.suggest_float("lr2", 1e-5, 1e-2)
    dr1 = trial.suggest_float("dropout1", 0.1, 0.8)
    dr2 = trial.suggest_float("dropout2", 0.1, 0.8)
    hidden_classes1 = trial.suggest_int("hidden_classes1", 16, 180)
    hidden_classes2 = trial.suggest_int("hidden_classes2", 16, 180)
    print("learning rate 1:", lr1,"learning rate 2:", lr2,"dropout rate1:", dr1, "dropout rate2:", dr2,"hidden_classes1 :", hidden_classes1, "hidden_classes2:", hidden_classes2)

    save_dir=f'/rds/user/sss77/hpc-work/GCN_Late/GCN_checkpoints_{task_name}/'
    trial_subdir = os.path.join(save_dir, f"trial_{trial.number}")
    os.makedirs(trial_subdir, exist_ok=True)

    plot_dir=f'/rds/user/sss77/hpc-work/GCN_Late/GCN_plots_{task_name}/'
    plot_subdir = os.path.join(plot_dir, f"trial_{trial.number}")
    os.makedirs(plot_subdir, exist_ok=True)

    # test_acc = train_valid_loop_ADAM(lr, dr, hidden_classes,num_heads, trial_subdir)
    val_loss, stop_epoch = train_valid_loop_ADAM_late(lr1,dr1,hidden_classes1,lr2, dr2, hidden_classes2,trial_subdir, plot_subdir, task_name, batch_size)
    if val_loss < best_trial_loss:
      best_trial_loss = val_loss
      remove_trial_subdir= os.path.join(save_dir,f"trial_{best_trial_dir}")
      best_trial_dir= trial.number
      
    evaluate_model_late(task_name=task_name, 
                    full_dataset_V=full_dataset_V, 
                    full_dataset_A=full_dataset_A,
                    n=n, device=device, 
                    trial=trial.number, 
                    epoch=stop_epoch, 
                    mean_target=mean_target, 
                    std_target=std_target,
                    batch_size=batch_size)
    return val_loss

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed= seed_value))
study.optimize(objective_ADAM_late, n_trials= n_trials )
print(study.best_trial)