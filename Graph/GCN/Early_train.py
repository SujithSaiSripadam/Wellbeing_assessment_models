import os
import pdb
import copy
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
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
import json

from preprocessing import cut_videos, flat_data, fourier_transform_resample, generate_spectral_heatmap, getVideoFeature, graph_visualization, plot_heatmap, preprocess, set_seed 
from Layers import MEFG, TTP, CrossAttention, CrossTransformer, CrossTransformerEncoder, FullAttention, GatedGCNLayer, LinearAttention, elu_feature_map
from GCN_model import GCN, apply_MEFG, create_basic_graph, create_graph_list_A, create_graph_list_EarlyFusion, create_graph_list_V, train_epoch, valid_epoch 

# Base directory - Change this when moving the code to another system
base_dir = "/rds/user/sss77/hpc-work/GCN_Early"  # <-- Modify this path as needed

# Task-specific base directories
task_base_dir = os.path.join("/rds/user/sss77/hpc-work/New/Graph/Graph_Data/Modify_Data")
task_audio_base_dir = os.path.join("/rds/user/sss77/hpc-work/New/Graph/Graph_Data/Modify_Audio_Data")

# GCN-related directories
gcn_checkpoints_dir = os.path.join(base_dir, "GCN_checkpoints")
gcn_plots_dir = os.path.join(base_dir, "GCN_plots")
gcn_study_dir = os.path.join(base_dir, "GCN_study")
gcn_results_dir = os.path.join(base_dir, "GCN_results")

# Ensure required directories exist
for d in [gcn_checkpoints_dir, gcn_plots_dir, gcn_study_dir, gcn_results_dir]:
    os.makedirs(d, exist_ok=True)


labels_path = os.path.join("/rds/user/sss77/hpc-work/New/Graph/Graph_Data", 'labels.npy') 
values_path = os.path.join("/rds/user/sss77/hpc-work/New/Graph/Graph_Data", 'values.npy')  # Update this path if needed

labels = np.load(labels_path)
values = np.load(values_path)

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

# Argument parser setup (commented out for now)
import argparse
parser = argparse.ArgumentParser(description="Train GCN model for Audio-Visual regression")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
parser.add_argument("--n_trials", type=int, required=True, help="Number of Optuna trials")
parser.add_argument("--task_name", type=str, required=True, help="Task name for audio (e.g., Task2h_A)")
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

# Define task directories
task_dir_V = os.path.join(task_base_dir, taskv_name)
task_csv_files_V = os.listdir(task_dir_V)
task_csv_files_V.sort()
n = len(task_csv_files_V)
print(f"n1 = {n}")

task_dir_A = os.path.join(task_audio_base_dir, task_name)
task_csv_files_A = os.listdir(task_dir_A)
task_csv_files_A.sort()
n = len(task_csv_files_A)
print(f"n1 = {n}")

n= 3

G_list_MEFG_Early = create_graph_list_EarlyFusion(task_dir_V, task_csv_files_V, task_dir_A, task_csv_files_A, n)

if n == 39:
    y_train_i = torch.tensor(labels, dtype=torch.float32)
else:
    print("check1")
    y_train_i = torch.tensor(values, dtype=torch.float32)

mean_target = y_train_i.mean()
std_target = y_train_i.std()
y_train = normalize_target(y_train_i, mean_target, std_target)

full_dataset = []
for idx in range(0, n):
    data = Data(x=G_list_MEFG_Early[idx].ndata['feature'],
                edge_index=torch.stack([G_list_MEFG_Early[idx].all_edges()[0], G_list_MEFG_Early[idx].all_edges()[1]], dim=0),
                edge_attr=G_list_MEFG_Early[idx].edata['feature'],
                y=y_train[idx])
    full_dataset.append(data)

# --- Utility: CCC and Save Intermediate ---
def ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

def save_intermediate_gcn(test_idx, preds, trues, params, task_name, plot_dir, model_dir, trial_num=None):
    prefix = f"trial_{trial_num}_test_{test_idx}" if trial_num is not None else f"test_{test_idx}"
    try:
        rmse = float(np.sqrt(mean_squared_error(trues, preds)))
        mae = float(mean_absolute_error(trues, preds))
        pcc = pearsonr(trues, preds)[0]
        ccc_score = float(ccc(trues, preds))
        loss_std = float(np.std(np.abs(np.array(trues) - np.array(preds))))

        scatter_dir = os.path.join(plot_dir, "scatter")
        csv_dir = os.path.join(plot_dir, "csv")
        os.makedirs(scatter_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.scatter(trues, preds, alpha=0.7)
        plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Test Subject {test_idx} - {task_name}')
        plt.grid(True)
        plt.savefig(os.path.join(scatter_dir, f"{prefix}_scatter.png"))
        plt.close()

        pd.DataFrame({'True': trues, 'Predicted': preds}).to_csv(
            os.path.join(csv_dir, f"{prefix}_true_vs_pred.csv"), index=False)

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'PCC': pcc,
            'CCC': ccc_score,
            'Loss_STD': loss_std,
            'Params': params
        }

        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, f'{prefix}_metrics.json'), 'w', encoding='utf-8') as jf:
            json.dump(metrics, jf, indent=2)

        with open(os.path.join(model_dir, f'intermediate_results_{task_name}.txt'), 'a', encoding='utf-8') as f:
            f.write(f"\n{prefix}\n")
            f.write(f"MAE:  {mae:.4f}\nRMSE: {rmse:.4f}\nPCC:  {pcc:.4f}\nCCC:  {ccc_score:.4f}\n")
            f.write(f"Loss STD: {loss_std:.4f}\n")
            f.write(f"Params: {params}\n" + '-' * 40 + '\n')

    except Exception as e:
        print(f"[WARNING] Failed to save intermediate results for {prefix}: {e}")

# Evaluation function
def evaluate_model(task_name, full_dataset, n, device, trial, epoch, mean_target, std_target, batch_size):
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr
    def concordance_corr_coef(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) < 2 or len(y_pred) < 2:
            return np.nan
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        corr = pearsonr(y_true, y_pred)[0]
        ccc = (2 * corr * np.sqrt(var_true) * np.sqrt(var_pred)) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        return ccc

    trial_save_dir = os.path.join(gcn_checkpoints_dir, f"{task_name}", f"trial_{trial}", f"epoch_{epoch}")
    print(f"testing on :: {trial_save_dir}")
    total_rmse_list, all_predicted_values, all_actual_values = [], [], []

    for idx in range(n):
        checkpoint_path = os.path.join(trial_save_dir, f"subject_{idx}_model.pt")
        checkpoint = torch.load(checkpoint_path)
        model_state_dict, lr, dr, hidden_classes = checkpoint['model_state_dict'], checkpoint['learning rate'], checkpoint['dropout'], checkpoint['hidden_classes']
        print(f"learning rate: {lr}, dropout: {dr}, hidden_classes: {hidden_classes}")
        model_GCN = GCN(node_features=240, hidden_channels=hidden_classes, num_classes=1).to(device)
        model_GCN.load_state_dict(model_state_dict)
        model_GCN.eval()
        test_dataset = [full_dataset[idx].to(device)]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        predicted_values_i, actual_values_i = [], []

        with torch.no_grad():
            for data in test_loader:
                output_test, target_test = model_GCN(data, dropout_rate=dr).cpu().numpy().flatten(), data.y.cpu().numpy().flatten()
                predicted_values_i.extend(output_test.tolist())
                actual_values_i.extend(target_test.tolist())

        predicted_values = denormalize_output(np.array(predicted_values_i), mean_target, std_target)
        actual_values = denormalize_output(np.array(actual_values_i), mean_target, std_target)
        mse = mean_squared_error(actual_values, predicted_values)
        subject_rmse = np.sqrt(mse)
        total_rmse_list.append(subject_rmse)
        all_predicted_values.extend(predicted_values)
        all_actual_values.extend(actual_values)

    final_rmse = np.mean(total_rmse_list)
    X, Y = np.array(all_predicted_values), np.array(all_actual_values)
    pcc, _ = pearsonr(X, Y)
    ccc = concordance_corr_coef(X, Y)

    output_file_path = os.path.join(gcn_results_dir, f"{task_name}", f"{task_name}_results_{trial}.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w") as f:
        f.write(f"Task Name: {task_name}\nFinal Test RMSE: {final_rmse:.4f}\n"
                f"Pearson Correlation Coefficient (PCC): {pcc:.4f}\n"
                f"Concordance Correlation Coefficient (CCC): {ccc:.4f}")

    print(f"RMSE: {final_rmse}\nPCC: {pcc}\nCCC: {ccc}\nResults saved to {output_file_path}")

# Objective function
best_trial_dir = None
best_trial_loss = float('-inf')

def objective_ADAM_early(trial):
    global best_trial_dir, best_trial_loss
    lr = trial.suggest_float("lr", 5e-4, 1e-2)
    dr = trial.suggest_float("dropout", 0.1, 0.8)
    hidden_classes = trial.suggest_int('hidden_classes', 16, 180)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop'])
    print(f"learning rate:{lr},dropout: {dr}, hidden_classes: {hidden_classes}, optimizer: {optimizer_name}")

    trial_save_dir = os.path.join(gcn_checkpoints_dir, f"{task_name}", f"trial_{trial.number}")
    trial_plot_dir = os.path.join(gcn_plots_dir, f"{task_name}", f"trial_{trial.number}")
    os.makedirs(trial_save_dir, exist_ok=True)
    os.makedirs(trial_plot_dir, exist_ok=True)

    val_loss = train_valid_loop_ADAM_early(
        lr=lr,
        dr=dr,
        hidden_classes=hidden_classes,
        optimizer_name=optimizer_name,
        trial_save_dir=trial_save_dir,
        trial_plot_dir=trial_plot_dir,
        batch_size=batch_size,
        n_subjects=n,
        trial=trial
    )
    return float(val_loss)


def train_valid_loop_ADAM_early(lr, dr, hidden_classes, optimizer_name, trial_save_dir, trial_plot_dir, batch_size, n_subjects, trial):
    num_epochs = 100
    patience = 3
    all_train_losses = [[] for _ in range(n)]
    all_valid_losses = [[] for _ in range(n)]
    avg_train_losses = []
    all_val_losses = []
    best_stop_epoch = -1
    best_val_loss = float('inf')
    early_stop_count = 0
    y_pred_all, y_true_all = [], []  
    val_optuna = []
    for test_idx in range(n):
        best_val_mae = float('inf')
        best_model_state = None
        best_train_losses = None
        best_val_losses = None
        print(f"Training for test subject {test_idx} - {task_name}")
        # Inner loop for validation subject (not equal to test)
        for val_idx in range(n):
            if val_idx == test_idx:
                continue
            train_indices = [i for i in range(n) if i != test_idx and i != val_idx]
            train_dataset = [full_dataset[i].to(device) for i in train_indices]
            val_dataset = [full_dataset[val_idx].to(device)]
            test_dataset = [full_dataset[test_idx].to(device)]

            model_GCN = GCN(node_features=240, hidden_channels=hidden_classes, num_classes=1).to(device)
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(model_GCN.parameters(), lr=lr)
            elif optimizer_name == 'adamw':
                optimizer = torch.optim.AdamW(model_GCN.parameters(), lr=lr)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(model_GCN.parameters(), lr=lr)
            elif optimizer_name == 'rmsprop':
                optimizer = torch.optim.RMSprop(model_GCN.parameters(), lr=lr)
            else:
                optimizer = torch.optim.Adam(model_GCN.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
            criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            train_losses = []
            val_losses = []
            early_stop = 0
            best_val = float('inf')
            best_state = None
            best_val = float('inf')  
            for epoch in range(num_epochs):
                # Training
                for batch in train_loader:
                    batch = batch.to(device)
                train_loss, model_GCN, optimizer = train_epoch(train_loader, dr, optimizer, criterion, model_GCN, scheduler)
                train_losses.append(train_loss)
                # Validation
                for batch in val_loader:
                    batch = batch.to(device)
                val_loss = valid_epoch(val_loader, dr, criterion, model_GCN)
                val_losses.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(model_GCN.state_dict())
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= patience:
                    break
            # Track best validation for this test_idx
            if best_val < best_val_mae:
                best_val_mae = best_val
                best_model_state = best_state
                checkpoint_path = os.path.join(trial_save_dir, f"test_{test_idx}_best_model.pt")
                torch.save({
                    'model_state_dict': best_state,
                    'learning rate': lr,
                    'dropout': dr,
                    'hidden_classes': hidden_classes,
                    'optimizer': optimizer_name
                }, checkpoint_path)
                best_train_losses = train_losses.copy()
                best_val_losses = val_losses.copy()
        # After all val_idx, test on test_idx with best model
        val_optuna.append(best_val)
        if best_model_state is not None and best_train_losses is not None:
            model_GCN = GCN(node_features=240, hidden_channels=hidden_classes, num_classes=1).to(device)
            checkpoint_path = os.path.join(trial_save_dir, f"test_{test_idx}_best_model.pt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  
            checkpoint = torch.load(checkpoint_path)
            model_GCN.load_state_dict(checkpoint['model_state_dict'])
            test_loader = DataLoader([full_dataset[test_idx].to(device)], batch_size=batch_size, shuffle=False)
            for batch in test_loader:
                batch = batch.to(device)
            all_train_losses[test_idx] = best_train_losses if best_train_losses is not None else []
            all_valid_losses[test_idx] = best_val_losses if best_val_losses is not None else []
            all_val_losses.append(best_val_mae)
        else:
            all_train_losses[test_idx] = []
            all_val_losses.append(float('inf'))

    # Plotting and aggregation (as before)
    min_len = min(len(l) for l in all_train_losses if len(l) > 0)
    avg_train_losses = np.mean([l[:min_len] for l in all_train_losses if len(l) > 0], axis=0)
    rows = (n // 6) + (1 if n % 6 != 0 else 0)
    cols = min(n, 6)
    fig_loss, axs_loss = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    if rows == 1 and cols == 1:
        axs_loss = np.array([axs_loss])
    elif rows == 1 or cols == 1:
        axs_loss = np.ravel(axs_loss)
    else:
        axs_loss = axs_loss.flatten()
    for idx in range(n):
        ax_loss = axs_loss[idx]
        ax_loss.plot(range(len(all_train_losses[idx])), all_train_losses[idx], label='Train', color='red')
        ax_loss.plot(range(len(all_valid_losses[idx])), all_valid_losses[idx], label='Val', color='green')
        ax_loss.set_title(f"Loss - Subject {idx}-{task_name}")
        ax_loss.legend()
    for i in range(n, len(axs_loss)):
        fig_loss.delaxes(axs_loss[i])
    fig_loss.tight_layout()
    fig, ax = plt.subplots(figsize=(10, 6))
    if all_val_losses and avg_train_losses is not None:
        ax.plot(range(len(all_val_losses)), all_val_losses, label='Validation Loss', color='orange')
        ax.plot(range(len(avg_train_losses)), avg_train_losses, label='Training Loss', color='blue')
    ax.set_title('Train + Val Loss vs. Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
    plt.tight_layout()
    fig_loss.savefig(os.path.join(trial_plot_dir, "training_loss.png"))
    fig.savefig(os.path.join(trial_plot_dir, "validation_plots.png"))
    print("Final validation loss:", np.mean(all_val_losses))

    # After all test subjects, evaluate each subject with its best model and aggregate predictions
    y_pred_all, y_true_all = [], []
    for test_idx in range(n):
        if all_train_losses[test_idx]:  # Only if a model was trained for this subject
            model_GCN = GCN(node_features=240, hidden_channels=hidden_classes, num_classes=1).to(device)
            checkpoint_path = os.path.join(trial_save_dir, f"test_{test_idx}_best_model.pt")
            checkpoint = torch.load(checkpoint_path)
            model_GCN.load_state_dict(checkpoint['model_state_dict'])
            test_loader = DataLoader([full_dataset[test_idx].to(device)], batch_size=batch_size, shuffle=False)
            y_pred, y_true = [], []
            model_GCN.eval()
            with torch.no_grad():
                for data in test_loader:
                    output, target = model_GCN(data, dropout_rate=dr).cpu().numpy().flatten(), data.y.cpu().numpy().flatten()
                    y_pred.extend(denormalize_output(np.array(output), mean_target, std_target))
                    y_true.extend(denormalize_output(np.array(target), mean_target, std_target))
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)
    print(f"Total predictions collected: {len(y_pred_all)}")
    save_intermediate_gcn(
        np.mean(val_optuna),
        y_pred_all,
        y_true_all,
        {'lr': lr, 'dropout': dr, 'hidden_classes': hidden_classes, 'optimizer': optimizer_name},
        task_name,
        trial_plot_dir,
        trial_save_dir,
        trial_num=trial.number
    )
    return float(np.mean(val_optuna))
# Main block
if __name__ == "__main__":
    study_dir = os.path.join(gcn_study_dir, task_name)
    os.makedirs(study_dir, exist_ok=True)
    study_path = os.path.join(study_dir, f"optuna_study_{task_name}.db")
    storage_str = f"sqlite:///{study_path}"

    save_dir = os.path.join(study_dir, "checkpoints")
    plot_dir = os.path.join(study_dir, "plots")
    optuna_dir = os.path.join(study_dir, "optuna")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(optuna_dir, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"gcn_early_{task_name}",
        storage=storage_str,
        load_if_exists=True,
        sampler=TPESampler(seed=seed_value)
    )
    print(f"lenoftrials :  {len(study.trials)}")
    num_trials_remaining = n_trials
    if len(study.trials) > 0:
        last_finished = max(
            [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            default=None
        )
        if last_finished is not None:
            print(f"Previous run found, continuing from trial: {last_finished+1}")
            num_trials_remaining = max(0, n_trials - (last_finished + 1))
    print("num trials", num_trials_remaining)
    study.optimize(objective_ADAM_early, n_trials=num_trials_remaining)
    print(study.best_trial)