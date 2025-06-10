import os
import copy
import shutil
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader  # Updated import
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import optuna
from GCN_model import GCN, train_epoch, valid_epoch
from preprocessing import set_seed, create_graph_list_V, create_graph_list_A

labels=[13,4,19,14,3,3,2,4,3,7,1,5,4,0,7,0,3,4,2,13,8,0,0,2,2,1,6,3,2,3,4,2,4,2,0,7,1,13,2]
values=[5,5,13,4,19,14,3,3,2,4,3,7,1,5,4,0,7,0,3,4,2,13,8,0,0,2,2,1,6,3,2,3,4,2,4,2,0,7,1,13,2]

def normalize_target(target, mean_target, std_target):
    return (target - mean_target) / std_target

def denormalize_output(output, mean, std):
    if isinstance(mean, torch.Tensor):
        mean = mean.item()
    if isinstance(std, torch.Tensor):
        std = std.item()
    return (output * std) + mean

seed_value = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device', device)

"""import argparse
parser = argparse.ArgumentParser(description="Train GCN model for Late Fusion regression")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
parser.add_argument("--n_trials", type=int, required=True, help="Number of Optuna trials")
parser.add_argument("--task_name", type=str, required=True, help="Task name for audio (e.g., Task2h_A)")
args = parser.parse_args()

batch_size = args.batch_size
n_trials = args.n_trials
task_name = args.task_name
taskv_name = task_name.replace("Task", "Task_").replace("_A", "")
"""
set_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device', device)

batch_size = 4
n_trials = 2
taskv_name = "Task_4_P3"
task_name = "Task4_P3_A"

BASE_DIR = f"/kaggle/working/GCN_LF/{task_name}"
task_dir_V = os.path.join("/kaggle/input/graph-data/Modify_Data", taskv_name)
task_dir_A = os.path.join("/kaggle/input/graph-data/Modify_Audio_Data", task_name)

task_csv_files_V = sorted(os.listdir(task_dir_V))
task_csv_files_A = sorted(os.listdir(task_dir_A))

study_dir = os.path.join(BASE_DIR, f"GCN_study_{task_name}")
os.makedirs(study_dir, exist_ok=True)
save_dir = os.path.join(study_dir, "checkpoints")
plot_dir = os.path.join(study_dir, "plots")
optuna_dir = os.path.join(study_dir, "optuna")
study_path = os.path.join(optuna_dir, f"optuna_study_{task_name}.db")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(optuna_dir, exist_ok=True)

print(f"[INFO] Batch Size      : {batch_size}")
print(f"[INFO] Num Trials      : {n_trials}")
print(f"[INFO] Task Name (A)   : {task_name}")
print(f"[INFO] Task Name (V)   : {taskv_name}")

# Assume create_graph_list_V and create_graph_list_A are defined elsewhere
# For example:
# from data_utils import create_graph_list_V, create_graph_list_A

n = 6  # Manually set number of subjects

if n == 39:
    y_train_i = torch.tensor(labels, dtype=torch.float32)
else:
    print("check1")
    y_train_i = torch.tensor(values, dtype=torch.float32)

mean_target = y_train_i.mean()
std_target = y_train_i.std()
y_train = normalize_target(y_train_i, mean_target, std_target)

# Load graphs
G_list_MEFG_V = create_graph_list_V(task_dir_V, task_csv_files_V, n)
G_list_MEFG_A = create_graph_list_A(task_dir_A, task_csv_files_A, n)

# Create separate datasets for each modality
full_dataset_V = []
full_dataset_A = []

for idx in range(n):
    data_v = Data(
        x=G_list_MEFG_V[idx].ndata['feature'],
        edge_index=torch.stack(G_list_MEFG_V[idx].all_edges(), dim=0),
        edge_attr=G_list_MEFG_V[idx].edata['feature'],
        y=y_train[idx]
    )
    full_dataset_V.append(data_v)

    data_a = Data(
        x=G_list_MEFG_A[idx].ndata['feature'],
        edge_index=torch.stack(G_list_MEFG_A[idx].all_edges(), dim=0),
        edge_attr=G_list_MEFG_A[idx].edata['feature'],
        y=y_train[idx]
    )
    full_dataset_A.append(data_a)

full_dataset = []
for idx in range(n):
    data_v = full_dataset_V[idx]
    data_a = full_dataset_A[idx]
    # Combine or store them together as a tuple or custom object
    full_dataset.append((data_v, data_a))
#====================================================================================================================================
##################################################  Adam Train ######################################################################
#====================================================================================================================================

# --- Utility: CCC and Save Intermediate (Late Fusion) ---
def ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

best_rmse = float('inf')
best_obj_loss = float('inf')
def save_intermediate_gcn(obj_loss, preds, trues, params, task_name, plot_dir, model_dir, trial_num):
    global best_rmse, best_obj_loss, BASE_DIR
    rmse = np.sqrt(mean_squared_error(trues, preds)) if len(trues) > 0 and len(preds) > 0 else np.nan
    if rmse < best_rmse:
        best_rmse = rmse
    # Update best objective loss
    if obj_loss < best_obj_loss:
        best_obj_loss = obj_loss
    mae = mean_absolute_error(trues, preds) if len(trues) > 0 and len(preds) > 0 else np.nan
    if len(trues) >= 2 and len(preds) >= 2:
        pcc, _ = pearsonr(trues, preds)
        ccc_score = ccc(trues, preds)
    else:
        pcc = np.nan
        ccc_score = np.nan
    loss_std = np.std(np.abs(np.array(trues) - np.array(preds))) if len(trues) > 0 and len(preds) > 0 else np.nan
    scatter_path = os.path.join(plot_dir, "scatter")
    csv_path = os.path.join(plot_dir, "csv")
    os.makedirs(scatter_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(trues, preds, alpha=0.7)
    if len(trues) > 0 and len(preds) > 0:
        plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Scatterplot')
    plt.grid(True)
    plt.savefig(os.path.join(scatter_path, f"test_{trial_num}_scatter.png"))
    plt.close()
    pd.DataFrame({'True': trues, 'Predicted': preds}).to_csv(
        os.path.join(csv_path, f"test_{trial_num}_true_vs_pred.csv"), index=False
    )
    with open(os.path.join(BASE_DIR, f'intermediate_results.txt'), 'a', encoding='utf-8') as f:
        f.write(f"\nTest Subject {trial_num}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nPCC: {pcc}\nCCC: {ccc_score}\nLoss STD: {loss_std:.4f}\n best RMSE : {best_rmse:.4f}\n best obj_loss : {best_obj_loss:.4f}")
        f.write(f"Objective loss: {obj_loss:.4f}\nParams: {params}\n{'-' * 40}\n")

def train_valid_loop_ADAM_late(lr1, dr1, hidden_classes1, optimizer1, lr2, dr2, hidden_classes2, optimizer2, trial_subdir, plot_subdir, task_name, batch_size, trial):
    num_epochs = 100
    patience = 4
    all_train_losses_v = [[] for _ in range(n)]
    all_train_losses_a = [[] for _ in range(n)]
    all_valid_losses = [[] for _ in range(n)]
    avg_train_losses_v = []
    avg_train_losses_a = []
    best_stop_epoch = -1
    best_val_loss = float('inf')
    early_stop_count = 0
    y_pred_all, y_true_all = [], []  
    val_optuna = []
    for test_idx in range(n):
        best_val_mae = float('inf')
        best_model_state_v = None
        best_model_state_a = None
        best_train_losses_v = None
        best_train_losses_a = None
        best_val_losses = None
        print(f"Training for test subject {test_idx} - {task_name}")
        # Inner loop for validation subject (not equal to test)
        for val_idx in range(n):
            if val_idx == test_idx:
                continue
            train_indices = [i for i in range(n) if i != test_idx and i != val_idx]
            train_dataset_v = [full_dataset_V[i].to(device) for i in train_indices]
            val_dataset_v = [full_dataset_V[val_idx].to(device)]
            train_dataset_a = [full_dataset_A[i].to(device) for i in train_indices]
            val_dataset_a = [full_dataset_A[val_idx].to(device)]

            model_GCN_v = GCN(node_features=240, hidden_channels=hidden_classes1, num_classes=1).to(device)
            model_GCN_a = GCN(node_features=240, hidden_channels=hidden_classes2, num_classes=1).to(device)
            # Video optimizer
            if optimizer1 == 'adam':
                optimizer_v = torch.optim.Adam(model_GCN_v.parameters(), lr=lr1)
            elif optimizer1 == 'adamw':
                optimizer_v = torch.optim.AdamW(model_GCN_v.parameters(), lr=lr1)
            elif optimizer1 == 'sgd':
                optimizer_v = torch.optim.SGD(model_GCN_v.parameters(), lr=lr1)
            elif optimizer1 == 'rmsprop':
                optimizer_v = torch.optim.RMSprop(model_GCN_v.parameters(), lr=lr1)
            else:
                optimizer_v = torch.optim.Adam(model_GCN_v.parameters(), lr=lr1)
            # Audio optimizer
            if optimizer2 == 'adam':
                optimizer_a = torch.optim.Adam(model_GCN_a.parameters(), lr=lr2)
            elif optimizer2 == 'adamw':
                optimizer_a = torch.optim.AdamW(model_GCN_a.parameters(), lr=lr2)
            elif optimizer2 == 'sgd':
                optimizer_a = torch.optim.SGD(model_GCN_a.parameters(), lr=lr2)
            elif optimizer2 == 'rmsprop':
                optimizer_a = torch.optim.RMSprop(model_GCN_a.parameters(), lr=lr2)
            else:
                optimizer_a = torch.optim.Adam(model_GCN_a.parameters(), lr=lr2)
            scheduler_v = StepLR(optimizer_v, step_size=20, gamma=0.9)
            scheduler_a = StepLR(optimizer_a, step_size=20, gamma=0.9)
            criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)

            train_loader_v = DataLoader(train_dataset_v, batch_size=batch_size, shuffle=True)
            val_loader_v = DataLoader(val_dataset_v, batch_size=batch_size, shuffle=False)
            train_loader_a = DataLoader(train_dataset_a, batch_size=batch_size, shuffle=True)
            val_loader_a = DataLoader(val_dataset_a, batch_size=batch_size, shuffle=False)

            train_losses_v = []
            train_losses_a = []
            val_losses = []
            early_stop = 0
            best_val = float('inf')
            best_state_v = None
            best_state_a = None
            for epoch in range(num_epochs):
                # Training
                for batch in train_loader_v:
                    batch = batch.to(device)
                for batch in train_loader_a:
                    batch = batch.to(device)
                # Train both GCNs separately
                train_loss_v, model_GCN_v, optimizer_v = train_epoch(train_loader_v, dr1, optimizer_v, criterion, model_GCN_v, scheduler_v)
                train_loss_a, model_GCN_a, optimizer_a = train_epoch(train_loader_a, dr2, optimizer_a, criterion, model_GCN_a, scheduler_a)
                train_losses_v.append(train_loss_v)
                train_losses_a.append(train_loss_a)
                # Validation
                for batch in val_loader_v:
                    batch = batch.to(device)
                for batch in val_loader_a:
                    batch = batch.to(device)
                val_loss_v = valid_epoch(val_loader_v, dr1, criterion, model_GCN_v)
                val_loss_a = valid_epoch(val_loader_a, dr2, criterion, model_GCN_a)
                val_loss = (val_loss_v + val_loss_a) / 2.0
                val_losses.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state_v = copy.deepcopy(model_GCN_v.state_dict())
                    best_state_a = copy.deepcopy(model_GCN_a.state_dict())
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= patience:
                    break
            # Track best validation for this test_idx
            if best_val < best_val_mae:
                best_val_mae = best_val
                best_model_state_v = best_state_v
                best_model_state_a = best_state_a
                checkpoint_path_v = os.path.join(trial_subdir, f"test_{test_idx}_best_model_v.pt")
                checkpoint_path_a = os.path.join(trial_subdir, f"test_{test_idx}_best_model_a.pt")
                torch.save({
                    'model_state_dict': best_state_v,
                    'learning rate': lr1,
                    'dropout': dr1,
                    'hidden_classes': hidden_classes1,
                    'optimizer': optimizer1
                }, checkpoint_path_v)
                torch.save({
                    'model_state_dict': best_state_a,
                    'learning rate': lr2,
                    'dropout': dr2,
                    'hidden_classes': hidden_classes2,
                    'optimizer': optimizer2
                }, checkpoint_path_a)
                best_train_losses_v = train_losses_v.copy()
                best_train_losses_a = train_losses_a.copy()
                best_val_losses = val_losses.copy()
        # After all val_idx, test on test_idx with best models
        val_optuna.append(best_val_mae)
        if best_model_state_v is not None and best_model_state_a is not None and best_train_losses_v is not None and best_train_losses_a is not None:
            model_GCN_v = GCN(node_features=240, hidden_channels=hidden_classes1, num_classes=1).to(device)
            model_GCN_a = GCN(node_features=240, hidden_channels=hidden_classes2, num_classes=1).to(device)
            checkpoint_path_v = os.path.join(trial_subdir, f"test_{test_idx}_best_model_v.pt")
            checkpoint_path_a = os.path.join(trial_subdir, f"test_{test_idx}_best_model_a.pt")
            os.makedirs(os.path.dirname(checkpoint_path_v), exist_ok=True)  
            os.makedirs(os.path.dirname(checkpoint_path_a), exist_ok=True)  
            checkpoint_v = torch.load(checkpoint_path_v)
            checkpoint_a = torch.load(checkpoint_path_a)
            model_GCN_v.load_state_dict(checkpoint_v['model_state_dict'])
            model_GCN_a.load_state_dict(checkpoint_a['model_state_dict'])
            test_loader_v = DataLoader([full_dataset_V[test_idx].to(device)], batch_size=batch_size, shuffle=False)
            test_loader_a = DataLoader([full_dataset_A[test_idx].to(device)], batch_size=batch_size, shuffle=False)
            for batch_v, batch_a in zip(test_loader_v, test_loader_a):
                batch_v = batch_v.to(device)
                batch_a = batch_a.to(device)
            all_train_losses_v[test_idx] = best_train_losses_v
            all_train_losses_a[test_idx] = best_train_losses_a
            all_valid_losses[test_idx] = best_val_losses
        else:
            all_train_losses_v[test_idx] = []
            all_train_losses_a[test_idx] = []
            all_valid_losses[test_idx] = []
            val_optuna[-1] = float('inf')

    # Plotting and aggregation (as before)
    min_len_v = min(len(l) for l in all_train_losses_v if len(l) > 0)
    min_len_a = min(len(l) for l in all_train_losses_a if len(l) > 0)
    avg_train_losses_v = np.mean([l[:min_len_v] for l in all_train_losses_v if len(l) > 0], axis=0)
    avg_train_losses_a = np.mean([l[:min_len_a] for l in all_train_losses_a if len(l) > 0], axis=0)
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
        ax_loss.plot(range(len(all_train_losses_v[idx])), all_train_losses_v[idx], label='Train Video', color='red')
        ax_loss.plot(range(len(all_train_losses_a[idx])), all_train_losses_a[idx], label='Train Audio', color='blue')
        ax_loss.plot(range(len(all_valid_losses[idx])), all_valid_losses[idx], label='Val', color='green')
        ax_loss.set_title(f"Loss - Subject {idx}-{task_name}")
        ax_loss.legend()
    for i in range(n, len(axs_loss)):
        fig_loss.delaxes(axs_loss[i])
    fig_loss.tight_layout()
    fig, ax = plt.subplots(figsize=(10, 6))
    if all_valid_losses and avg_train_losses_v is not None and avg_train_losses_a is not None:
        ax.plot(range(len(val_optuna)), val_optuna, label='Validation Loss', color='orange')
        ax.plot(range(len(avg_train_losses_v)), avg_train_losses_v, label='Avg Train Loss Video', color='red')
        ax.plot(range(len(avg_train_losses_a)), avg_train_losses_a, label='Avg Train Loss Audio', color='blue')
    ax.set_title('Train + Val Loss vs. Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
    plt.tight_layout()
    fig_loss.savefig(os.path.join(plot_subdir+"/"+f"training_loss.png"))
    fig.savefig(os.path.join(plot_subdir+"/"+f"validation_plots.png"))
    print("Final validation loss:", np.mean(val_optuna))

    # After all test subjects, evaluate each subject with its best model and aggregate predictions
    y_pred_all, y_true_all = [], []
    for test_idx in range(n):
        if all_train_losses_v[test_idx] and all_train_losses_a[test_idx]:  # Only if a model was trained for this subject
            model_GCN_v = GCN(node_features=240, hidden_channels=hidden_classes1, num_classes=1).to(device)
            model_GCN_a = GCN(node_features=240, hidden_channels=hidden_classes2, num_classes=1).to(device)
            checkpoint_path_v = os.path.join(trial_subdir, f"test_{test_idx}_best_model_v.pt")
            checkpoint_path_a = os.path.join(trial_subdir, f"test_{test_idx}_best_model_a.pt")
            checkpoint_v = torch.load(checkpoint_path_v)
            checkpoint_a = torch.load(checkpoint_path_a)
            model_GCN_v.load_state_dict(checkpoint_v['model_state_dict'])
            model_GCN_a.load_state_dict(checkpoint_a['model_state_dict'])
            test_loader_v = DataLoader([full_dataset_V[test_idx].to(device)], batch_size=batch_size, shuffle=False)
            test_loader_a = DataLoader([full_dataset_A[test_idx].to(device)], batch_size=batch_size, shuffle=False)
            y_pred, y_true = [], []
            model_GCN_v.eval()
            model_GCN_a.eval()
            with torch.no_grad():
                for data_v, data_a in zip(test_loader_v, test_loader_a):
                    output_v, target_v = model_GCN_v(data_v, dropout_rate=dr1).cpu().numpy().flatten(), data_v.y.cpu().numpy().flatten()
                    output_a, target_a = model_GCN_a(data_a, dropout_rate=dr2).cpu().numpy().flatten(), data_a.y.cpu().numpy().flatten()
                    pred_v = denormalize_output(np.array(output_v), mean_target, std_target)
                    pred_a = denormalize_output(np.array(output_a), mean_target, std_target)
                    prediction = (pred_v + pred_a) / 2.0
                    target = (denormalize_output(np.array(target_v), mean_target, std_target) + denormalize_output(np.array(target_a), mean_target, std_target)) / 2.0
                    y_pred.extend(prediction.tolist())
                    y_true.extend(target.tolist())
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)
    print(f"Total predictions collected: {len(y_pred_all)}")
    save_intermediate_gcn(
        np.mean(val_optuna),
        y_pred_all,
        y_true_all,
        {'lr1': lr1, 'dropout1': dr1, 'hidden_classes1': hidden_classes1, 'optimizer1': optimizer1, 'lr2': lr2, 'dropout2': dr2, 'hidden_classes2': hidden_classes2, 'optimizer2': optimizer2},
        task_name,
        plot_subdir,
        trial_subdir,
        trial_num=trial.number
    )
    return np.mean(val_optuna), None
#====================================================================================================================================
################################################## Test Epoch #######################################################################
#====================================================================================================================================

# Function to compute Concordance Correlation Coefficient (CCC)

def evaluate_model_late(task_name, full_dataset_V, full_dataset_A, n, device, trial, mean_target, std_target, batch_size, trial_subdir):
    def ccc(y_true, y_pred):
        mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
        var_true, var_pred = np.var(y_true), np.var(y_pred)
        covariance = np.cov(y_true, y_pred)[0][1]
        return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    all_preds, all_targets, total_rmse_list = [], [], []
    for idx in range(n):
        # Load best model checkpoints for this subject
        ckpt1_pth = os.path.join(trial_subdir, f"Model1_subject_{idx}_best.pt")
        checkpoint1 = torch.load(ckpt1_pth, map_location=device)
        model_state_dict1, dr1, hidden_classes1 = checkpoint1['model_state_dict1'], checkpoint1['dropout1'], checkpoint1['hidden_classes1']
        model_GCN1 = GCN(node_features=240, hidden_channels=hidden_classes1, num_classes=1).to(device)
        model_GCN1.load_state_dict(model_state_dict1)
        model_GCN1.eval()
        ckpt2_pth = os.path.join(trial_subdir, f"Model2_subject_{idx}_best.pt")
        checkpoint2 = torch.load(ckpt2_pth, map_location=device)
        model_state_dict2, dr2, hidden_classes2 = checkpoint2['model_state_dict2'], checkpoint2['dropout2'], checkpoint2['hidden_classes2']
        model_GCN2 = GCN(node_features=240, hidden_channels=hidden_classes2, num_classes=1).to(device)
        model_GCN2.load_state_dict(model_state_dict2)
        model_GCN2.eval()
        test_dataset1 = [full_dataset_V[idx].to(device)]
        test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)
        test_dataset2 = [full_dataset_A[idx].to(device)]
        test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)
        prediction = None
        target = None
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
                target = (target1 + target2) / 2.0
                if prediction is not None and target is not None:
                    all_preds.extend(prediction.tolist())
                    all_targets.extend(target.tolist())
            if prediction is not None and target is not None:
                mse = mean_squared_error(target, prediction)
                rmse = np.sqrt(mse)
                total_rmse_list.append(rmse)
    final_rmse = np.mean(total_rmse_list) if total_rmse_list else float('nan')
    pcc, _ = pearsonr(all_preds, all_targets)
    ccc_score = ccc(all_targets, all_preds)
    output_file = f"/rds/user/sss77/hpc-work/GCN_Late/{task_name}/{task_name}_results_{trial}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(f"Task Name: {task_name}\n")
        f.write(f"Final Test RMSE: {final_rmse:.4f}\n")
        f.write(f"PCC: {pcc:.4f}\n")
        f.write(f"CCC: {ccc_score:.4f}\n")
    print(f"RMSE: {final_rmse:.4f}, PCC: {pcc:.4f}, CCC: {ccc_score:.4f}\nResults saved to: {output_file}")

#====================================================================================================================================
################################################## Epoch Train ######################################################################
#====================================================================================================================================

best_trial_dir = None
best_trial_loss = float('-inf')
def objective_ADAM_late(trial):
    lr1 = trial.suggest_float("lr1", 1e-5, 1e-2)
    lr2 = trial.suggest_float("lr2", 1e-5, 1e-2)
    dr1 = trial.suggest_float("dropout1", 0.1, 0.8)
    dr2 = trial.suggest_float("dropout2", 0.1, 0.8)
    hidden_classes1 = trial.suggest_int("hidden_classes1", 16, 180)
    hidden_classes2 = trial.suggest_int("hidden_classes2", 16, 180)
    optimizer1 = trial.suggest_categorical('optimizer1', ['adam', 'adamw', 'sgd', 'rmsprop'])
    optimizer2 = trial.suggest_categorical('optimizer2', ['adam', 'adamw', 'sgd', 'rmsprop'])
    print(f"lr1: {lr1}, dr1: {dr1}, hidden_classes1: {hidden_classes1}, optimizer1: {optimizer1}")
    print(f"lr2: {lr2}, dr2: {dr2}, hidden_classes2: {hidden_classes2}, optimizer2: {optimizer2}")

    trial_save_dir = os.path.join(BASE_DIR, f"GCN_checkpoints_{task_name}", f"trial_{trial.number}")
    trial_plot_dir = os.path.join(BASE_DIR, f"GCN_plots_{task_name}", f"trial_{trial.number}")
    os.makedirs(trial_save_dir, exist_ok=True)
    os.makedirs(trial_plot_dir, exist_ok=True)

    val_loss, _ = train_valid_loop_ADAM_late(
        lr1=lr1,
        dr1=dr1,
        hidden_classes1=hidden_classes1,
        optimizer1=optimizer1,
        lr2=lr2,
        dr2=dr2,
        hidden_classes2=hidden_classes2,
        optimizer2=optimizer2,
        trial_subdir=trial_save_dir,
        plot_subdir=trial_plot_dir,
        task_name=task_name,
        batch_size=batch_size,
        trial=trial
    )
    return float(val_loss)

# --- Optuna DB and Output Directory Setup ---
study_dir = os.path.join(BASE_DIR, f"GCN_study_{task_name}")
os.makedirs(study_dir, exist_ok=True)
study_path = os.path.join(study_dir, f"optuna_study_{task_name}.db")
storage_str = f"sqlite:///{study_path}"
# Use study_dir for all trial, plot, and checkpoint directories
save_dir = os.path.join(study_dir, "checkpoints")
plot_dir = os.path.join(study_dir, "plots")
optuna_dir = os.path.join(study_dir, "optuna")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(optuna_dir, exist_ok=True)
n = 6 #33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

if n == 39:
    y_train_i = torch.tensor(labels, dtype=torch.float32)
else:
    print("check1")
    y_train_i = torch.tensor(values, dtype=torch.float32)

mean_target = y_train_i.mean()
std_target = y_train_i.std()

y_train = normalize_target(y_train_i, mean_target, std_target)
# Prepare full_dataset_V and full_dataset_A for late fusion
G_list_MEFG_V = create_graph_list_V(task_dir_V, task_csv_files_V, n)
G_list_MEFG_A = create_graph_list_A(task_dir_A, task_csv_files_A, n)
full_dataset_V = []
full_dataset_A = []
for idx in range(0, n):
    data_v = Data(x=G_list_MEFG_V[idx].ndata['feature'],
                  edge_index=torch.stack([G_list_MEFG_V[idx].all_edges()[0], G_list_MEFG_V[idx].all_edges()[1]], dim=0),
                  edge_attr=G_list_MEFG_V[idx].edata['feature'],
                  y=y_train[idx])
    data_a = Data(x=G_list_MEFG_A[idx].ndata['feature'],
                  edge_index=torch.stack([G_list_MEFG_A[idx].all_edges()[0], G_list_MEFG_A[idx].all_edges()[1]], dim=0),
                  edge_attr=G_list_MEFG_A[idx].edata['feature'],
                  y=y_train[idx])
    full_dataset_V.append(data_v)
    full_dataset_A.append(data_a)

class LateFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_v_list, data_a_list):
        assert len(data_v_list) == len(data_a_list)
        self.data_v_list = data_v_list
        self.data_a_list = data_a_list

    def __len__(self):
        return len(self.data_v_list)

    def __getitem__(self, idx):
        return self.data_v_list[idx], self.data_a_list[idx]

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name=f"gcn_late_{task_name}",
        storage=storage_str,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed_value)
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
    study.optimize(objective_ADAM_late, n_trials=num_trials_remaining)
    print(study.best_trial)
    # Write completion log and best trial summary (like LSTM_LF.py)
    output_dir = f"/kaggle/working/GCN_LF/{task_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "LateFusion_completed.txt")
    with open(output_path, "a") as f:
        f.write(f"{task_name}\n")
    print(f"Best trial for task {task_name}:")
    print(f"RMSE: {study.best_value:.4f}")
    print(f"Params: {study.best_trial.params}")
