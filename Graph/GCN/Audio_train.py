import os
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch
import copy
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FuncFormatter
from torch.optim.lr_scheduler import StepLR
import optuna
import argparse
from optuna.samplers import TPESampler
from preprocessing import set_seed
from GCN_model import GCN, create_graph_list_A, train_epoch, valid_epoch


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

import argparse

def ccc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

def save_intermediate_gcn(obj_loss, preds, trues, params, task_name, plot_dir, model_dir, trial_num):
    global best_rmse, best_obj_loss, BASE_DIR
    rmse = np.sqrt(mean_squared_error(trues, preds))
    if rmse < best_rmse:
        best_rmse = rmse
    # Update best objective loss
    if obj_loss < best_obj_loss:
        best_obj_loss = obj_loss
        
    mae = mean_absolute_error(trues, preds)
    pcc, _ = pearsonr(trues, preds)
    ccc_score = ccc(trues, preds)
    loss_std = np.std(np.abs(np.array(trues) - np.array(preds)))

    scatter_path = os.path.join(plot_dir, "scatter")
    csv_path = os.path.join(plot_dir, "csv")
    os.makedirs(scatter_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(trues, preds, alpha=0.7)
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
        f.write(f"\nTest Subject {trial_num}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nPCC: {pcc:.4f}\nCCC: {ccc_score:.4f}\nLoss STD: {loss_std:.4f}\n best RMSE : {best_rmse:.4f}\n best obj_loss : {best_obj_loss:.4f}")
        f.write(f"Objective loss: {obj_loss:.4f}\nParams: {params}\n{'-' * 40}\n")

#====================================================================================================================================
##################################################  Adam Train ######################################################################
#====================================================================================================================================

def train_valid_loop_ADAM(lr, dr, hidden_classes, trial_subdir, plot_subdir, task_name, batch_size, optimizer_name, trial):
    num_epochs = 100
    patience = 4
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
                checkpoint_path = os.path.join(trial_subdir, f"test_{test_idx}_best_model.pt")
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
            checkpoint_path = os.path.join(trial_subdir, f"test_{test_idx}_best_model.pt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  
            checkpoint = torch.load(checkpoint_path)
            model_GCN.load_state_dict(checkpoint['model_state_dict'])
            test_loader = DataLoader([full_dataset[test_idx].to(device)], batch_size=batch_size, shuffle=False)
            for batch in test_loader:
                batch = batch.to(device)
            
            all_train_losses[test_idx] = best_train_losses
            all_valid_losses[test_idx] = best_val_losses
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
    fig_loss.savefig(os.path.join(plot_subdir+"/"+f"training_loss.png"))
    fig.savefig(os.path.join(plot_subdir+"/"+f"validation_plots.png"))
    print("Final validation loss:", np.mean(all_val_losses))

    # After all test subjects, evaluate each subject with its best model and aggregate predictions
    y_pred_all, y_true_all = [], []
    for test_idx in range(n):
        if all_train_losses[test_idx]:  # Only if a model was trained for this subject
            model_GCN = GCN(node_features=240, hidden_channels=hidden_classes, num_classes=1).to(device)
            checkpoint_path = os.path.join(trial_subdir,f"test_{test_idx}_best_model.pt")
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
        plot_subdir,
        trial_subdir,
        trial_num=trial.number
    )
    return np.mean(val_optuna), None

#====================================================================================================================================
################################################## Test Epoch #######################################################################
#====================================================================================================================================

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def evaluate_model(task_name, full_dataset, n, device, trial, epoch, mean_target, std_target, batch_size):
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
        return (2 * corr * np.sqrt(var_true) * np.sqrt(var_pred)) / (
            var_true + var_pred + (mean_true - mean_pred) ** 2
        )

    print(f"[EVAL] Task: {task_name} | Trial: {trial} ")
    
    save_dir = os.path.join(BASE_DIR, f"GCN_checkpoints_{task_name}", f"trial_{trial}")
    output_file_path = os.path.join(BASE_DIR, task_name, f"{task_name}_results_{trial}.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    total_rmse_list, all_predicted_values, all_actual_values = [], [], []

    for idx in range(n):
        checkpoint_path = os.path.join(save_dir, f"test_{idx}_best_model.pt")
        if not os.path.exists(checkpoint_path):
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']
        lr = checkpoint['learning rate']
        dr = checkpoint['dropout']
        hidden_classes = checkpoint['hidden_classes']

        model_GCN = GCN(node_features=240, hidden_channels=hidden_classes, num_classes=1).to(device)
        model_GCN.load_state_dict(model_state_dict)
        model_GCN.eval()

        test_dataset = [full_dataset[idx].to(device)]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predicted_values_i, actual_values_i = [], []
        with torch.no_grad():
            for data in test_loader:
                output_test = model_GCN(data, dropout_rate=dr).cpu().numpy().flatten()
                target_test = data.y.cpu().numpy().flatten()
                predicted_values_i.extend(output_test.tolist())
                actual_values_i.extend(target_test.tolist())

        predicted_values = denormalize_output(np.array(predicted_values_i), mean_target, std_target)
        actual_values = denormalize_output(np.array(actual_values_i), mean_target, std_target)

        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        total_rmse_list.append(rmse)
        all_predicted_values.extend(predicted_values)
        all_actual_values.extend(actual_values)

    final_rmse = np.mean(total_rmse_list)
    pcc, _ = pearsonr(all_predicted_values, all_actual_values)
    ccc_val = concordance_corr_coef(all_predicted_values, all_actual_values)

    with open(output_file_path, "w", encoding='utf-8') as f:
        f.write(f"Task Name: {task_name}\n")
        f.write(f"Final Test RMSE: {final_rmse:.4f}\n")
        f.write(f"Pearson Correlation Coefficient (PCC): {pcc:.4f}\n")
        f.write(f"Concordance Correlation Coefficient (CCC): {ccc_val:.4f}\n")

    print(f"[RESULTS] RMSE: {final_rmse:.4f} | PCC: {pcc:.4f} | CCC: {ccc_val:.4f}")
    print(f"[SAVED] Results to: {output_file_path}")

#====================================================================================================================================
################################################## Epoch Train ######################################################################
#====================================================================================================================================



def objective_ADAM(trial):
    lr = trial.suggest_float("lr", 5e-4, 1e-2)
    dr = trial.suggest_float("dropout", 0.1, 0.3)
    hidden_classes_num = trial.suggest_int('hidden_classes_num', 4, 8)
    hidden_classes = 2 ** hidden_classes_num
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adamw'])
    best_trial_loss = float('inf')
    best_trial_dir = None

    print(f"[TRIAL {trial.number}] LR: {lr}, Dropout: {dr}, Hidden: {hidden_classes}, Optimizer: {optimizer_name}")

    save_dir = os.path.join(BASE_DIR, f"GCN_checkpoints_{task_name}", f"trial_{trial.number}")
    plot_dir = os.path.join(BASE_DIR, f"GCN_plots_{task_name}", f"trial_{trial.number}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    val_loss, _ = train_valid_loop_ADAM(lr, dr, hidden_classes, save_dir, plot_dir, task_name, batch_size, optimizer_name, trial)
    if val_loss < best_trial_loss:
      best_trial_loss = val_loss
      remove_trial_subdir= os.path.join(save_dir,f"trial_{best_trial_dir}")
      best_trial_loss= trial.number
    evaluate_model(
        task_name=task_name,
        full_dataset=full_dataset,
        n=n,
        device=device,
        trial=best_trial_loss,
        epoch=0,
        mean_target=mean_target,
        std_target=std_target,
        batch_size=batch_size
    )

    return float(val_loss)


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

global BASE_DIR

BASE_DIR = f"/rds/user/sss77/hpc-work/New/Graph/GCN_A/{task_name}"
task_dir_A = os.path.join("/rds/user/sss77/hpc-work/New/Graph/Graph_Data/Modify_Audio_Data", task_name)
task_csv_files_A = sorted(os.listdir(task_dir_A))
study_dir = os.path.join(BASE_DIR, f"GCN_study_{task_name}")
os.makedirs(study_dir, exist_ok=True)

# Use study_dir for all trial, plot, and checkpoint directories
save_dir = os.path.join(study_dir, "checkpoints")
plot_dir = os.path.join(study_dir, "plots")
optuna_dir = os.path.join(study_dir, "optuna")
study_path = os.path.join(optuna_dir, f"optuna_study_{task_name}.db")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(optuna_dir, exist_ok=True)

# Print them out
print(f"[INFO] Batch Size      : {batch_size}")
print(f"[INFO] Num Trials      : {n_trials}")
print(f"[INFO] Task Name (A)   : {task_name}")
print(f"[INFO] Task Name (V)   : {taskv_name}")

global best_obj_loss 
global best_rmse

best_obj_loss = float('inf')
best_rmse = float('inf')

task_csv_files_A=os.listdir(task_dir_A)
task_csv_files_A.sort()
n = len(task_csv_files_A)
print(f"n1 = {n}")
G_list_MEFG_A = create_graph_list_A(task_dir_A,task_csv_files_A,n)
print(f"n2 = {len(G_list_MEFG_A)}")

if n == 39:
  y_train_i = torch.tensor(labels, dtype=torch.float32)
else :
  print("check1")
  y_train_i = torch.tensor(values, dtype=torch.float32)
  
mean_target = y_train_i.mean()
std_target = y_train_i.std()

y_train = normalize_target(y_train_i, mean_target, std_target)

full_dataset = []

for idx in range(0 ,n):
  data=Data(x=G_list_MEFG_A[idx].ndata['feature'],  edge_index=torch.stack([G_list_MEFG_A[idx].all_edges()[0], G_list_MEFG_A[idx].all_edges()[1]], dim=0) ,edge_attr=G_list_MEFG_A[idx].edata['feature'],y=y_train[idx])
  full_dataset.append(data)
  

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name=f"gcn_{task_name}",
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        sampler=TPESampler(seed=seed_value)
    )

    existing_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[INFO] Existing Trials: {existing_trials}")
    study.optimize(objective_ADAM, n_trials=max(0, n_trials - existing_trials))

    print(f"[INFO] Best Trial: {study.best_trial}")