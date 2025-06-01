import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import optuna
from torch.nn.functional import smooth_l1_loss
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import random
import signal

# Task setup
# Task setup
num_trials = 20
#task_index = int(sys.argv[1])
task_index = 1
task_names = ['2h', '2s', '3', '4_P1', '4_P2', '4_P3', '5']
task_name = task_names[task_index]

# Data loading
audio_data = np.load(f'/rds/user/sss77/hpc-work/New/MLModels/Data/Audio/training_data_combined_{task_name}.npy')
if task_name in ['3', '5']:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/values_for3,5.npy')
    print(f"task is 3 or 5: {task_name}")
else:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/labels.npy')
labels = labels.astype(float)

X_combined = audio_data
n_subjects = len(labels)

# Directories
model_dir = f'/rds/user/sss77/hpc-work/New/NN/audio/'
plot_dir = f'/rds/user/sss77/hpc-work/New/NN/audio/plots_{task_name}/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir + 'models/', exist_ok=True)
os.makedirs(plot_dir + 'csv/', exist_ok=True)
os.makedirs(f"{plot_dir}/loss_plots/", exist_ok=True)
os.makedirs(f"{plot_dir}/subjectwise_loss/", exist_ok=True)
os.makedirs(model_dir+ 'optuna/', exist_ok=True)

# Create or load the Optuna study
study_path = f"{model_dir}/optuna/optuna_study_{task_name}.db"
storage_str = f"sqlite:///{study_path}"

def handler(signum, frame):
    """Handle termination signals."""
    print("Termination signal received. Exiting gracefully...")
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

#pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
study = optuna.create_study(
    direction="minimize",
    study_name=f"ef_{task_name}",
    storage=storage_str,
    load_if_exists=True,
    #pruner=pruner
)

print(f"lenoftrails :  {len(study.trials)}")

# Check if we are resuming from a previous run
if len(study.trials) > 0:
    # Get the highest trial number among completed trials
    last_finished = max(
        [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        default=None
    )
    if last_finished is not None:
        # Print the next trial number to indicate continuation
        print(f"Previous run found, continuing from trial: {last_finished+1}")

        num_trials = max(0, num_trials - (last_finished + 1))
print("num trails",num_trials)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CCC metric
def ccc(y_true, y_pred):
    """Calculate Concordance Correlation Coefficient."""
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.cov(y_true, y_pred)[0][1]
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)

# Neural Network
class MLP(nn.Module):
    """Multi-Layer Perceptron model."""
    def __init__(self, input_dim, hidden_dims, dropout):
        super(MLP, self).__init__()
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

results = {}
best_mse_global = float('inf')

def save_intermediate(trial_num, mae, preds, trues, trial_params, task_name, plot_dir, model_dir):
    global best_mse_global
    rmse = np.sqrt(mean_squared_error(trues, preds))

    if rmse < best_mse_global:
        best_mse_global = rmse
        
    pcc, _ = pearsonr(trues, preds)
    ccc_score = ccc(trues, preds)
    loss_std = np.std(np.abs(np.array(trues) - np.array(preds)))

    # Plot scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(trues, preds, alpha=0.7)
    plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'Trial {trial_num} - {task_name}')
    plt.grid(True)
    plt.savefig(f"{plot_dir}/trial_{trial_num}_scatter.png")
    plt.close()

    # Save CSV
    pd.DataFrame({'True': trues, 'Predicted': preds}).to_csv(
        f"{plot_dir}/csv/trial_{trial_num}_true_vs_pred.csv", index=False)

    # Write results
    with open(os.path.join(model_dir, f'intermediate_best_results_{task_name}.txt'), 'a') as f:
        f.write(f"\nTrial {trial_num}\n")
        f.write(f"MAE:  {mae:.4f}\nRMSE: {rmse:.4f}\nPCC:  {pcc:.4f}\nCCC:  {ccc_score:.4f}\n")
        f.write(f"Loss STD: {loss_std:.4f}\n")
        f.write(f"Best RMSE_Global: {best_mse_global:.4f}\n")
        f.write(f"Params: {trial_params}\n" + '-' * 40 + '\n')
        
        #best_mae_global = mae  # Update best MAE after saving

def plot_average_losses(avg_train_losses, avg_val_losses, trial_num, plot_dir, min_len):
    """Plot average training and validation losses across subjects."""
    # Plot average training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_len), avg_train_losses, label='Avg Training Loss', color='blue')
    plt.title('Average Training Loss Across Subjects')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_plots/trial_{trial_num}_avg_train_loss.png")
    plt.close()
    
    # Plot average validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_len), avg_val_losses, label='Avg Validation Loss', color='orange')
    plt.title('Average Validation Loss Across Subjects')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_plots/trial_{trial_num}_avg_val_loss.png")
    plt.close()

def plot_subjectwise_losses(all_train_losses, all_val_losses, n_subjects, trial_num, plot_dir):
    """Plot subject-wise training and validation losses."""
    rows = (n_subjects // 6) + (1 if n_subjects % 6 != 0 else 0)
    cols = min(n_subjects, 6)
    
    # Training losses
    fig_train, axs_train = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    axs_train = np.ravel(axs_train) if rows * cols > 1 else [axs_train]
    
    for idx in range(n_subjects):
        ax = axs_train[idx]
        ax.plot(all_train_losses[idx], label='Train Loss', color='red')
        ax.set_title(f"Subject {idx} - Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
    
    for i in range(n_subjects, len(axs_train)):
        fig_train.delaxes(axs_train[i])
        
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/subjectwise_loss/trial_{trial_num}_subjectwise_train_loss.png")
    plt.close()
    
    # Validation losses
    fig_val, axs_val = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    axs_val = np.ravel(axs_val) if rows * cols > 1 else [axs_val]
    
    for idx in range(n_subjects):
        ax = axs_val[idx]
        if len(all_val_losses[idx]) > 0:
            ax.plot(all_val_losses[idx], label='Val Loss', color='green')
            ax.set_title(f"Subject {idx} - Validation Loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()
    
    for i in range(n_subjects, len(axs_val)):
        fig_val.delaxes(axs_val[i])
        
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/subjectwise_loss/trial_{trial_num}_subjectwise_val_loss.png")
    plt.close()

def objective(trial):
    """Objective function for Optuna optimization."""
    # Hyperparameter suggestions
    hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
    start_exp = trial.suggest_int("start_exp", 4 + hidden_layers - 1, 9)
    exponents = sorted(random.sample(range(4, start_exp + 1), hidden_layers), reverse=True)
    hidden_dims = [2 ** exp for exp in exponents]
    patience = trial.suggest_int("patience", 2, 7)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    epochs = 100
    opt_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd", "rmsprop"])
    
    # Print hyperparameters
    print("hidden_layers:", hidden_layers)
    print("start_exp:", start_exp)
    print("exponents:", exponents)
    print("hidden_dims:", hidden_dims)
    print("patience:", patience)
    print("dropout:", dropout)
    print("lr:", lr)
    print("weight_decay:", weight_decay)
    print("optimizer: ", opt_name)
    
    val_maes = []
    all_test_preds = []
    all_test_true = []
    fold_models = []
    all_train_losses = [[] for _ in range(n_subjects)]
    all_val_losses = [[] for _ in range(n_subjects)]
    step_counter = 0
    cal_optuna = []
    val_optuna = []
    # Leave-one-out cross-validation loop
    for test_id in range(n_subjects):
        X_test = X_combined[test_id:test_id + 1]
        y_test = labels[test_id:test_id + 1]
        train_val_ids = [i for i in range(n_subjects) if i != test_id]
        best_model_state = None
        best_scalers = None
        train_losses_inner = []
        val_maes_inner = []
        #print(test_id)
        # Nested cross-validation
        for val_id in train_val_ids:
            #print(f"           val :  {val_id}")
            train_ids = [i for i in train_val_ids if i != val_id]
            X_train = X_combined[train_ids]
            y_train = labels[train_ids]
            X_val = X_combined[val_id:val_id + 1]
            y_val = labels[val_id:val_id + 1]
            
            # Scaling
            scaler_X = StandardScaler().fit(X_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
            X_train_scaled = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32).to(device)
            y_train_scaled = torch.tensor(scaler_y.transform(y_train.reshape(-1, 1)), dtype=torch.float32).to(device)
            X_val_scaled = torch.tensor(scaler_X.transform(X_val), dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            
            # Model creation
            model = MLP(X_train.shape[1], hidden_dims, dropout).to(device)
            
            # Optimizer selection
            if opt_name == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt_name == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt_name == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            elif opt_name == "rmsprop":
                optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
                
            criterion = nn.L1Loss()
            best_mae_val = float('inf')
            counter = 0
            
            train_losses = []
            val_losses = []
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output = model(X_train_scaled)
                loss = criterion(output, y_train_scaled)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    pred_val_scaled = model(X_val_scaled).cpu().numpy()
                    pred_val = scaler_y.inverse_transform(pred_val_scaled)
                    mae_val = mean_squared_error(y_val_tensor.cpu().numpy(), pred_val)
                
                val_losses.append(mae_val)
                loss_epoch_val = mae_val
                # Pruning check
                #trial.report(mae_val, step_counter)
                #if trial.should_prune():
                #    print(f"[Pruning] Trial {trial.number} | Epoch: {epoch} | subject: {test_id} | validation: {val_id} | step_cnt: {step_counter} | Validation Loss: {mae_val:.4f}")
                #    raise optuna.exceptions.TrialPruned()
                    
                step_counter += 1
                
                # Early stopping
                if mae_val < best_mae_val:
                    best_mae_val = mae_val
                    best_model_state = model.state_dict()
                    torch.save(model.state_dict(), f"{plot_dir}/models/subject_{test_id}_model.pt")
                    best_scalers = (scaler_X, scaler_y)
                    counter = 0
                else:
                    counter += 1
                    
                if counter >= patience:
                    break
                    
            train_losses_inner.append(train_losses)
            val_maes_inner.append(val_losses)
            val_optuna.append(best_mae_val)
            
        
        # Select best fold
        print(f"len of val-optuna: {len(val_optuna)}")
        best_fold_idx = np.argmin([np.mean(v) for v in val_maes_inner])
        all_train_losses[test_id] = train_losses_inner[best_fold_idx]
        all_val_losses[test_id] = val_maes_inner[best_fold_idx]
        val_maes.append(np.mean([np.mean(v) for v in val_maes_inner]))
        
        # Test phase
        final_model = MLP(X_combined.shape[1], hidden_dims, dropout).to(device)
        final_model.load_state_dict(best_model_state)
        fold_models.append((final_model, best_scalers))
        
        scaler_X, scaler_y = best_scalers
        X_test_scaled = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32).to(device)
        
        final_model.eval()
        with torch.no_grad():
            pred_test_scaled = final_model(X_test_scaled).cpu().numpy()
            pred_test = scaler_y.inverse_transform(pred_test_scaled).ravel()
            all_test_preds.append(pred_test[0])
            all_test_true.append(y_test[0])
    
    # Calculate average losses across subjects
    min_len = min(len(l) for l in all_train_losses if len(l) > 0)
    avg_train_losses = np.mean([l[:min_len] for l in all_train_losses if len(l) > 0], axis=0)
    avg_val_losses = np.mean([l[:min_len] for l in all_val_losses if len(l) > 0], axis=0)
    
    # Generate plots
    plot_average_losses(avg_train_losses, avg_val_losses, trial.number, plot_dir, min_len)
    plot_subjectwise_losses(all_train_losses, all_val_losses, n_subjects, trial.number, plot_dir)
    
    # Save results
    results[trial.number] = (np.mean(val_maes), np.array(all_test_preds), np.array(all_test_true))
        #save_intermediate(trial.number, np.mean(val_maes), all_test_preds, all_test_true, trial.params, task_name, plot_dir, model_dir)
    save_intermediate(trial.number, np.mean(val_optuna) , all_test_preds, all_test_true, trial.params, task_name, plot_dir, model_dir)
    #print(f"{trial.number}=== {np.mean(val_optuna)} ====== {np.std(val_optuna)}")
    return np.mean(val_optuna) #+ 0.5 * np.std(val_optuna) #np.mean(val_optuna)

# Run Optuna optimization
study.optimize(objective, num_trials)

#=============================================================================
output_dir = "/rds/user/sss77/hpc-work/New/NN/"
os.makedirs(output_dir, exist_ok=True)  # Ensures the directory exists
output_path = os.path.join(output_dir, "audio_completed.txt")

with open(output_path, "a") as f:
    f.write(f"{task_name}\n")
#=============================================================================

best_trial_number = study.best_trial.number
best_mse, all_test_preds, all_test_true = results[best_trial_number]

rmse = np.sqrt(mean_squared_error(all_test_true, all_test_preds))
mae = mean_absolute_error(all_test_true, all_test_preds)
pcc, _ = pearsonr(all_test_true, all_test_preds)
ccc_score = ccc(all_test_true, all_test_preds)

print(f"\nFinal Test Results for {task_name}:")
print(f"RMSE: {rmse:.4f}\nMAE:  {mae:.4f}\nPCC:  {pcc:.4f}\nCCC:  {ccc_score:.4f}")

# Save results and plot
plt.figure(figsize=(8, 6))
plt.scatter(all_test_true, all_test_preds, alpha=0.7)
plt.plot([min(all_test_true), max(all_test_true)],
         [min(all_test_true), max(all_test_true)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'True vs Predicted - {task_name}')
plt.grid(True)
plt.savefig(f"{plot_dir}/true_vs_pred_{task_name}.png")
plt.close()

df = pd.DataFrame({'True': all_test_true, 'Predicted': all_test_preds})
df.to_csv(os.path.join(model_dir, f'true_vs_pred_{task_name}.csv'), index=False)

with open(os.path.join(model_dir, f'results_nn_{task_name}.txt'), 'a') as f:
    f.write(f"\nTask: {task_name}\n")
    f.write(f"Best MAE (Optuna): {study.best_value:.4f}\n")
    f.write(f"Best Hyperparameters: {study.best_trial.params}\n")
    f.write(f"Final Test RMSE: {rmse:.4f}\n")
    f.write(f"Final Test MAE:  {mae:.4f}\n")
    f.write(f"Final Test PCC:  {pcc:.4f}\n")
    f.write(f"Final Test CCC:  {ccc_score:.4f}\n")
    f.write("-" * 40 + "\n")