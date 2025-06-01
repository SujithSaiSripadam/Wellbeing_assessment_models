import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import optuna
import sys
import signal
import torch
import torch.nn as nn
import torch.optim as optim

results = {}
best_mse_global = float('inf')

def pad_sequence(sequences, max_length):
    return np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant') for seq in sequences])

def ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

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

def plot_average_losses(avg_train_losses, avg_val_losses, trial_num, plot_dir, min_len):
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_len), avg_train_losses, label='Avg Training Loss', color='blue')
    plt.title('Average Training Loss Across Subjects')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_plots/trial_{trial_num}_avg_train_loss.png")
    plt.close()
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
    rows = (n_subjects // 6) + (1 if n_subjects % 6 != 0 else 0)
    cols = min(n_subjects, 6)
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

def build_ef_model(audio_dim, video_dim, lstm_units, dropout_rate):
    class EarlyFusionLSTM(nn.Module):
        def __init__(self, audio_dim, video_dim, lstm_units, dropout_rate):
            super().__init__()
            self.lstm = nn.LSTM(audio_dim + video_dim, lstm_units, batch_first=True)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(lstm_units, 1)
        def forward(self, audio, video):
            x = torch.cat([audio, video], dim=2)
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.fc(out)
            return out
    return EarlyFusionLSTM(audio_dim, video_dim, lstm_units, dropout_rate)

def objective(trial, task_name, audio_data, video_data, labels, plot_dir, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_audio_length = max(len(seq) for seq in audio_data)
    max_video_length = max(len(seq) for seq in video_data)
    audio_data = pad_sequence(audio_data, max_audio_length)
    video_data = pad_sequence(video_data, max_video_length)
    audio_scaler = StandardScaler()
    video_scaler = StandardScaler()
    label_scaler = StandardScaler()
    n_subjects = len(audio_data)
    y_true_all, y_pred_all = [], []
    all_train_losses = [[] for _ in range(n_subjects)]
    all_val_losses = [[] for _ in range(n_subjects)]
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop'])
    num_epochs = 20
    batch_size = 16
    for test_index in range(n_subjects):
        X_test_audio = audio_data[test_index:test_index+1]
        X_test_video = video_data[test_index:test_index+1]
        y_test = labels[test_index:test_index+1]
        train_val_ids = [i for i in range(n_subjects) if i != test_index]
        best_mae_val = float('inf')
        best_state = None
        best_train_losses = []
        best_val_losses = []
        best_weights = None
        best_train_losses_final = []
        best_val_losses_final = []
        for val_index in train_val_ids:
            train_ids = [i for i in train_val_ids if i != val_index]
            X_train_audio = audio_data[train_ids]
            X_val_audio = audio_data[val_index:val_index+1]
            X_train_video = video_data[train_ids]
            X_val_video = video_data[val_index:val_index+1]
            y_train = labels[train_ids]
            y_val = labels[val_index:val_index+1]
            audio_scaler.fit(X_train_audio.reshape(-1, X_train_audio.shape[-1]))
            video_scaler.fit(X_train_video.reshape(-1, X_train_video.shape[-1]))
            label_scaler.fit(y_train.reshape(-1, 1))
            X_train_audio_scaled = audio_scaler.transform(X_train_audio.reshape(-1, X_train_audio.shape[-1])).reshape(X_train_audio.shape)
            X_val_audio_scaled = audio_scaler.transform(X_val_audio.reshape(-1, X_val_audio.shape[-1])).reshape(X_val_audio.shape)
            X_train_video_scaled = video_scaler.transform(X_train_video.reshape(-1, X_train_video.shape[-1])).reshape(X_train_video.shape)
            X_val_video_scaled = video_scaler.transform(X_val_video.reshape(-1, X_val_video.shape[-1])).reshape(X_val_video.shape)
            y_train_scaled = label_scaler.transform(y_train.reshape(-1, 1))
            y_val_scaled = label_scaler.transform(y_val.reshape(-1, 1))
            X_train_audio_tensor = torch.tensor(X_train_audio_scaled, dtype=torch.float32).to(device)
            X_val_audio_tensor = torch.tensor(X_val_audio_scaled, dtype=torch.float32).to(device)
            X_train_video_tensor = torch.tensor(X_train_video_scaled, dtype=torch.float32).to(device)
            X_val_video_tensor = torch.tensor(X_val_video_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
            model = build_ef_model(X_train_audio_tensor.shape[2], X_train_video_tensor.shape[2], lstm_units, dropout_rate).to(device)
            if optimizer_name == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            criterion = nn.SmoothL1Loss()
            train_losses = []
            val_losses = []
            best_val = float('inf')
            patience = 3
            patience_counter = 0
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                output = model(X_train_audio_tensor, X_train_video_tensor)
                loss = criterion(output, y_train_tensor)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                model.eval()
                with torch.no_grad():
                    val_output = model(X_val_audio_tensor, X_val_video_tensor)
                    val_loss = criterion(val_output, y_val_tensor)
                    val_losses.append(val_loss.item())
                    val_pred = label_scaler.inverse_transform(val_output.cpu().numpy())
                    val_true = label_scaler.inverse_transform(y_val_tensor.cpu().numpy())
                    mae_val = mean_absolute_error(val_true, val_pred)
                if mae_val < best_val:
                    best_val = mae_val
                    best_state = model.state_dict()
                    best_train_losses = train_losses.copy()
                    best_val_losses = val_losses.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
            if best_val < best_mae_val:
                best_mae_val = best_val
                best_weights = best_state
                best_train_losses_final = best_train_losses
                best_val_losses_final = best_val_losses
        if best_mae_val < float('inf') and best_weights is not None:
            model = build_ef_model(X_test_audio.shape[2], X_test_video.shape[2], lstm_units, dropout_rate).to(device)
            model.load_state_dict(best_weights)
            model.eval()
            X_test_audio_scaled = audio_scaler.transform(X_test_audio.reshape(-1, X_test_audio.shape[-1])).reshape(X_test_audio.shape)
            X_test_video_scaled = video_scaler.transform(X_test_video.reshape(-1, X_test_video.shape[-1])).reshape(X_test_video.shape)
            X_test_audio_tensor = torch.tensor(X_test_audio_scaled, dtype=torch.float32).to(device)
            X_test_video_tensor = torch.tensor(X_test_video_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                y_test_pred_scaled = model(X_test_audio_tensor, X_test_video_tensor).cpu().numpy()
            y_test_pred = label_scaler.inverse_transform(y_test_pred_scaled)
            y_true_all.append(y_test[0])
            y_pred_all.append(y_test_pred.ravel()[0])
            all_train_losses[test_index] = best_train_losses_final
            all_val_losses[test_index] = best_val_losses_final
            torch.save(best_weights, os.path.join(model_dir, f'model_subject_{test_index}.pt'))
        else:
            all_train_losses[test_index] = []
            all_val_losses[test_index] = []
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    min_len = min(len(l) for l in all_train_losses if len(l) > 0)
    avg_train_losses = np.mean([l[:min_len] for l in all_train_losses if len(l) > 0], axis=0)
    avg_val_losses = np.mean([l[:min_len] for l in all_val_losses if len(l) > 0], axis=0)
    plot_average_losses(avg_train_losses, avg_val_losses, trial.number, plot_dir, min_len)
    plot_subjectwise_losses(all_train_losses, all_val_losses, n_subjects, trial.number, plot_dir)
    save_intermediate(trial.number, mean_absolute_error(y_true_all, y_pred_all), y_pred_all, y_true_all, trial.params, task_name, plot_dir, model_dir)
    return mean_absolute_error(y_true_all, y_pred_all)
# Tasks list

task_index = int(sys.argv[1])
task_names = ['2h', '2s', '3', '4_P1', '4_P2', '4_P3', '5']
task_name = task_names[task_index]

print(f"Running task: {task_name}")
audio_data = np.load(f'/rds/user/sss77/hpc-work/New/LSTM/Data/Audio/training_data_{task_name}.npy', allow_pickle=True)
video_data = np.load(f'/rds/user/sss77/hpc-work/New/LSTM/Data/Video/training_data_{task_name}.npy', allow_pickle=True)
if task_name in ['3', '5']:
    labels = np.load('/rds/user/sss77/hpc-work/New/LSTM/Data/values_for3,5.npy')
else:
    labels = np.load('/rds/user/sss77/hpc-work/New/LSTM/Data/labels.npy')
labels = labels.astype(float)


model_dir = f'/rds/user/sss77/hpc-work/New/LSTM/EF/task_{task_name}'
plot_dir = f'/rds/user/sss77/hpc-work/New/LSTM/EF/task_{task_name}/Plots_{task_name}'
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Remove TensorFlow GPU check and related code
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create extra directories
os.makedirs(f"{plot_dir}/loss_plots/", exist_ok=True)
os.makedirs(f"{plot_dir}/subjectwise_loss/", exist_ok=True)
os.makedirs(f"{plot_dir}/csv/", exist_ok=True)
os.makedirs(model_dir + '/optuna/', exist_ok=True)

# Optuna study setup
study_path = f"{model_dir}/optuna/optuna_study_{task_name}.db"
storage_str = f"sqlite:///{study_path}"

def handler(signum, frame):
    """Handle termination signals."""
    print("Termination signal received. Exiting gracefully...")
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

num_trials = 30  # Default, can be changed

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
print("num trails", num_trials)

study.optimize(lambda trial: objective(trial, task_name, audio_data, video_data, labels, plot_dir, model_dir), n_trials=num_trials)

print(f"Best trial for task {task_name}:\nMAE: {study.best_value:.4f}\nParams: {study.best_trial.params}")
#=============================================================================
output_dir = "/rds/user/sss77/hpc-work/New/LSTM/"
os.makedirs(output_dir, exist_ok=True)  # Ensures the directory exists
output_path = os.path.join(output_dir, "EF_completed.txt")

with open(output_path, "a") as f:
    f.write(f"{task_name}\n")
#=============================================================================
with open(os.path.join(model_dir, f'best_trial_{task_name}.txt'), 'w') as f:
    f.write(f"Best MAE: {study.best_value:.4f}\n")
    f.write(f"Best Parameters: {study.best_trial.params}\n")
