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
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Task setup
task_index = int(sys.argv[1])
task_names = ['2h', '2s', '3', '4_P1', '4_P2', '4_P3', '5']
task_name = task_names[task_index]

# Data loading
video_data = np.load(f'/rds/user/sss77/hpc-work/New/MLModels/Data/Video/training_data_combined_{task_name}.npy')
if task_name in ['3', '5']:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/values_for3,5.npy')
else:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/labels.npy')
labels = labels.astype(float)

X_combined = video_data
n_subjects = len(labels)

# Directories
model_dir = f'/rds/user/sss77/hpc-work/New/NN/video/'
plot_dir = f'/rds/user/sss77/hpc-work/New/NN/video/plots_{task_name}/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir + 'models/', exist_ok=True)
os.makedirs(plot_dir + 'csv/', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CCC metric
def ccc(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.cov(y_true, y_pred)[0][1]
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)

# Neural Network
class MLP(nn.Module):
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
best_mae_global = float('inf')

def save_intermediate(trial_num, mae, preds, trues, trial_params):
    global best_mae_global
    if mae < best_mae_global:
        best_mae_global = mae
        rmse = np.sqrt(mean_squared_error(trues, preds))
        pcc, _ = pearsonr(trues, preds)
        ccc_score = ccc(trues, preds)

        # Plot
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

        # Save trial log
        with open(os.path.join(model_dir, f'intermediate_best_results_{task_name}.txt'), 'a') as f:
            f.write(f"\nTrial {trial_num}\n")
            f.write(f"MAE:  {mae:.4f}\nRMSE: {rmse:.4f}\nPCC:  {pcc:.4f}\nCCC:  {ccc_score:.4f}\n")
            f.write(f"Params: {trial_params}\n" + '-' * 40 + '\n')

def objective(trial):
    hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    epochs = 100

    val_maes = []
    all_test_preds = []
    all_test_true = []
    fold_models = []

    for test_id in range(n_subjects):
        X_test = X_combined[test_id:test_id+1]
        y_test = labels[test_id:test_id+1]
        train_val_ids = [i for i in range(n_subjects) if i != test_id]

        best_val_mae = float('inf')
        best_model_state = None
        best_scalers = None

        for val_id in train_val_ids:
            train_ids = [i for i in train_val_ids if i != val_id]

            X_train = X_combined[train_ids]
            X_val = X_combined[val_id:val_id+1]
            y_train = labels[train_ids]
            y_val = labels[val_id:val_id+1]

            scaler_X = StandardScaler().fit(X_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

            X_train_scaled = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32).to(device)
            y_train_scaled = torch.tensor(scaler_y.transform(y_train.reshape(-1, 1)), dtype=torch.float32).to(device)
            X_val_scaled = torch.tensor(scaler_X.transform(X_val), dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

            model = MLP(X_train.shape[1], [hidden_size]*hidden_layers, dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()

            best_mae_val = float('inf')
            patience, counter = 10, 0

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output = model(X_train_scaled)
                loss = criterion(output, y_train_scaled)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred_val_scaled = model(X_val_scaled).cpu().numpy()
                    pred_val = scaler_y.inverse_transform(pred_val_scaled)
                    mae_val = mean_absolute_error(y_val.cpu().numpy(), pred_val)

                if mae_val < best_mae_val:
                    best_mae_val = mae_val
                    best_model_state = model.state_dict()
                    best_scalers = (scaler_X, scaler_y)
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    break

        val_maes.append(best_mae_val)
        final_model = MLP(X_combined.shape[1], [hidden_size]*hidden_layers, dropout).to(device)
        final_model.load_state_dict(best_model_state)
        fold_models.append((final_model, best_scalers))

        # Test
        scaler_X, scaler_y = best_scalers
        X_test_scaled = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32).to(device)
        final_model.eval()
        with torch.no_grad():
            pred_test_scaled = final_model(X_test_scaled).cpu().numpy()
            pred_test = scaler_y.inverse_transform(pred_test_scaled).ravel()
            all_test_preds.append(pred_test[0])
            all_test_true.append(y_test[0])

    results[trial.number] = (np.mean(val_maes), np.array(all_test_preds), np.array(all_test_true))
    save_intermediate(trial.number, np.mean(val_maes), all_test_preds, all_test_true, trial.params)

    return np.mean(val_maes)

# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_trial_number = study.best_trial.number
best_mae, all_test_preds, all_test_true = results[best_trial_number]

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