import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
task_index = int(sys.argv[1])

# ---------------------------
# Task List
# ---------------------------
task_names = ['2h', '2s', '3', '4_P1', '4_P2', '4_P3', '5']

# ---------------------------
# Task Selection
# ---------------------------
task_name = task_names[task_index]
print(f"\nRunning Random Forest + Optuna for task: {task_name}")

# Load data
audio_data = np.load(f'/rds/user/sss77/hpc-work/New/MLModels/Data/Audio/training_data_combined_{task_name}.npy')
video_data = np.load(f'/rds/user/sss77/hpc-work/New/MLModels/Data/Video/training_data_combined_{task_name}.npy')

if task_name in ['3', '5']:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/values_for3,5.npy')
else:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/labels.npy')

labels = labels.astype(float)
n_subjects = len(labels)

# Early fusion
fused_data = np.concatenate([audio_data, video_data], axis=1)

# Save directories
model_dir = f'/rds/user/sss77/hpc-work/New/MLModels/early_fusion/rf/'
plot_dir = f'/rds/user/sss77/hpc-work/New/MLModels/early_fusion/rf/plots_{task_name}/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(plot_dir + 'models/', exist_ok=True)
os.makedirs(plot_dir + 'csv/', exist_ok=True)
# For caching results
results = {}
best_mae_global = float('inf')
def ccc(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.cov(y_true, y_pred)[0][1]
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)

def save_intermediate(trial_num, mae, preds, trues, trial_params):
    global best_mae_global
    if mae < best_mae_global:
        best_mae_global = mae
        rmse = np.sqrt(mean_squared_error(trues, preds))
        pcc, _ = pearsonr(trues, preds)
        ccc_score = ccc(trues, preds)

        # Save plot
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

        # Save trial result log
        with open(os.path.join(model_dir, f'intermediate_best_results_{task_name}.txt'), 'a') as f:
            f.write(f"\nTrial {trial_num}\n")
            f.write(f"MAE:  {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"PCC:  {pcc:.4f}\n")
            f.write(f"CCC:  {ccc_score:.4f}\n")
            f.write(f"Params: {trial_params}\n")
            f.write('-' * 40 + '\n')
            
# Objective function for Optuna
def objective(trial):
    print(f"Running Trial #{trial.number}")
    all_test_preds = []
    all_test_true = []
    val_maes = []
    fold_models = []

    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    for test_id in range(n_subjects):
        X_test = fused_data[test_id:test_id+1]
        y_test = labels[test_id:test_id+1]

        train_val_ids = [i for i in range(n_subjects) if i != test_id]

        best_val_mae_fold = float('inf')
        best_model_fold = None
        best_scalers = None

        for val_id in train_val_ids:
            train_ids = [i for i in train_val_ids if i != val_id]

            X_train = fused_data[train_ids]
            X_val = fused_data[val_id:val_id+1]
            y_train = labels[train_ids]
            y_val = labels[val_id:val_id+1]

            # Standardization
            scaler_X = StandardScaler().fit(X_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

            X_train_scaled = scaler_X.transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

            # Train model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train_scaled)

            # Predict and inverse scale
            y_pred_val = scaler_y.inverse_transform(model.predict(X_val_scaled).reshape(-1, 1)).ravel()
            val_mae = mean_absolute_error(y_val, y_pred_val)

            if val_mae < best_val_mae_fold:
                best_val_mae_fold = val_mae
                best_model_fold = model
                best_scalers = (scaler_X, scaler_y)

        val_maes.append(best_val_mae_fold)
        fold_models.append((best_model_fold, best_scalers))

        # Predict on test sample
        scaler_X, scaler_y = best_scalers
        X_test_scaled = scaler_X.transform(X_test)
        y_pred_test = scaler_y.inverse_transform(best_model_fold.predict(X_test_scaled).reshape(-1, 1)).ravel()

        all_test_preds.append(y_pred_test[0])
        all_test_true.append(y_test[0])

    # Save best model
    best_fold_index = np.argmin(val_maes)
    final_model, _ = fold_models[best_fold_index]
    joblib.dump(final_model, f'{plot_dir}/models/best_model_{trial.number}.pkl')

    # Save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(val_maes, label='Validation MAE per Fold', marker='o')
    plt.title(f"Early Fusion RF - {task_name}")
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plot_dir}/loss_plot_{task_name}.png')
    plt.close()

    results[trial.number] = (np.mean(val_maes), np.array(all_test_preds), np.array(all_test_true))
    save_intermediate(trial.number, np.mean(val_maes), all_test_preds, all_test_true, trial.params)
    return np.mean(val_maes)

# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Retrieve best results
best_trial_number = study.best_trial.number
best_mae, all_test_preds, all_test_true = results[best_trial_number]

# Final evaluation metrics

rmse = np.sqrt(mean_squared_error(all_test_true, all_test_preds))
mae = mean_absolute_error(all_test_true, all_test_preds)
pcc, _ = pearsonr(all_test_true, all_test_preds)
ccc_score = ccc(all_test_true, all_test_preds)

print(f"\nFinal Test Results for {task_name}:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"PCC:  {pcc:.4f}")
print(f"CCC:  {ccc_score:.4f}")

# Plot true vs predicted
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

df = pd.DataFrame({
    'True': all_test_true,
    'Predicted': all_test_preds
})
csv_path = os.path.join(model_dir, f'true_vs_pred_{task_name}.csv')
df.to_csv(csv_path, index=False)

# Save results
results_path = os.path.join(model_dir, f'results_rf_{task_name}.txt')
with open(results_path, 'a') as f:
    f.write(f"\nTask: {task_name}\n")
    f.write(f"Best MAE (Optuna): {study.best_value:.4f}\n")
    f.write(f"Best Hyperparameters: {study.best_trial.params}\n")
    f.write(f"Final Test RMSE: {rmse:.4f}\n")
    f.write(f"Final Test MAE:  {mae:.4f}\n")
    f.write(f"Final Test PCC:  {pcc:.4f}\n")
    f.write(f"Final Test CCC:  {ccc_score:.4f}\n")
    f.write("-" * 40 + "\n")