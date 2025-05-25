import numpy as np
from sklearn.ensemble import AdaBoostRegressor
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
task_name = task_names[task_index]
print(f"\nRunning AdaBoost + Optuna for task: {task_name}")

# Load data
audio_data = np.load(f'/rds/user/sss77/hpc-work/New/MLModels/Data/Audio/training_data_combined_{task_name}.npy')
video_data = np.load(f'/rds/user/sss77/hpc-work/New/MLModels/Data/Video/training_data_combined_{task_name}.npy')

if task_name in ['3', '5']:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/values_for3,5.npy')
else:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/labels.npy')

labels = labels.astype(float)
n_subjects = len(labels)

# Save directories
model_dir = f'/rds/user/sss77/hpc-work/New/MLModels/late_fusion/ab/'
plot_dir = f'/rds/user/sss77/hpc-work/New/MLModels/late_fusion/ab/plots_{task_name}/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(plot_dir + 'models/', exist_ok=True)
os.makedirs(plot_dir + 'csv/', exist_ok=True)
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

def objective(trial):
    print(f"Running Trial #{trial.number}")
    all_test_preds = []
    all_test_true = []
    val_maes = []
    fold_models = []

    # Hyperparameter suggestions for AdaBoost
    n_estimators_audio = trial.suggest_int('n_estimators_audio', 10, 300)
    n_estimators_video = trial.suggest_int('n_estimators_video', 10, 300)
    learning_rate_audio = trial.suggest_float('learning_rate_audio', 0.01, 1.0, log=True)
    learning_rate_video = trial.suggest_float('learning_rate_video', 0.01, 1.0, log=True)
    loss_audio = trial.suggest_categorical('loss_audio', ['linear', 'square', 'exponential'])
    loss_video = trial.suggest_categorical('loss_video', ['linear', 'square', 'exponential'])

    for test_id in range(n_subjects):
        X_test_audio = audio_data[test_id:test_id+1]
        X_test_video = video_data[test_id:test_id+1]
        y_test = labels[test_id:test_id+1]
        train_val_ids = [i for i in range(n_subjects) if i != test_id]

        best_val_mae_fold = float('inf')
        best_audio_model_fold = None
        best_video_model_fold = None
        best_scalers = None

        for val_id in train_val_ids:
            train_ids = [i for i in train_val_ids if i != val_id]

            X_train_audio = audio_data[train_ids]
            X_val_audio = audio_data[val_id:val_id+1]
            X_train_video = video_data[train_ids]
            X_val_video = video_data[val_id:val_id+1]
            y_train = labels[train_ids]
            y_val = labels[val_id:val_id+1]

            # Standardize
            scaler_audio = StandardScaler().fit(X_train_audio)
            scaler_video = StandardScaler().fit(X_train_video)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

            X_train_audio_scaled = scaler_audio.transform(X_train_audio)
            X_val_audio_scaled = scaler_audio.transform(X_val_audio)
            X_train_video_scaled = scaler_video.transform(X_train_video)
            X_val_video_scaled = scaler_video.transform(X_val_video)
            y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

            # Train AdaBoost models
            audio_model = AdaBoostRegressor(
                n_estimators=n_estimators_audio,
                learning_rate=learning_rate_audio,
                loss=loss_audio,
                random_state=42
            )
            video_model = AdaBoostRegressor(
                n_estimators=n_estimators_video,
                learning_rate=learning_rate_video,
                loss=loss_video,
                random_state=42
            )

            audio_model.fit(X_train_audio_scaled, y_train_scaled)
            video_model.fit(X_train_video_scaled, y_train_scaled)

            y_pred_audio_val = scaler_y.inverse_transform(audio_model.predict(X_val_audio_scaled).reshape(-1, 1)).ravel()
            y_pred_video_val = scaler_y.inverse_transform(video_model.predict(X_val_video_scaled).reshape(-1, 1)).ravel()
            y_pred_combined_val = (y_pred_audio_val + y_pred_video_val) / 2

            val_mae = mean_absolute_error(y_val, y_pred_combined_val)

            if val_mae < best_val_mae_fold:
                best_val_mae_fold = val_mae
                best_audio_model_fold = audio_model
                best_video_model_fold = video_model
                best_scalers = (scaler_audio, scaler_video, scaler_y)

        val_maes.append(best_val_mae_fold)
        fold_models.append((best_audio_model_fold, best_video_model_fold, best_scalers))

        # Predict on test sample
        scaler_audio, scaler_video, scaler_y = best_scalers
        X_test_audio_scaled = scaler_audio.transform(X_test_audio)
        X_test_video_scaled = scaler_video.transform(X_test_video)

        y_pred_audio_test = scaler_y.inverse_transform(best_audio_model_fold.predict(X_test_audio_scaled).reshape(-1, 1)).ravel()
        y_pred_video_test = scaler_y.inverse_transform(best_video_model_fold.predict(X_test_video_scaled).reshape(-1, 1)).ravel()
        y_pred_combined_test = (y_pred_audio_test + y_pred_video_test) / 2

        all_test_preds.append(y_pred_combined_test[0])
        all_test_true.append(y_test[0])

    # Save best models
    best_fold_index = np.argmin(val_maes)
    final_audio_model, final_video_model, _ = fold_models[best_fold_index]
    joblib.dump(final_audio_model, f'{plot_dir}/models/best_audio_model_{trial.number}.pkl')
    joblib.dump(final_video_model, f'{plot_dir}/models/best_video_model_{trial.number}.pkl')

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(val_maes, label='Validation MAE per Fold', marker='x')
    plt.title(f"Late Fusion AdaBoost - {task_name}")
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

# Final evaluation
best_trial_number = study.best_trial.number
best_mae, all_test_preds, all_test_true = results[best_trial_number]


rmse = np.sqrt(mean_squared_error(all_test_true, all_test_preds))
mae = mean_absolute_error(all_test_true, all_test_preds)
pcc, _ = pearsonr(all_test_true, all_test_preds)
ccc_score = ccc(all_test_true, all_test_preds)

print(f"\nFinal Test Results for {task_name}:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"PCC:  {pcc:.4f}")
print(f"CCC:  {ccc_score:.4f}")

# Scatter plot
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

# Save CSV and results
df = pd.DataFrame({
    'True': all_test_true,
    'Predicted': all_test_preds
})
csv_path = os.path.join(model_dir, f'true_vs_pred_{task_name}.csv')
df.to_csv(csv_path, index=False)

results_path = os.path.join(model_dir, f'results_ab_{task_name}.txt')
with open(results_path, 'a') as f:
    f.write(f"\nTask: {task_name}\n")
    f.write(f"Best MAE (Optuna): {study.best_value:.4f}\n")
    f.write(f"Best Hyperparameters: {study.best_trial.params}\n")
    f.write(f"Final Test RMSE: {rmse:.4f}\n")
    f.write(f"Final Test MAE:  {mae:.4f}\n")
    f.write(f"Final Test PCC:  {pcc:.4f}\n")
    f.write(f"Final Test CCC:  {ccc_score:.4f}\n")
    f.write("-" * 40 + "\n")