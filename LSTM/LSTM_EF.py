import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import optuna
import sys

best_mae_global = float('inf')

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

def save_intermediate(trial_num, mae, preds, trues, trial_params, plot_dir, model_dir, task_name):
    global best_mae_global
    if mae < best_mae_global:
        best_mae_global = mae
        rmse = np.sqrt(mean_squared_error(trues, preds))
        pcc, _ = pearsonr(trues, preds)
        ccc_score = ccc(trues, preds)

        plt.figure(figsize=(8, 6))
        plt.scatter(trues, preds, alpha=0.7)
        plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Trial {trial_num} - {task_name}')
        plt.grid(True)
        os.makedirs(f"{plot_dir}/csv", exist_ok=True)
        plt.savefig(f"{plot_dir}/trial_{trial_num}_scatter.png")
        plt.close()

        pd.DataFrame({'True': trues, 'Predicted': preds}).to_csv(
            f"{plot_dir}/csv/trial_{trial_num}_true_vs_pred.csv", index=False)

        with open(os.path.join(model_dir, f'intermediate_best_results_{task_name}.txt'), 'a') as f:
            f.write(f"\nTrial {trial_num}\n")
            f.write(f"MAE:  {mae:.4f}\nRMSE: {rmse:.4f}\nPCC:  {pcc:.4f}\nCCC:  {ccc_score:.4f}\n")
            f.write(f"Params: {trial_params}\n" + '-' * 40 + '\n')
            
        model_save_path = os.path.join(model_dir, f"models/model_{trial_num}")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)

def objective(trial, task_name, audio_data, video_data, labels, plot_dir, model_dir):
    max_audio_length = max(len(seq) for seq in audio_data)
    max_video_length = max(len(seq) for seq in video_data)
    
    print(f"max_len = {max_audio_length}")
    print(f"initial shape = {audio_data.shape}")
    print(f"data_shape1 = {len(audio_data)}")

    audio_data = pad_sequence(audio_data, max_audio_length)
    video_data = pad_sequence(video_data, max_video_length)
    
    print(f"data_shape2 = {len(audio_data)}")

    audio_scaler = StandardScaler()
    video_scaler = StandardScaler()
    label_scaler = StandardScaler()

    n_subjects = len(audio_data)
    loo = LeaveOneOut()
    print(f"len_subjs=={n_subjects}")

    y_true_all, y_pred_all = [], []

    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    for test_index in range(n_subjects):
        print(test_index)
        X_test_audio = audio_data[test_index:test_index+1]
        X_test_video = video_data[test_index:test_index+1]
        y_test = labels[test_index:test_index+1]
        print(len(X_test_audio))
        print(X_test_video.shape)
        train_val_ids = [i for i in range(n_subjects) if i != test_index]
        best_mae_val = float('inf')
        best_weights = None

        for val_index in train_val_ids:
            train_ids = [i for i in train_val_ids if i != val_index]

            X_train_audio = audio_data[train_ids]
            X_val_audio = audio_data[val_index:val_index+1]
            X_train_video = video_data[train_ids]
            X_val_video = video_data[val_index:val_index+1]
            y_train = labels[train_ids]
            y_val = labels[val_index:val_index+1]

            audio_flat = X_train_audio.reshape(-1, X_train_audio.shape[-1])
            video_flat = X_train_video.reshape(-1, X_train_video.shape[-1])
            
            print(f"audio_flat={audio_flat.shape}")

            audio_scaler.fit(audio_flat)
            video_scaler.fit(video_flat)
            label_scaler.fit(y_train.reshape(-1, 1))

            X_train_audio = audio_scaler.transform(audio_flat).reshape(X_train_audio.shape)
            X_val_audio = audio_scaler.transform(X_val_audio.reshape(-1, X_val_audio.shape[-1])).reshape(X_val_audio.shape)
            X_train_video = video_scaler.transform(video_flat).reshape(X_train_video.shape)
            X_val_video = video_scaler.transform(X_val_video.reshape(-1, X_val_video.shape[-1])).reshape(X_val_video.shape)
            y_train_scaled = label_scaler.transform(y_train.reshape(-1, 1))
            y_val_scaled = label_scaler.transform(y_val.reshape(-1, 1))

            X_train = np.concatenate([X_train_audio, X_train_video], axis=-1)
            X_val = np.concatenate([X_val_audio, X_val_video], axis=-1)

            input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
            lstm_out = LSTM(lstm_units, activation='tanh')(input_layer)
            lstm_out = Dropout(dropout_rate)(lstm_out)
            output = Dense(1)(lstm_out)
            model = Model(input_layer, output)

            model.compile(optimizer=Adam(learning_rate), loss='mse')
            early_stop = EarlyStopping(patience=3, restore_best_weights=True)

            model.fit(X_train, y_train_scaled, epochs=20, batch_size=16, verbose=0, validation_data=(X_val, y_val_scaled), callbacks=[early_stop])

            y_val_pred_scaled = model.predict(X_val)
            y_val_pred = label_scaler.inverse_transform(y_val_pred_scaled)
            mae_val = mean_absolute_error(y_val, y_val_pred)

            if mae_val < best_mae_val:
                best_mae_val = mae_val
                best_weights = model.get_weights()

        X_test_audio = audio_scaler.transform(X_test_audio.reshape(-1, X_test_audio.shape[-1])).reshape(X_test_audio.shape)
        X_test_video = video_scaler.transform(X_test_video.reshape(-1, X_test_video.shape[-1])).reshape(X_test_video.shape)
        X_test = np.concatenate([X_test_audio, X_test_video], axis=-1)

        model.set_weights(best_weights)
        y_test_pred_scaled = model.predict(X_test)
        y_test_pred = label_scaler.inverse_transform(y_test_pred_scaled)

        y_true_all.append(y_test[0])
        y_pred_all.append(y_test_pred.ravel()[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    
    final_model_path = os.path.join(model_dir, f'models/best_model_{trial.number}.keras')
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)

    save_intermediate(trial.number, mae, y_pred_all, y_true_all, trial.params, plot_dir, model_dir, task_name)
    return mae

# Tasks list

task_index = int(sys.argv[1])
task_names = ['2h', '2s', '3', '4_P1', '4_P2', '4_P3', '5']
task_name = task_names[task_index]

print(f"Running task: {task_name}")
audio_data = np.load(f'/Users/sujithsaisripadam/Downloads/Data/Audio/training_data_{task_name}.npy', allow_pickle=True)
video_data = np.load(f'/Users/sujithsaisripadam/Downloads/Data/Video/training_data_{task_name}.npy', allow_pickle=True)
if task_name in ['3', '5']:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/values_for3,5.npy')
else:
    labels = np.load('/rds/user/sss77/hpc-work/New/MLModels/Data/labels.npy')
labels = labels.astype(float)


model_dir = f'/Users/sujithsaisripadam/Downloads/EF/task_{task_name}'
plot_dir = f'/Users/sujithsaisripadam/Downloads/EF/task_{task_name}/Plots_{task_name}'
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, task_name, audio_data, video_data, labels, plot_dir, model_dir), n_trials=30)

print(f"Best trial for task {task_name}:\nMAE: {study.best_value:.4f}\nParams: {study.best_trial.params}")

with open(os.path.join(model_dir, f'best_trial_{task_name}.txt'), 'w') as f:
    f.write(f"Best MAE: {study.best_value:.4f}\n")
    f.write(f"Best Parameters: {study.best_trial.params}\n")
