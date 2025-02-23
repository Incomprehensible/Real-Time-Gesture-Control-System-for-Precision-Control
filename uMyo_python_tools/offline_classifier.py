import argparse
import os
import re
from pickle import load

import numpy as np
import pandas as pd
import torch
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows
from preprocessing import EMG_preprocessor

lf = 20
hf = 500
fs = 1150
trim = 4 * 8 * 5
bandpass_order = 4
outlier_rejection_stds = 6

NUM_SENSORS = 5
NUM_READINGS = 8
NUM_FFT_READINGS = 4

USE_FFT = True

def check_sensors_present(file, num_sensors=NUM_SENSORS):
    df = pd.read_csv(file)
    return len(df.columns) >= num_sensors * NUM_READINGS

def load_data(data_dir, test_subjects, gestures, window_size, sensor_placement=0):
    recordings = []

    pattern = re.compile(rf"({'|'.join(test_subjects)})_({'|'.join(gestures)})\d*_{sensor_placement}.csv")

    for date_dir, _, filenames in os.walk(data_dir):
        if date_dir.endswith('data') and not 'static' in date_dir:
            for filename in filenames:
                if pattern.match(filename):
                    gesture = next((gesture for gesture in gestures if gesture in filename), None)
                    
                    emg_data_file = os.path.join(date_dir, filename)
                    if check_sensors_present(emg_data_file):
                        recordings.append({
                            "gesture": gesture,
                            "raw_data_filepath": emg_data_file,
                            "imu_data_filepath": os.path.join(date_dir, f'imu_{filename}') if os.path.exists(os.path.join(date_dir, f'imu_{filename}')) else None,
                            'fft_data_filepath': os.path.join(date_dir, f'fft_{filename}') if os.path.exists(os.path.join(date_dir, f'fft_{filename}')) else None,
                        })
    
    dfs = [[] for _ in range(len(gestures))]
    fft_dfs = [[] for _ in range(len(gestures))]

    if USE_FFT:
        for recording in recordings:
            if not recording["fft_data_filepath"]:
                print('Recording without fft_file detected.')
                recordings.remove(recording)

    for recording in recordings:
        dfs[gestures.index(recording["gesture"])].append(pd.read_csv(recording["raw_data_filepath"], header=None))
        if USE_FFT:
            fft_dfs[gestures.index(recording["gesture"])].append(pd.read_csv(recording["fft_data_filepath"], header=None))

    concat_dfs = []
    concat_fft_dfs = []

    for i in range(len(gestures)):
        concat_dfs.append(pd.concat(dfs[i], ignore_index=True).iloc[:, 0:(NUM_SENSORS*NUM_READINGS)])
        if USE_FFT:
            concat_fft_dfs.append(pd.concat(fft_dfs[i], ignore_index=True).iloc[:, 0:(NUM_SENSORS*NUM_FFT_READINGS)])
    
    if USE_FFT:
        for i in range(len(concat_fft_dfs)):
            concat_fft_dfs[i] = concat_fft_dfs[i].iloc[:, [i*NUM_FFT_READINGS+j+1 for i in range(0, NUM_SENSORS) for j in range(NUM_FFT_READINGS-1)]]
            concat_fft_dfs[i].columns = range(concat_fft_dfs[i].columns.size)

    preprocessor = EMG_preprocessor(lf, hf, fs, trim, bandpass_order, outlier_rejection_stds, None, filter_type='band', library='libemg')

    windows = [[] for _ in range(len(gestures))]
    data_arrays = [[] for _ in range(len(gestures))]
    for i, df in enumerate(concat_dfs):
        data_arrays[i], windows[i] = preprocess_per_sensor_avg(df, preprocessor, i, num_sensors=NUM_SENSORS, window_size=window_size, window_increment=(window_size // 5) + (window_size // 5 % 2))

    features = [[] for _ in range(len(gestures))]
    for i in range(len(gestures)):
        features[i] = np.hstack((get_features_per_sensor(windows[i], feature_groups=('HJORTH', 'HTD')), np.full((windows[i].shape[0], 1), i)))
    
    if USE_FFT:
        fft_data_arrays = [[] for _ in range(len(gestures))]
        fft_windows = [[] for _ in range(len(gestures))]
        for i, df in enumerate(concat_fft_dfs):
            fft_data_arrays[i], fft_windows[i] = preprocess_per_sensor_fft(df, i, num_sensors=NUM_SENSORS, window_size=window_size, window_increment=(window_size // 5) + (window_size // 5 % 2), num_readings=NUM_FFT_READINGS)

        fft_features = [[] for _ in range(len(gestures))]
        for i in range(len(gestures)):
            fft_features[i] = np.hstack((get_fft_features_per_sensor(fft_windows[i]), np.full((fft_windows[i].shape[0], 1), i)))

    X = np.vstack([arr[:, :-1] for arr in data_arrays])
    y = np.hstack([arr[:, -1] for arr in data_arrays])

    X_feat = np.vstack([arr[:, :-1] for arr in features])
    y_feat = y

    if USE_FFT:
        X_fft_feat = np.vstack([arr[:, :-1] for arr in fft_features])
        X_feat = np.hstack((X_feat, X_fft_feat))
    
    return X_feat, y_feat

def preprocess_by_window(preprocessor, windows):
    for window in windows:
        for sensor in range(NUM_SENSORS):
            window[sensor, :] = preprocessor.preprocess(window[sensor, :], sensor)

def preprocess_per_sensor_avg(df, preprocessor, class_name, num_sensors=4, window_size=200, window_increment=100, num_readings=NUM_READINGS):
    for sensor in range(num_sensors):
        df[f'avg_{sensor}'] = df.iloc[:, sensor*num_readings:sensor*num_readings+num_readings].mean(axis=1)
    raw_signal = df.iloc[:, -NUM_SENSORS:].values
    raw_signal = get_windows(raw_signal, window_size=window_size, window_increment=window_increment)
    preprocess_by_window(preprocessor, raw_signal) 
    features_preshaped = raw_signal
    raw_signal = raw_signal.reshape(raw_signal.shape[0], -1)
    raw_signal = np.hstack((raw_signal, np.full((raw_signal.shape[0], 1), class_name)))
    return raw_signal, features_preshaped

class FFT_wrapper():
    def __init__(self, array):
        self.array = array
    def get_data(self):
        return self.array

def preprocess_per_sensor_fft(df, class_name, num_sensors=4, window_size=200, window_increment=100, num_readings=NUM_READINGS):
    raw_signal = df.values
    new_shape = (raw_signal.shape[0], num_sensors)
    raw_signal_wrapped = np.empty(shape=new_shape, dtype=object)
    for i in range(len(raw_signal)):
        sensor_data = np.array_split(raw_signal[i], num_sensors, axis=0)
        for sensor in range(num_sensors):
            raw_signal_wrapped[i, sensor] = FFT_wrapper(sensor_data[sensor])
    # print(raw_signal_wrapped[5, 0].get_data()) # 5th fft reading of 1st sensor
    raw_signal = get_windows(raw_signal_wrapped, window_size=window_size, window_increment=window_increment)
    # print(raw_signal[0, 0, 5].get_data()) # 5th fft reading of 1st sensor
    features_preshaped = raw_signal
    raw_signal = raw_signal.reshape(raw_signal.shape[0], -1)
    raw_signal = np.hstack((raw_signal, np.full((raw_signal.shape[0], 1), class_name)))
    return raw_signal, features_preshaped

def get_fft_features_per_sensor(windows, num_sensors=NUM_SENSORS):
    # obtain minimum and maximum for each bin
    features_shape = (windows.shape[0], num_sensors, 2*(NUM_FFT_READINGS-1))
    features = np.zeros(shape=features_shape, dtype=float)
    for i in range(windows.shape[0]):
        for sensor in range(num_sensors):
            min = np.array([[np.min([windows[i, sensor, w].get_data()[k] for w in range(windows.shape[2])])] for k in range(NUM_FFT_READINGS-1)])
            max = np.array([[np.max([windows[i, sensor, w].get_data()[k] for w in range(windows.shape[2])])] for k in range(NUM_FFT_READINGS-1)])
            # min = np.min([windows[i, sensor, w].get_data()[k] for w in range(windows.shape[2]) for k in range(NUM_FFT_READINGS-1)])
            features[i, sensor] = np.array([min, max]).reshape(-1)
    return features.reshape(features.shape[0], -1)

def preprocess_per_sensor(df, preprocessor, class_name, num_sensors=4, window_size=200, window_increment=100):
    raw_signal = df.values
    raw_signal = np.hstack([raw_signal[:, i::8] for i in range(8)])
    raw_signal = raw_signal.reshape(-1, num_sensors)
    raw_signal = get_windows(raw_signal, window_size=window_size, window_increment=window_increment)
    preprocess_by_window(preprocessor, raw_signal)
    
    features_preshaped = raw_signal
    raw_signal = raw_signal.reshape(raw_signal.shape[0], -1)
    raw_signal = np.hstack((raw_signal, np.full((raw_signal.shape[0], 1), class_name)))
    return raw_signal, features_preshaped

def get_features_per_sensor(windows, feature_groups=('HTD',)):
    fe = FeatureExtractor()
    
    features_list = []
    
    for feature_group in feature_groups:
        if feature_group not in fe.get_feature_groups().keys():
            raise ValueError(f"Invalid feature group: {feature_group}")

        features = fe.extract_feature_group(feature_group, windows, array=True)
        features_list.append(features)
    return np.concatenate(features_list, axis=1)

def evaluateModel(outputs, y):
    _, predicted = outputs.max(1)
    test_accuracy = predicted.eq(y).sum().item() / y.size(0) * 100
    return test_accuracy, predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline classification of EMG data")
    parser.add_argument(
        "--model_type",
        choices=["sklearn", "torch", "lstm", "tf"],
        default="sklearn",
        help="Type of model to use for classification",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../pretraining/custom_classifier_gen2.pkl",
        help="Path to classifier",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="../pretraining/custom_scaler.pkl",
        help="Path to scaler",
    )
    parser.add_argument('-d', '--data_dir', type=str, default="../recordings/31_12_24_5", help="Directory containing data files")
    parser.add_argument('-s', '--subject', type=str, default="gilbert", help="Subject to use for classification")
    parser.add_argument('-g','--gestures', nargs='+', default=("fist", "index", "ok", "peace", "thumb", "up", "down") ,help='Set of gestures')
    parser.add_argument("--window_size", type=int, default=400, help="Size of the data window for predictions")
    parser.add_argument("--prediction_interval", type=int, default=50, help="Number of readings before making a new prediction")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_type in ["torch", "lstm", "tf"]:
        model = torch.jit.load(args.model_path)
        model.eval()
        model.to(device)
    elif args.model_type == "sklearn":
        with open(args.model_path, "rb") as f:
            model = load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = load(f)
    
    X_feat, y_feat = load_data(args.data_dir, [args.subject], args.gestures, args.window_size)
    
    X_feat = scaler.transform(X_feat)
    
    try:
        if args.model_type == "torch" or args.model_type == 'tf':
            
            if args.model_type == "tf":
                n_features = NUM_SENSORS
                n_sequence = int(X_feat.shape[1] / n_features)

                X_feat = X_feat.reshape(X_feat.shape[0], n_features, n_sequence)
                X_feat = np.swapaxes(X_feat, 2, 1)

            X_feat_tensor = torch.tensor(X_feat, dtype=torch.float32).to(device)
            y_feat_tensor = torch.tensor(y_feat, dtype=torch.long).to(device)

            model.eval()
            with torch.no_grad():
                y_hat_test = model(X_feat_tensor)
                acc, predicted = evaluateModel(y_hat_test, y_feat_tensor)
                print(f"Test Accuracy: {acc:.2f}%")
        elif args.model_type == "sklearn":
            acc = model.score(X_feat, y_feat) * 100
            print(f"Test Accuracy: {acc:.2f}%")

    except KeyboardInterrupt:
        print("Stopped classification session.")

