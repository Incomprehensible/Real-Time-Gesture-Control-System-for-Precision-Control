import argparse
import os
import re
from collections import deque
from pickle import load
from threading import Lock, Thread
from time import sleep

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

NUM_SENSORS = 4
NUM_READINGS = 8

def load_data(data_dir, test_subjects, gestures, sensor_placement=0):
    recordings = []
    baseline_file = None
    NUM_CLASSES = len(gestures)

    pattern = re.compile(rf"({'|'.join(test_subjects)})_({'|'.join(gestures)})\d*_{sensor_placement}.csv")
    # pattern = re.compile(rf"({'|'.join(TEST_SUBJECTS)})_({'|'.join(GESTURES)})1_{SENSOR_PLACEMENT}.csv")

    for date_dir, _, filenames in os.walk(data_dir):
        if date_dir.endswith('data'):
            for filename in filenames:
                if pattern.match(filename):
                    gesture = next((gesture for gesture in gestures if gesture in filename), None)
                    
                    # TODO: Add check if the file has the desired number of sensors recorded
                    
                    recordings.append({
                        "gesture": gesture,
                        "raw_data_filepath": os.path.join(date_dir, filename),
                        "imu_data_filepath": os.path.join(date_dir, f'imu_{filename}') if os.path.exists(os.path.join(date_dir, f'imu_{filename}')) else None,
                        'fft_data_filepath': os.path.join(date_dir, f'fft_{filename}') if os.path.exists(os.path.join(date_dir, f'fft_{filename}')) else None,
                    })
    
    dfs = [[] for _ in range(len(gestures))]

    for recording in recordings:
        dfs[gestures.index(recording["gesture"])].append(pd.read_csv(recording["raw_data_filepath"], header=None))

    concat_dfs = []
    
    for i in range(len(gestures)):
        concat_dfs.append(pd.concat(dfs[i], ignore_index=True).iloc[:, 0:(NUM_SENSORS*NUM_READINGS)])
    
    preprocessor = EMG_preprocessor(lf, hf, fs, trim, bandpass_order, outlier_rejection_stds, None, filter_type='band', library='libemg')

    windows = [[] for _ in range(NUM_CLASSES)]
    data_arrays = [[] for _ in range(NUM_CLASSES)]
    for i, df in enumerate(concat_dfs):
        data_arrays[i], windows[i] = preprocess_per_sensor(df, preprocessor, i, num_sensors=NUM_SENSORS, window_size=400, window_increment=50)

    features = [[] for _ in range(NUM_CLASSES)]
    for i in range(NUM_CLASSES):
        features[i] = np.hstack((get_features_per_sensor(windows[i], feature_groups=('HJORTH', 'HTD')), np.full((windows[i].shape[0], 1), i)))
    
    X = np.vstack([arr[:, :-1] for arr in data_arrays])
    y = np.hstack([arr[:, -1] for arr in data_arrays])

    X_feat = np.vstack([arr[:, :-1] for arr in features])
    y_feat = y
    
    return X_feat, y_feat


def preprocess_per_sensor(df, preprocessor, class_name, num_sensors=4, window_size=200, window_increment=100):
    raw_signal = df.values
    signals = np.array_split(raw_signal, num_sensors, axis=1)
    for sensor in range(0, num_sensors):
        sigs = signals[sensor]
        sigs = np.array(sigs).flatten()
        sigs = preprocessor.preprocess(sigs, sensor)
        sigs = sigs.reshape(-1, 8)
        signals[sensor] = sigs
    raw_signal = np.concatenate(signals, axis=1)
    raw_signal = np.hstack([raw_signal[:, i::8] for i in range(8)])
    raw_signal = raw_signal.reshape(-1, num_sensors)
    raw_signal = get_windows(raw_signal, window_size=window_size, window_increment=window_increment)
    
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
        choices=["sklearn", "torch", "lstm"],
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
    parser.add_argument('-g','--gestures', nargs='+', default=["fist", "index", "middle", "ok", "peace", "thumb", "baseline"] ,help='Set of gestures')
    parser.add_argument("--window_size", type=int, default=200, help="Size of the data window for predictions")
    parser.add_argument("--prediction_interval", type=int, default=50, help="Number of readings before making a new prediction")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_type in ["torch", "lstm"]:
        model = torch.jit.load(args.model_path)
        model.eval()
        model.to(device)
    elif args.model_type == "sklearn":
        with open(args.model_path, "rb") as f:
            model = load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = load(f)
    
    X_feat, y_feat = load_data(args.data_dir, [args.subject], args.gestures)
    
    X_feat = scaler.transform(X_feat)
    
    try:
        if args.model_type == "torch":
            
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

