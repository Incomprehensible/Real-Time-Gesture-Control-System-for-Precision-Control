import argparse
from collections import deque
from pickle import load
from threading import Lock, Thread
from time import sleep

import numpy as np
import pandas as pd
import serial
import torch
import umyo_parser
from libemg.feature_extractor import FeatureExtractor
from preprocessing import EMG_preprocessor

lf = 20
hf = 500
fs = 1150
trim = 4 * 8 * 5
bandpass_order = 4
OUTLIER_REJECTION_STDS = 6
PREDICTION_THRESHOLD = 0.0

NUM_SENSORS = 5
NUM_FFT_READINGS = 4
USE_FFT = True

ids = [1633709441, 3274504362, 2749159433, 3048451580, 3899692357]


def get_features_per_sensor(windows, feature_groups=('HTD',)):
    fe = FeatureExtractor()
    
    features_list = []
    
    for feature_group in feature_groups:
        if feature_group not in fe.get_feature_groups().keys():
            raise ValueError(f"Invalid feature group: {feature_group}")

        features = fe.extract_feature_group(feature_group, windows, array=True)
        features_list.append(features)
    return np.concatenate(features_list, axis=1)


def data_collector(serial_port, raw_data_queue, fft_data_queue, lock):
    last_data_ids = [0] * NUM_SENSORS
    while True:
        try:
            cnt = serial_port.in_waiting
            if cnt > 0:
                data_raw = serial_port.read(cnt)
                umyo_parser.umyo_parse_preprocessor(data_raw)
                sensors_proc = umyo_parser.umyo_get_list()
                
                num_sensors = len(sensors_proc)
                if num_sensors < NUM_SENSORS:
                    print(f"Sensors found: {str(num_sensors)}")
                    sleep(1)
                    continue

                raw_data = np.zeros((NUM_SENSORS, 8), dtype=np.float32)
                fft_data = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
                data_ids = [0] * NUM_SENSORS
                for sensor_read in sensors_proc:
                    raw_data[ids.index(sensor_read.unit_id)] = sensor_read.data_array[:8]
                    fft_data[ids.index(sensor_read.unit_id)] = sensor_read.device_spectr[1:4]
                    data_ids[ids.index(sensor_read.unit_id)] = sensor_read.data_id
                
                if last_data_ids == data_ids:  # Skip if no new data
                    continue
                last_data_ids = data_ids
                
                raw_data = raw_data.mean(axis=1)
                raw_data = raw_data.flatten()
                
                with lock:
                    raw_data_queue.append(list(raw_data)) # Append averaged data
                    fft_data_queue.append(list(fft_data)) # Append averaged FFT data

        except Exception as e:
            print(f"Data collection error: {e}")
            sleep(1)


def classification(windowed_data, fft_data, model, scaler, preprocessor, model_type="sklearn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preprocess data by sensor
    for i in range(NUM_SENSORS):
        windowed_data[:, i] = preprocessor.preprocess(windowed_data[:, i], i)
    
    windowed_data = np.expand_dims(windowed_data, axis=0) # Correct format for LSTM (batch, seq_len, num_sensors)
    features = get_features_per_sensor(windowed_data.swapaxes(1, 2), feature_groups=('HJORTH', 'HTD'))
    
    if USE_FFT:
        fft_data = np.concatenate([np.min(fft_data, axis=0), np.max(fft_data, axis=0)], axis=1)
        fft_data = fft_data.reshape(1, -1)
        features = np.hstack([features, fft_data])
    
    if model_type == "lstm":
        shape = windowed_data.shape
        data = scaler.transform(windowed_data.reshape(windowed_data.shape[0], -1))
        data = torch.tensor(data.reshape(*shape)).float().to(device)
        prediction = model(data).argmax(dim=1).numpy().item()
    elif model_type == "torch" or model_type == "tf":
        features = scaler.transform(features)
        if model_type == "tf":
            n_features = NUM_SENSORS
            n_sequence = int(features.shape[1] / n_features)

            features = features.reshape(features.shape[0], n_features, n_sequence)
            features = np.swapaxes(features, 2, 1)
        data = torch.tensor(features, dtype=torch.float32).to(device)
        prediction = model(data).argmax(dim=1).numpy().item()
    elif model_type == "sklearn":
        features = scaler.transform(features)
        prediction = int(model.predict(features).squeeze())
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time classification of EMG data")
    parser.add_argument(
        "--model_type",
        choices=["sklearn", "torch", "lstm", "tf"],
        default="sklearn",
        help="Type of model to use for classification",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../training/resources/custom_classifier_gen2.pkl",
        help="Path to classifier",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="../training/resources/custom_scaler.pkl",
        help="Path to scaler",
    )
    parser.add_argument('-g','--gestures', nargs='+', default=("baseline", "fist", "peace", "up", "down", "lift") ,help='Set of gestures') 
    parser.add_argument("--window_size", type=int, default=200, help="Size of the data window for predictions")
    parser.add_argument("--prediction_delay", type=int, default=1, help="Delay between predictions")
    parser.add_argument("-p", "--port", type=str, default="COM7", help="USB receiving station port")

    args = parser.parse_args()
    
    ser = serial.Serial(
        port=args.port,
        baudrate=921600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_type in ["torch", "lstm", "tf"]:
        model = torch.jit.load(args.model_path, map_location=torch.device(device))
        model.eval()
        model.to(device)
    elif args.model_type == "sklearn":
        with open(args.model_path, "rb") as f:
            model = load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = load(f)
    
    preprocessor = EMG_preprocessor(lf, hf, fs, trim, bandpass_order, OUTLIER_REJECTION_STDS, None, filter_type='band', library='libemg')
    
    raw_data_queue = deque([], maxlen=args.window_size)
    fft_data_queue = deque([], maxlen=args.window_size)
    lock = Lock()
    
    data_thread = Thread(target=data_collector, args=(ser, raw_data_queue, fft_data_queue, lock), daemon=True)
    data_thread.start()
    
    try:
        while True:
            with lock:
                windowed_data = np.array(raw_data_queue)
                fft_data = np.array(fft_data_queue)
            
            if len(windowed_data) == args.window_size:
                prediction = classification(windowed_data, fft_data, model, scaler, preprocessor, args.model_type)
                print("Prediction: ", args.gestures[prediction])
            sleep(args.prediction_delay)
    except KeyboardInterrupt:
        print("Stopped classification session.")
