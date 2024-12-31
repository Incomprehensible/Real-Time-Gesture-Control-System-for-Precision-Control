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

NUM_SENSORS = 4

ser = serial.Serial(
    port="COM7",
    baudrate=921600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=0,
)

ids = [1633709441, 3274504362, 2749159433, 3048451580, 3899692357]

classes = ["fist", "index", "middle", "ok", "peace", "thumb", "baseline"]


def get_features_per_sensor(windows, feature_groups=('HTD',)):
    fe = FeatureExtractor()
    
    features_list = []
    
    for feature_group in feature_groups:
        if feature_group not in fe.get_feature_groups().keys():
            raise ValueError(f"Invalid feature group: {feature_group}")

        features = fe.extract_feature_group(feature_group, windows, array=True)
        features_list.append(features)
    return np.concatenate(features_list, axis=1)


def data_collector(serial_port, data_queue, lock):
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

                sensor_data = np.zeros((num_sensors, 8), dtype=np.float32)
                for sensor_read in sensors_proc:
                    sensor_data[ids.index(sensor_read.unit_id)] = sensor_read.data_array[:8]
                sensor_data = sensor_data.flatten()
                sensor_data = np.hstack([sensor_data[i::8] for i in range(8)])
                sensor_data = sensor_data.reshape(-1, NUM_SENSORS)
                
                with lock:
                    data_queue.extend(list(sensor_data)) # Append all 8 rows individually

        except Exception as e:
            print(f"Data collection error: {e}")
            sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from uMyo")
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
    parser.add_argument("--baseline_path", type=str, default="../recordings/19_12_24/data/nad_baseline_0.csv", help="Path to baseline file")
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
    
    baseline_array = pd.read_csv(args.baseline_path, header=None).values

    preprocessor = EMG_preprocessor(lf, hf, fs, trim, bandpass_order, OUTLIER_REJECTION_STDS, np.zeros(baseline_array.flatten().shape), filter_type='band', library='libemg')
    fe = FeatureExtractor()
    
    data_queue = deque([], maxlen=args.window_size)
    lock = Lock()
    
    data_thread = Thread(target=data_collector, args=(ser, data_queue, lock), daemon=True)
    data_thread.start()
    
    try:
        while True:
            with lock:
                windowed_data = np.array(data_queue)
            
            if len(windowed_data) == args.window_size:
                # Preprocess data by sensor
                for i in range(NUM_SENSORS):
                    windowed_data[:, i] = preprocessor.preprocess(windowed_data[:, i], i)
                
                windowed_data = np.expand_dims(windowed_data, axis=0) # Correct format for LSTM (batch, seq_len, num_sensors)
                features = get_features_per_sensor(windowed_data.swapaxes(1, 2), feature_groups=('HJORTH', 'HTD'))
                
                if args.model_type == "lstm":
                    shape = windowed_data.shape
                    data = scaler.transform(windowed_data.reshape(windowed_data.shape[0], -1))
                    data = torch.tensor(data.reshape(*shape)).float().to(device)
                    logits = model(data)
                    
                    shifted_logits = logits - logits.min(dim=1, keepdim=True).values
                    normalized_logits = shifted_logits / (shifted_logits.sum(dim=1, keepdim=True) + 1e-8)
                    
                    if normalized_logits.max() > PREDICTION_THRESHOLD:
                        prediction = normalized_logits.argmax(dim=1).cpu().numpy().item()
                        print("Prediction: ", classes[prediction])
                elif args.model_type == "torch":
                    features = scaler.transform(features)
                    
                    #print(features)
                    
                    data = torch.tensor(features).float().to(device)
                    logits = model(data)
                    
                    #print("Logits: ", logits.cpu().detach().numpy().tolist())
                    
                    shifted_logits = logits - logits.min(dim=1, keepdim=True).values
                    normalized_logits = shifted_logits / (shifted_logits.sum(dim=1, keepdim=True) + 1e-8)
                    
                    #print("Normalized logits: ", normalized_logits.cpu().detach().numpy().tolist())
                    
                    
                    if normalized_logits.max() > PREDICTION_THRESHOLD:
                        prediction = normalized_logits.argmax(dim=1).cpu().numpy().item()
                        print("Prediction: ", classes[prediction])
                    #print()
                elif args.model_type == "sklearn":
                    features = scaler.transform(features)
                    prediction = int(model.predict(features).squeeze())
                    print("Prediction: ", classes[prediction])
                sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped classification session.")

