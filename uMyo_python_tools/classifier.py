import argparse
from collections import deque
from pickle import load
from threading import Lock, Thread
from time import sleep

import numpy as np
import serial
import torch
import umyo_parser
from libemg.feature_extractor import FeatureExtractor
from parameters import IDS
from preprocessing import EMG_preprocessor


class EMG_Inference:
    def __init__(
        self,
        port="/dev/ttyUSB0",
        model_path="../training/resources/custom_classifier_gen2.pkl",
        model_type="sklearn",
        scaler_path="../training/resources/custom_scaler.pkl",
        window_size=100,
    ) -> None:
        self.ser = serial.Serial(
            port=port,
            baudrate=921600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=0,
        )

        self.model_path = model_path
        self.model_type = model_type
        self.scaler_path = scaler_path
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()

        self.raw_data_queue = deque([], maxlen=self.window_size)
        self.fft_data_queue = deque([], maxlen=self.window_size)
        self.lock = Lock()

        data_thread = Thread(
            target=self.data_collector,
            args=(self.ser, self.raw_data_queue, self.fft_data_queue, self.lock),
            daemon=True,
        )
        data_thread.start()

    def get_features_per_sensor(self, windows, feature_groups=("HTD",)):
        fe = FeatureExtractor()

        features_list = []

        for feature_group in feature_groups:
            if feature_group not in fe.get_feature_groups().keys():
                raise ValueError(f"Invalid feature group: {feature_group}")

            features = fe.extract_feature_group(feature_group, windows, array=True)
            features_list.append(features)
        return np.concatenate(features_list, axis=1)

    def data_collector(self, serial_port, raw_data_queue, fft_data_queue, lock):
        last_data_ids = [0] * self.NUM_SENSORS
        while True:
            try:
                cnt = serial_port.in_waiting
                if cnt > 0:
                    data_raw = serial_port.read(cnt)
                    umyo_parser.umyo_parse_preprocessor(data_raw)
                    sensors_proc = umyo_parser.umyo_get_list()

                    num_sensors = len(sensors_proc)
                    if num_sensors < self.NUM_SENSORS:
                        print(f"Sensors found: {str(num_sensors)}")
                        sleep(1)
                        continue

                    raw_data = np.zeros((self.NUM_SENSORS, 8), dtype=np.float32)
                    fft_data = np.zeros((self.NUM_SENSORS, 3), dtype=np.float32)
                    data_ids = [0] * self.NUM_SENSORS
                    for sensor_read in sensors_proc:
                        raw_data[IDS.index(sensor_read.unit_id)] = (
                            sensor_read.data_array[:8]
                        )
                        fft_data[IDS.index(sensor_read.unit_id)] = (
                            sensor_read.device_spectr[1:4]
                        )
                        data_ids[IDS.index(sensor_read.unit_id)] = sensor_read.data_id

                    if last_data_ids == data_ids:  # Skip if no new data
                        continue
                    last_data_ids = data_ids

                    raw_data = raw_data.mean(axis=1)
                    raw_data = raw_data.flatten()

                    with lock:
                        raw_data_queue.append(list(raw_data))  # Append averaged data
                        fft_data_queue.append(
                            list(fft_data)
                        )  # Append averaged FFT data

            except Exception as e:
                print(f"Data collection error: {e}")
                sleep(1)

    def classification(self):
        with self.lock:
            windowed_data = np.array(self.raw_data_queue)
            fft_data = np.array(self.fft_data_queue)

        if len(windowed_data) == self.window_size:
            windowed_data = np.array(windowed_data)
            fft_data = np.array(fft_data)
        else:
            return None

        for i in range(self.NUM_SENSORS):
            windowed_data[:, i] = self.preprocessor.preprocess(windowed_data[:, i], i)

        windowed_data = np.expand_dims(
            windowed_data, axis=0
        )  # Correct format for LSTM (batch, seq_len, num_sensors)
        features = self.get_features_per_sensor(
            windowed_data.swapaxes(1, 2), feature_groups=("HJORTH", "HTD")
        )

        if self.USE_FFT:
            fft_data = np.concatenate(
                [np.min(fft_data, axis=0), np.max(fft_data, axis=0)], axis=1
            )
            fft_data = fft_data.reshape(1, -1)
            features = np.hstack([features, fft_data])

        if self.model_type == "lstm":
            shape = windowed_data.shape
            data = self.scaler.transform(
                windowed_data.reshape(windowed_data.shape[0], -1)
            )
            data = torch.tensor(data.reshape(*shape)).float().to(self.device)
            prediction = self.model(data).argmax(dim=1).numpy().item()
        elif self.model_type == "torch" or self.model_type == "tf":
            features = self.scaler.transform(features)
            if self.model_type == "tf":
                n_features = self.NUM_SENSORS
                n_sequence = int(features.shape[1] / n_features)

                features = features.reshape(features.shape[0], n_features, n_sequence)
                features = np.swapaxes(features, 2, 1)
            data = torch.tensor(features, dtype=torch.float32).to(self.device)
            prediction = self.model(data).argmax(dim=1).to("cpu").numpy().item()
        elif self.model_type == "sklearn":
            features = self.scaler.transform(features)
            prediction = int(self.model.predict(features).squeeze())
        return prediction

    def _load_model(self):
        if self.model_type in ["torch", "lstm", "tf"]:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            self.model.to(self.device)
        elif self.model_type == "sklearn":
            with open(self.model_path, "rb") as f:
                self.model = load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = load(f)

        self.preprocessor = EMG_preprocessor()


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
    parser.add_argument(
        "-g",
        "--gestures",
        nargs="+",
        default=("baseline", "fist", "peace", "up", "down", "lift"),
        help="Set of gestures",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=200,
        help="Size of the data window for predictions",
    )
    parser.add_argument(
        "--prediction_delay", type=int, default=1, help="Delay between predictions"
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="USB receiving station port",
    )

    args = parser.parse_args()

    inference = EMG_Inference(
        port=args.port,
        model_path=args.model_path,
        model_type=args.model_type,
        scaler_path=args.scaler_path,
        window_size=args.window_size,
    )

    try:
        while True:
            prediction = inference.classification()

            if prediction is not None:
                print(f"Prediction: {args.gestures[prediction]}")
            sleep(args.prediction_delay)
    except KeyboardInterrupt:
        print("Stopped classification session.")
