import argparse
from pickle import load
from time import sleep

import serial
import torch
import torch.nn as nn
import umyo_parser
from scipy import signal
from scipy.signal import butter


# TODO: Save model architecture, classes and preprocessing parameters in pickle file
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.softmax(self.fc3(x), dim=1)
        return x


lf = 50
hf = 400

def preprocess_data(data):
    sos = butter(3, (lf, hf), btype="bandpass", fs=1150, output="sos")
    return signal.sosfilt(sos, data)

ser = serial.Serial(
    port="COM7",
    baudrate=921600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=0,
)

ids = [1633709441, 3274504362, 2749159433, 3048451580]

classes = ["neutral", "fist", "index", "middle", "ok", "peace", "thumb"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from uMyo")
    parser.add_argument(
        "--model_type",
        choices=["sklearn", "torch"],
        default="sklearn",
        help="Type of model to use for classification",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../pretraining/custom_classifier.pkl",
        help="Path to classifier",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="../pretraining/custom_scaler.pkl",
        help="Path to scaler",
    )
    args = parser.parse_args()
    
    if args.model_type == "torch":
        model = NeuralNet(32, 3)
        model.load_state_dict(torch.load(args.model_path))
    else:
        with open(args.model_path, "rb") as f:
            model = load(f)
    with open(args.scaler_path, "rb") as f:
        scaler = load(f)
    
    while True:
        try:
            cnt = ser.in_waiting
            if cnt > 0:
                data_raw = ser.read(cnt)
                parse_unproc_cnt = umyo_parser.umyo_parse_preprocessor(data_raw)
                sensors_proc = umyo_parser.umyo_get_list()

                num_sensors = len(sensors_proc)
                if num_sensors < 4:
                    print("Sensors found: ", str(num_sensors))
                    sleep(1)
                    continue

                sensor_data = [[], [], [], []]
                for sensor_read in sensors_proc:
                    sensor_data[ids.index(sensor_read.unit_id)] = (
                        sensor_read.data_array[:8]
                    )
                flattened_data = [item for sublist in sensor_data for item in sublist]
                
                transformed_data = scaler.transform(preprocess_data([flattened_data]))
                if args.model_type == "torch":
                    prediction = model(torch.tensor(transformed_data).float())
                    prediction = prediction.argmax(dim=1).numpy().item()
                else:
                    prediction = model.predict(transformed_data)[0]
                
                print("Prediction: ", classes[prediction])
                sleep(0.5) # Only print predictions every 0.5 seconds
        except KeyboardInterrupt:
            print("Stopped classification session.")
            break