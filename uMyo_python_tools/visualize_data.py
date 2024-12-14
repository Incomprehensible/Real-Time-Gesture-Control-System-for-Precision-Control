import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean, std
from scipy import signal
from scipy.signal import butter

# OBSOLETE SCRIPT

lf = 15 #50
hf = 400

def preprocess_data(data):
    sos = butter(3, (lf, hf), btype="bandpass", fs=1150, output="sos")
    return signal.sosfilt(sos, data)

def remove_outliers1(data):
    data_mean, data_std = mean(data), std(data)
    cut_off = data_std * 6
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    return outliers_removed

def remove_outliers2(data):
    threshold_val = abs(data).max()
    threshold = 0.45 * threshold_val
    data = np.array(
        [
            val if (abs(val) < threshold or i > (len(data) - 1)) else 0.0
            for i, val in enumerate(data)
        ]
    )
    return data

def plot_no_outliers(sensors_data, filename):
    fig, axs = plt.subplots(2, 2, figsize=(24, 16))
    axs = axs.flatten()
    axs[0].set_title("Filtered sensor 1")
    axs[0].plot(sensors_data[0])
    axs[1].set_title("Filtered sensor 2")
    axs[1].plot(sensors_data[1])
    axs[2].set_title("Filtered sensor 3")
    axs[2].plot(sensors_data[2])
    axs[3].set_title("Filtered sensor 4")
    axs[3].plot(sensors_data[3])
    fig.suptitle(filename + " without outliers")
    plt.show()


parser = argparse.ArgumentParser(description="Visualize data from uMyo")
parser.add_argument(
    "--file", type=str, default="teacher_fist_0.csv", help="File to visualize"
)
args = parser.parse_args()

df = pd.read_csv(args.file, header=None)
sensor1 = np.array(df.iloc[:, :8]).flatten()
sensor2 = np.array(df.iloc[:, 8:16]).flatten()
sensor3 = np.array(df.iloc[:, 16:24]).flatten()
sensor4 = np.array(df.iloc[:, 24:]).flatten()

# fig, axs = plt.subplots(2, 2, figsize=(24, 16))
# axs = axs.flatten()
# axs[0].set_title('Sensor 1')
# axs[0].plot(sensor1)
# axs[1].set_title('Sensor 2')
# axs[1].plot(sensor2)
# axs[2].set_title('Sensor 3')
# axs[2].plot(sensor3)
# axs[3].set_title('Sensor 4')
# axs[3].plot(sensor4)
# fig.suptitle(args.file)

trim = 50
sensor1 = preprocess_data(sensor1)
sensor1 = sensor1[trim:]
sensor2 = preprocess_data(sensor2)
sensor2 = sensor2[trim:]
sensor3 = preprocess_data(sensor3)
sensor3 = sensor3[trim:]
sensor4 = preprocess_data(sensor4)
sensor4 = sensor4[trim:]
fig2, axs2 = plt.subplots(2, 2, figsize=(24, 16))
axs2 = axs2.flatten()
axs2[0].set_title("Filtered sensor 1")
axs2[0].plot(sensor1)
axs2[1].set_title("Filtered sensor 2")
axs2[1].plot(sensor2)
axs2[2].set_title("Filtered sensor 3")
axs2[2].plot(sensor3)
axs2[3].set_title("Filtered sensor 4")
axs2[3].plot(sensor4)
fig2.suptitle(args.file + " filtered: (" + str(lf) + ", " + str(hf) + ")")
# plt.show()
# plt.show()

sensors = [sensor1, sensor2, sensor3, sensor4]

for s in range(4):
    sensors[s] = remove_outliers1(sensors[s])
    # threshold_val = abs(sensors[s]).max()
    # threshold = 0.45 * threshold_val
    # sensors[s] = np.array([val if (abs(val) < threshold or i > (len(sensors[s]) - 1)) else 0.0 for i, val in enumerate(sensors[s])])

    # sensors[s] = np.array([val if (abs(val) < threshold or i > (len(sensors[s]) - 1)) else sensors[s][i+1] for i, val in enumerate(sensors[s])])


plot_no_outliers(sensors, args.file)
