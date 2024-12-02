import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter
import os

lf = 60
hf = 400


def preprocess_data(data):
    sos = butter(3, (lf, hf), btype="bandpass", fs=1150, output="sos")
    return signal.sosfilt(sos, data)


def show_all_gestures():
    fig, axs = plt.subplots(4, 4, figsize=(24, 16))
    axs = axs.flatten()
    for i in range(16):
        df = pd.read_csv("recordings/nad_fist_" + str(i) + ".csv", header=None)
        sensor1 = np.array(df.iloc[:, :8]).flatten()
        sensor2 = np.array(df.iloc[:, 8:16]).flatten()
        sensor3 = np.array(df.iloc[:, 16:24]).flatten()
        sensor4 = np.array(df.iloc[:, 24:]).flatten()
        axs[i].set_title("Gesture " + str(i))
        axs[i].plot(sensor1)
        axs[i].plot(sensor2)
        axs[i].plot(sensor3)
        axs[i].plot(sensor4)
    plt.show()


parser = argparse.ArgumentParser(description="Visualize data from uMyo")
parser.add_argument(
    "--folder", type=str, default="recordings", help="Folder with recordings"
)
args = parser.parse_args()

folder = args.folder
gestures_files = [
    os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".csv")
]
print(gestures_files)
gestures_data = [[] for _ in range(len(gestures_files))]
print(gestures_data)

for i, file in enumerate(gestures_files):
    df = pd.read_csv(file, header=None)
    gestures_data[i].append(preprocess_data(np.array(df.iloc[:, :8]).flatten()))
    gestures_data[i].append(preprocess_data(np.array(df.iloc[:, 8:16]).flatten()))
    gestures_data[i].append(preprocess_data(np.array(df.iloc[:, 16:24]).flatten()))
    gestures_data[i].append(preprocess_data(np.array(df.iloc[:, 24:]).flatten()))

    fig, axs = plt.subplots(2, 2, figsize=(24, 16))
    axs = axs.flatten()
    axs[0].set_title("Sensor 1")
    axs[0].plot(sensor1)
    axs[1].set_title("Sensor 2")
    axs[1].plot(sensor2)
    axs[2].set_title("Sensor 3")
    axs[2].plot(sensor3)
    axs[3].set_title("Sensor 4")
    axs[3].plot(sensor4)
    fig.suptitle(file)

    sensor1 = preprocess_data(sensor1)
    sensor2 = preprocess_data(sensor2)
    sensor3 = preprocess_data(sensor3)
    sensor4 = preprocess_data(sensor4)
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
    fig2.suptitle(file + " filtered: (" + str(lf) + ", " + str(hf) + ")")
    plt.show()

# show neutral gesture for reference
df = pd.read_csv(args.file, header=None)
sensor1 = np.array(df.iloc[:, :8]).flatten()
sensor2 = np.array(df.iloc[:, 8:16]).flatten()
sensor3 = np.array(df.iloc[:, 16:24]).flatten()
sensor4 = np.array(df.iloc[:, 24:]).flatten()

fig, axs = plt.subplots(2, 2, figsize=(24, 16))
axs = axs.flatten()
axs[0].set_title("Sensor 1")
axs[0].plot(sensor1)
axs[1].set_title("Sensor 2")
axs[1].plot(sensor2)
axs[2].set_title("Sensor 3")
axs[2].plot(sensor3)
axs[3].set_title("Sensor 4")
axs[3].plot(sensor4)
fig.suptitle(args.file)

sensor1 = preprocess_data(sensor1)
sensor2 = preprocess_data(sensor2)
sensor3 = preprocess_data(sensor3)
sensor4 = preprocess_data(sensor4)
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
plt.show()
plt.show()
