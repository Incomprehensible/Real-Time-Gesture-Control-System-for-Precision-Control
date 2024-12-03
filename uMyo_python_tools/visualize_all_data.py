import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter
from numpy import mean, std
import os

fs = 1150
lf = 15
hf = 400
trim = 4*8*5
bandpass_order = 3
outlier_rejection_stds = 6
power_noise = 1.0
power_noise_filtered = 1.0

def _remove_outliers(data):
    data_mean, data_std = mean(data), std(data)
    cut_off = data_std * outlier_rejection_stds
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x if x >= lower and x <= upper else 0.0 for x in data]
    return outliers_removed

# def remove_outliers(data):
#     threshold_val = abs(data).max()
#     threshold = 0.45 * threshold_val
#     data = np.array(
#         [
#             val if (abs(val) < threshold or i > (len(data) - 1)) else 0.0
#             for i, val in enumerate(data)
#         ]
#     )
#     return data

def _remove_artefact(data):
    data[:trim] = 0
    return data

# calculation: https://www.electronics-tutorials.ws/filter/band-stop-filter.html
def _apply_notch_filter(data):
    f0 = 50
    # Q = 30
    fn_l = 49
    fn_h = 51
    fn_c = np.sqrt(fn_h * fn_l)
    fn_bw = fn_h - fn_l
    Q = fn_c / fn_bw
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, data)

def _butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos

def _apply_bandpass(data, lowcut, highcut, fs, order=5):
    sos = _butter_bandpass(lowcut, highcut, fs, order=order)
    # use sosfiltfilt for zero-phase filtering
    # use sosfilt for real-time filtering
    return signal.sosfiltfilt(sos, data)

def _calculate_baseline_noise(neutral_signal):
    return np.sqrt(np.sum(np.asanyarray(neutral_signal)**2))

# def signaltonoise(a, axis=0, ddof=0, filtered=False):
def signaltonoise(signal, filtered=False):
    global power_noise, power_noise_filtered

    if filtered:
        power_noise_ = power_noise_filtered
    else:
        power_noise_ = power_noise
    power_signal = np.sqrt(np.sum(np.asanyarray(signal)**2))
    snr = 10 * np.log10(power_signal / power_noise_)
    return snr

    # a = np.asanyarray(a)
    # m = a.mean(axis)
    # sd = a.std(axis=axis, ddof=ddof)
    # return np.where(sd == 0, 0, m/sd)

from scipy.signal import freqz

def show_bandpass():
    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = _butter_bandpass(lf, hf, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

def preprocess_data(data):
    data = _apply_bandpass(data, lf, hf, fs, order=bandpass_order)
    # sos = butter(bandpass_order, (lf, hf), btype="bandpass", fs=fs, output="sos")
    # data = signal.sosfilt(sos, data)
    # data = _apply_notch_filter(data)
    data = _remove_artefact(data)
    data = _remove_outliers(data)
    return data

def show_neutral(folder, subject, placement):
    global power_noise, power_noise_filtered
    file = os.path.join(folder, subject + "_neutral_" + str(placement) + ".csv")
    df = pd.read_csv(file, header=None)
    sensor1 = np.array(df.iloc[:, :8]).flatten() #* (1/fs)
    sensor1 = sensor1 - np.mean(sensor1)
    sensor2 = np.array(df.iloc[:, 8:16]).flatten()
    sensor2 = sensor2 - np.mean(sensor2)
    sensor3 = np.array(df.iloc[:, 16:24]).flatten()
    sensor3 = sensor3 - np.mean(sensor3)
    sensor4 = np.array(df.iloc[:, 24:]).flatten()
    sensor4 = sensor4 - np.mean(sensor4)
    time1 = np.array([i/fs for i in range(0, len(sensor1), 1)]) # sampling rate 1000 Hz
    time2 = np.array([i/fs for i in range(0, len(sensor2), 1)])
    time3 = np.array([i/fs for i in range(0, len(sensor3), 1)])
    time4 = np.array([i/fs for i in range(0, len(sensor4), 1)])

    if power_noise == 1.0:
        power_noise = _calculate_baseline_noise(np.concatenate([sensor1, sensor2, sensor3, sensor4]))

    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(file)
    axs = axs.flatten()
    axs[0].set_title("Sensor 1")
    axs[0].plot(time1, sensor1)
    axs[1].set_title("Sensor 2")
    axs[1].plot(time2, sensor2)
    axs[2].set_title("Sensor 3")
    axs[2].plot(time3, sensor3)
    axs[3].set_title("Sensor 4")
    axs[3].plot(time4, sensor4)

    # plt.text(0.0, -0.3, f"SNR for sensor 1: {signaltonoise(sensor1):.2f} [dB]", 
    #          horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes)
    # plt.text(0.0, -0.3, f"SNR for sensor 2: {signaltonoise(sensor2):.2f} [dB]", 
    #          horizontalalignment='left', verticalalignment='center', transform=axs[1].transAxes)
    # plt.text(0.0, -0.3, f"SNR for sensor 3: {signaltonoise(sensor3):.2f} [dB]",
    #         horizontalalignment='left', verticalalignment='center', transform=axs[2].transAxes)
    # plt.text(0.0, -0.3, f"SNR for sensor 4: {signaltonoise(sensor4):.2f} [dB]",
    #         horizontalalignment='left', verticalalignment='center', transform=axs[3].transAxes)
    
    sensor1 = preprocess_data(sensor1)
    sensor2 = preprocess_data(sensor2)
    sensor3 = preprocess_data(sensor3)
    sensor4 = preprocess_data(sensor4)

    if power_noise_filtered == 1.0:
        power_noise_filtered = _calculate_baseline_noise(np.concatenate([sensor1, sensor2, sensor3, sensor4]))

    axs[4].set_title("Filtered sensor 1")
    axs[4].plot(time1, sensor1)
    axs[5].set_title("Filtered sensor 2")
    axs[5].plot(time2, sensor2)
    axs[6].set_title("Filtered sensor 3")
    axs[6].plot(time3, sensor3)
    axs[7].set_title("Filtered sensor 4")
    axs[7].plot(time4, sensor4)
    # plt.text(0.0, -0.3, f"SNR for sensor 1: {signaltonoise(sensor1, filtered=True):.2f} [dB]", 
    #          horizontalalignment='left', verticalalignment='center', transform=axs[4].transAxes)
    # plt.text(0.0, -0.3, f"SNR for sensor 2: {signaltonoise(sensor2, filtered=True):.2f} [dB]",
    #         horizontalalignment='left', verticalalignment='center', transform=axs[5].transAxes)
    # plt.text(0.0, -0.3, f"SNR for sensor 3: {signaltonoise(sensor3, filtered=True):.2f} [dB]",
    #         horizontalalignment='left', verticalalignment='center', transform=axs[6].transAxes)
    # plt.text(0.0, -0.3, f"SNR for sensor 4: {signaltonoise(sensor4, filtered=True):.2f} [dB]",
    #         horizontalalignment='left', verticalalignment='center', transform=axs[7].transAxes)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

def show_all_gestures(gestures_files, folder, subject, placement):
    show_neutral(folder, subject, placement) # show neutral gesture for reference
    gestures_data = [[] for _ in range(len(gestures_files))]

    for i, file in enumerate(gestures_files):
        df = pd.read_csv(file, header=None)
        gestures_data[i].append(np.array(df.iloc[:, :8]).flatten())
        gestures_data[i].append(np.array(df.iloc[:, 8:16]).flatten())
        gestures_data[i].append(np.array(df.iloc[:, 16:24]).flatten())
        gestures_data[i].append(np.array(df.iloc[:, 24:]).flatten())
        gestures_data[i] = [data - np.mean(data) for data in gestures_data[i]]
        time1 = np.array([i/fs for i in range(0, len(gestures_data[i][0]), 1)]) # sampling rate 1000 Hz
        time2 = np.array([i/fs for i in range(0, len(gestures_data[i][1]), 1)])
        time3 = np.array([i/fs for i in range(0, len(gestures_data[i][2]), 1)])
        time4 = np.array([i/fs for i in range(0, len(gestures_data[i][3]), 1)])

        fig, axs = plt.subplots(4, 2, figsize=(16, 16))
        axs = axs.flatten()
        axs[0].set_title("Sensor 1")
        axs[0].plot(time1, gestures_data[i][0])
        axs[1].set_title("Sensor 2")
        axs[1].plot(time2, gestures_data[i][1])
        axs[2].set_title("Sensor 3")
        axs[2].plot(time3, gestures_data[i][2])
        axs[3].set_title("Sensor 4")
        axs[3].plot(time4, gestures_data[i][3])

        plt.text(0.0, -0.3, f"SNR for sensor 1: {signaltonoise(gestures_data[i][0]):.2f} [dB]", 
                horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes)
        plt.text(0.0, -0.3, f"SNR for sensor 2: {signaltonoise(gestures_data[i][1]):.2f} [dB]", 
                 horizontalalignment='left', verticalalignment='center', transform=axs[1].transAxes)
        plt.text(0.0, -0.3, f"SNR for sensor 3: {signaltonoise(gestures_data[i][2]):.2f} [dB]",
                horizontalalignment='left', verticalalignment='center', transform=axs[2].transAxes)
        plt.text(0.0, -0.3, f"SNR for sensor 4: {signaltonoise(gestures_data[i][3]):.2f} [dB]",
                horizontalalignment='left', verticalalignment='center', transform=axs[3].transAxes)
    
        sensor1 = preprocess_data(gestures_data[i][0])
        sensor2 = preprocess_data(gestures_data[i][1])
        sensor3 = preprocess_data(gestures_data[i][2])
        sensor4 = preprocess_data(gestures_data[i][3])
        axs[4].set_title("Filtered sensor 1")
        axs[4].plot(time1, sensor1)
        axs[5].set_title("Filtered sensor 2")
        axs[5].plot(time2, sensor2)
        axs[6].set_title("Filtered sensor 3")
        axs[6].plot(time3, sensor3)
        axs[7].set_title("Filtered sensor 4")
        axs[7].plot(time4, sensor4)

        plt.text(0.0, -0.3, f"SNR for sensor 1: {signaltonoise(sensor1, filtered=True):.2f} [dB]", 
                horizontalalignment='left', verticalalignment='center', transform=axs[4].transAxes)
        plt.text(0.0, -0.3, f"SNR for sensor 2: {signaltonoise(sensor2, filtered=True):.2f} [dB]",
                horizontalalignment='left', verticalalignment='center', transform=axs[5].transAxes)
        plt.text(0.0, -0.3, f"SNR for sensor 3: {signaltonoise(sensor3, filtered=True):.2f} [dB]",
                horizontalalignment='left', verticalalignment='center', transform=axs[6].transAxes)
        plt.text(0.0, -0.3, f"SNR for sensor 4: {signaltonoise(sensor4, filtered=True):.2f} [dB]",
                horizontalalignment='left', verticalalignment='center', transform=axs[7].transAxes)
    
        fig.suptitle(file)
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data from uMyo")
    
    parser.add_argument(
        "--folder", type=str, default="../recordings/01_12_24_initial_placement_test/data", help="Folder with recordings"
    )
    parser.add_argument(
        "--subject", type=str, default="nad", help="Test subject name"
    )
    parser.add_argument(
        "--placement", type=int, default=0, help="Placement version of the sensors"
    )

    args = parser.parse_args()

    folder = args.folder
    subject = args.subject
    placement = args.placement
    gestures_files = [
        os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(str(placement)+".csv") and file.startswith(subject)
    ]

    show_all_gestures(gestures_files, folder, subject, placement)
