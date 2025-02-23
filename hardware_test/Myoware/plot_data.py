import argparse
import sys

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean, std
from scipy import signal
from scipy.signal import butter

sys.append("../uMyo_python_tools")
from parameters import BANDPASS_ORDER, FS, HF, LF, OUTLIER_REJECTION_STDS, TRIM

power_noise = 1.0
power_noise_filtered = 1.0


def _butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="bandpass", output="sos")
    return sos


def _apply_bandpass(data, lowcut, highcut, fs, order=5):
    sos = _butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.sosfiltfilt(sos, data)


def _remove_artefact(data):
    data[:TRIM] = 0
    return data


def _remove_outliers(data):
    data_mean, data_std = mean(data), std(data)
    cut_off = data_std * OUTLIER_REJECTION_STDS
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x if x >= lower and x <= upper else 0.0 for x in data]
    return outliers_removed


def preprocess_data(data):
    data = _apply_bandpass(data, LF, HF, fs, order=BANDPASS_ORDER)
    data = _remove_artefact(data)
    data = _remove_outliers(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data from uMyo")
    parser.add_argument(
        "--file", type=str, default="gilbert_raw.csv", help="File to visualize"
    )
    parser.add_argument("--type", type=str, default="umyo", help="Type of sensor")
    args = parser.parse_args()

    df = pd.read_csv(args.file, dtype="float", header=None)

    data = df.to_numpy().flatten()

    if "umyo" in args.type:
        data = preprocess_data(data)
        fs = FS
    else:
        fs = 8000

    # remove first 10000 samples
    # data = data[10000:]

    print(f"Max value: {max(data)}")
    print(f"Min value: {min(data)}")

    plt.subplot(211)
    plt.plot(data, label="Raw EMG Signal")
    plt.subplot(212)
    plt.psd(data, NFFT=512, Fs=fs, window=mlab.window_none, scale_by_freq=False)
    plt.show()
