from enum import Enum

IDS = [1633709441, 3274504362, 2749159433, 3048451580, 3899692357]

LF = 20
HF = 500
FS = 1150
TRIM = 4 * 8 * 5
BANDPASS_ORDER = 4
OUTLIER_REJECTION_STDS = 6

NUM_SENSORS = 5
NUM_FFT_READINGS = 4
USE_FFT = True

class DATASET(Enum):
    HUGGING_FACE = 1
    LOCAL = 2
    BOTH = 3

GESTURES_TYPE = ("dynamic_gestures", "static_gestures")
GESTURES = ("baseline", "fist", "peace", "up", "down", "lift")

DATASET_SOURCE = DATASET.HUGGING_FACE