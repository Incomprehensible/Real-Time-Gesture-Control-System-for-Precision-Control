import argparse
import time
from collections import deque

import torch
from classifier import EMG_Inference
from djitellopy import Tello

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_LEN = 3
gestures = ("baseline", "fist", "peace", "up", "down", "lift")
ALT_THRESHOLD = 0.5

predictions = deque([], maxlen=PRED_LEN)


class RyzeTello:
    def __init__(self, inference, prediction_delay):
        # Initialize Tello object
        self.tello = Tello()

        # Initialize inference object
        self.inference = inference
        self.prediction_delay = prediction_delay

    def _choose_movement_from_gesture(self, predictions):
        counts = [predictions.count(gesture) for gesture in gestures]

        if not PRED_LEN in counts:
            return False

        selected_gesture = gestures[counts.index(PRED_LEN)]

        if selected_gesture == "fist":
            if not self.tello.is_flying:
                self.tello.takeoff()
            return True

        if not self.tello.is_flying:
            print("Drone needs to takeoff before executing any other commands")
            return True

        if selected_gesture == "up":
            self.tello.move_up(50)
        elif selected_gesture == "lift":
            self.tello.move_forward(50)
        elif selected_gesture == "peace":
            self.tello.rotate_clockwise(50)
        elif selected_gesture == "down":
            if self.tello.get_height() >= ALT_THRESHOLD * 100:
                self.tello.move_down(50)
            else:
                self.tello.land()

        return True

    def run(self):
        global predictions
        self.tello.connect()

        try:
            while True:
                prediction = self.inference.classification()

                if prediction is not None:
                    gesture = gestures[prediction]
                    predictions.append(gesture)
                    print(f"Prediction: {gesture}")

                    if len(predictions) == PRED_LEN:
                        found_gesture = self._choose_movement_from_gesture(predictions)

                        if found_gesture:
                            predictions = deque([], maxlen=PRED_LEN)
                    time.sleep(self.prediction_delay)
        except KeyboardInterrupt:
            self.tello.land()
            self.tello.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Drone control via EMG gesture commands"
    )
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
    )

    drone = RyzeTello(inference, args.prediction_delay)
    drone.run()
