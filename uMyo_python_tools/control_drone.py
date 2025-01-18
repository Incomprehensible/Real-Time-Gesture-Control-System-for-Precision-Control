import argparse
import time
from collections import deque

import torch
from classifier import EMG_Inference
from pymavlink import mavutil


def move_drone(x, y, z):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        int(0b100111111000),  # Type mask to ignore yaw and yaw rate, focus on position only
        x, y, z,              # Position in x, y, z 
        0, 0, 0,              # Velocity in m/s
        0, 0, 0,              # Acceleration
        0, 0,                 # Yaw, Yaw rate
    )

def turn_drone(degree):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        degree,  # Yaw angle (set to 360 for continuous rotation)
        50,
        1,  # Direction (1: CW, -1: CCW)
        1,  # Relative (1 for relative yaw change)
        0, 0, 0
    )

def get_current_altitude():
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        return msg.relative_alt / 1000.0  # Altitude in meters
    return None


def choose_movement_from_gesture(predictions):
    armed = master.motors_armed() != 0
    
    counts = [predictions.count(gesture) for gesture in gestures]
    
    if not PRED_LEN in counts:
        return False
    
    selected_gesture = gestures[counts.index(PRED_LEN)]
    
    if selected_gesture == 'fist':
        if not armed:
            print("ARMING")
            master.set_mode('GUIDED')
            time.sleep(0.5)
            master.arducopter_arm()
            time.sleep(0.5)
        return True
    
    if not armed:
        print("ARM WITH FIST FIRST")
        return True
    
    if selected_gesture == 'up':
        if get_current_altitude() < ALT_THRESHOLD:
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, 2
            )
        else:
            move_drone(0, 0, -0.5)
    elif selected_gesture == 'lift':
        move_drone(2, 0, 0)
        time.sleep(2)
    elif selected_gesture == 'peace':
        turn_drone(50)
        time.sleep(2)
    elif selected_gesture == 'down':
        if get_current_altitude() >= ALT_THRESHOLD:
            move_drone(0, 0, 0.5)
        else:
            land_drone()
    
    return True

def land_drone():
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        b'PLND_ENABLED',
        0,  # Disable precision landing
        mavutil.mavlink.MAV_PARAM_TYPE_INT8
    )
    time.sleep(2)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
        0, 0, 0, 0, 0, 0, 0
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drone control via EMG gesture commands")
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
    parser.add_argument("--window_size", type=int, default=200, help="Size of the data window for predictions")
    parser.add_argument("--prediction_delay", type=int, default=1, help="Delay between predictions")
    parser.add_argument("-p", "--port", type=str, default="/dev/ttyUSB0", help="USB receiving station port")
    
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    
    time.sleep(5)
        
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inference = EMG_Inference(port=args.port, model_path=args.model_path, model_type=args.model_type, scaler_path=args.scaler_path)
    
    PRED_LEN = 3
    gestures = ("baseline", "fist", "peace", "up", "down", "lift")
    ALT_THRESHOLD = 0.5
    
    predictions = deque([], maxlen=PRED_LEN)

    try:
        while True:
            master.wait_heartbeat(timeout=0.1) # Do heartbeat to keep connection alive (probably)
            prediction = inference.classification()

            if prediction is not None:
                gesture = gestures[prediction]
                predictions.append(gesture)
                print(f"Prediction: {gesture}")
                
                if len(predictions) == PRED_LEN:
                    
                    found_gesture = choose_movement_from_gesture(predictions)
                    
                    if found_gesture:
                        predictions = deque([], maxlen=PRED_LEN)
                time.sleep(args.prediction_delay)
    except KeyboardInterrupt:
        land_drone()
