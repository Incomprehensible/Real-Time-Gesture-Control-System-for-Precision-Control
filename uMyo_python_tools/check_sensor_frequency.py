
import argparse
import threading
import time

import serial
import umyo_parser

IDS = [1633709441, 3274504362, 2749159433, 3048451580, 3899692357]
PRINT_DELAY = 2 # seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMG sensors data rate test")
    parser.add_argument("-p", "--port", type=str, default="COM7", help="USB receiving station port")
    args = parser.parse_args()

    ser = serial.Serial(
        port=args.port,
        baudrate=921600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0,
    )

    recordings = 0
    last_data_ids = {}
    new_value_per_sensor_counts = {}
    
    def print_sensor_data():
        while True:
            time.sleep(PRINT_DELAY)
            if len(new_value_per_sensor_counts) == 0:
                continue
            for i, id in enumerate(IDS):
                print(f"Sensor {i + 1}: {new_value_per_sensor_counts.get(id, 0) / float(PRINT_DELAY)},", end=" ")
            print()
            new_value_per_sensor_counts.clear()

    thread = threading.Thread(target=print_sensor_data)
    thread.daemon = True
    thread.start()
    
    while True:
        try:
            cnt = ser.in_waiting
            if cnt > 0:
                data_raw = ser.read(cnt)
                parse_unproc_cnt = umyo_parser.umyo_parse_preprocessor(data_raw)
                sensors_proc = umyo_parser.umyo_get_list()
                
                old_counts = new_value_per_sensor_counts.copy()
                
                for sensor_read in sensors_proc:
                    if sensor_read.unit_id not in last_data_ids:
                        last_data_ids[sensor_read.unit_id] = 0
                    if sensor_read.unit_id not in new_value_per_sensor_counts:
                        new_value_per_sensor_counts[sensor_read.unit_id] = 0
                    
                    if last_data_ids[sensor_read.unit_id] != sensor_read.data_id:
                        new_value_per_sensor_counts[sensor_read.unit_id] += 8
                        last_data_ids[sensor_read.unit_id] = sensor_read.data_id

                if new_value_per_sensor_counts == old_counts:  # Skip if no new data
                    continue
                recordings += 1
                
        except KeyboardInterrupt:
            print("Finishing recording session.")
            break
