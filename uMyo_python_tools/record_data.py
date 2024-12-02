import argparse
from time import sleep

import serial
import umyo_parser

ser = serial.Serial(
    port="COM7",
    baudrate=921600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=0,
)

ids = [1633709441, 3274504362, 2749159433, 3048451580]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from uMyo")
    parser.add_argument(
        "--output",
        type=str,
        default="sensor_data.csv",
        help="Output file to save sensor data",
    )
    args = parser.parse_args()
    recordings = 0
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

                with open(args.output, "a") as f:
                    f.write(",".join(map(str, flattened_data)) + "\n")
                recordings += 1
                print("Recordings done: ", recordings)
        except KeyboardInterrupt:
            print("Finishing recording session.")
            break
