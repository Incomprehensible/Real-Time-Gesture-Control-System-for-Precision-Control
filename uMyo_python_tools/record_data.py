import argparse
import pathlib
from time import sleep

import serial
import umyo_parser

from parameters import IDS

MAX_DATA_LAG = 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from uMyo")
    parser.add_argument(
        "--output",
        type=str,
        default="sensor_data.csv",
        help="Output file to save sensor data",
    )
    parser.add_argument(
        "--num_sensors",
        type=int,
        default=4,
        help="Number of sensors to record",
    )
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

    output_path = pathlib.Path(args.output)
    if output_path.parent.exists() is False:
        output_path.parent.mkdir(parents=True)

    recordings = 0
    last_data_ids = [0] * args.num_sensors
    data_lag = [0] * args.num_sensors

    while True:
        try:
            cnt = ser.in_waiting
            if cnt > 0:
                data_raw = ser.read(cnt)
                parse_unproc_cnt = umyo_parser.umyo_parse_preprocessor(data_raw)
                sensors_proc = umyo_parser.umyo_get_list()

                num_sensors = len(sensors_proc)
                if num_sensors < args.num_sensors:
                    print("Sensors found: ", str(num_sensors))
                    sleep(1)
                    continue

                sensor_data = [[] for _ in range(args.num_sensors)]
                spectrum_data = [[] for _ in range(args.num_sensors)]
                imu_data = [[] for _ in range(args.num_sensors)]
                data_ids = [0] * args.num_sensors
                for sensor_read in sensors_proc:
                    data_ids[IDS.index(sensor_read.unit_id)] = sensor_read.data_id
                    sensor_data[IDS.index(sensor_read.unit_id)] = (
                        sensor_read.data_array[:8]
                    )
                    spectrum_data[IDS.index(sensor_read.unit_id)] = (
                        sensor_read.device_spectr[:4]
                    )
                    imu_data[IDS.index(sensor_read.unit_id)] = [
                        sensor_read.yaw,
                        sensor_read.pitch,
                        sensor_read.roll,
                        sensor_read.ax,
                        sensor_read.ay,
                        sensor_read.az,
                        sensor_read.mag_angle,
                    ]

                if last_data_ids == data_ids:  # Skip if no new data
                    continue
                data_lag = [data_lag[i] + 1 if data_ids[i] == last_data_ids[i] else 0 for i in range(args.num_sensors)]
                if any([lag > MAX_DATA_LAG for lag in data_lag]):
                    print('Data ids: ', data_ids)
                    print('sensors down: ', [i for i in range(args.num_sensors) if data_lag[i] > MAX_DATA_LAG])
                    print("Data lag too high, some sensors probably died, skipping recording.")
                    exit(1)
                last_data_ids = data_ids

                flattened_data = [item for sublist in sensor_data for item in sublist]
                flattened_fft = [item for sublist in spectrum_data for item in sublist]
                flattened_imu = [item for sublist in imu_data for item in sublist]

                with open(output_path, "a") as f:
                    f.write(",".join(map(str, flattened_data)) + "\n")
                with open(output_path.parent / ("fft_" + output_path.name), "a") as f:
                    f.write(",".join(map(str, flattened_fft)) + "\n")
                with open(output_path.parent / ("imu_" + output_path.name), "a") as f:
                    f.write(",".join(map(str, flattened_imu)) + "\n")
                recordings += 1
                print("Recordings done: ", recordings)
        except KeyboardInterrupt:
            print("Finishing recording session.")
            break
