import argparse
import time

import serial

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', default=20*2, help='Time recording should last in seconds')
    parser.add_argument('--port', default='COM18', help='Serial port to connect to')
    parser.add_argument('--baudrate', default='115200', help='Baudrate to use')
    parser.add_argument('--output', default='data.csv', help='Output file to write data to')
    args = parser.parse_args()
    
    with open(args.output, 'w') as f:
        f.write(f"") # Clear file

    ser = serial.Serial(args.port, args.baudrate)
    start_time = time.time()
    with open(args.output, 'a') as f:
        while time.time() - start_time < int(args.time):
            reading = ser.readline().decode('utf-8').strip()
            f.write(f"{reading}\n")
