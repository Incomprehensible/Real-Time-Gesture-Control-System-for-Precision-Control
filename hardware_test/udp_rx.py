import socket
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# simple UDP server to receive EMG data from 1 sensor and animate Average EMG and FFT data in real-time

# UDP server details
UDP_IP = "192.168.123.20"
UDP_PORT = 3333

# Buffer size for receiving UDP packets
BUFFER_SIZE = 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Listening on {UDP_IP}:{UDP_PORT}...")

bins = 3
max_length = 500
timestamps1 = []
timestamps2 = []
fft_history = [[] for _ in range(bins)]
avg_history = []

# Initialize plot
# fig, ax = plt.subplots()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
lines1 = [ax1.plot([], [], label=f"FFT {i}")[0] for i in range(2, 5)]
lines2 = [ax2.plot([], [], label=f"Average EMG")[0]]

ax1.set_xlim(0, max_length)  # history size
ax1.set_ylim(-1.5, 1.5)  # FFT value range
ax1.set_title("Real-Time FFT Data")
ax1.set_xlabel("Time")
ax1.set_ylabel("FFT Amplitude")
ax1.legend()

ax2.set_xlim(0, max_length)
ax2.set_ylim(-1500, 1500) 
ax2.set_title("Real-Time Average EMG Data")
ax2.set_xlabel("Time")
ax2.set_ylabel("Amplitude")
ax2.legend()

decoded = None

def update_fft_plot(frame):
    global decoded, timestamps1, fft_history
    
    try:
        # Receive UDP data
        data, _ = sock.recvfrom(BUFFER_SIZE)
        decoded = data.decode('utf-8')
        
        # Parse the FFT data
        if "FFT" in decoded:
            fields = decoded.split(',')
            fft_data = [float(fields[5].split(':')[1].strip().split()[i]) for i in range(1, 4)]
            print('FFT: ', fft_data)
            fft_vec = np.array(fft_data)
            fft_vec /= np.linalg.norm(fft_vec)
            print('Normalized FFT: ', fft_vec)
            
            # Update data history
            timestamps1.append(len(timestamps1))
            for i in range(0, bins):
                fft_history[i].append(fft_vec[i])
            
            # Trim history to fit plot
            if len(timestamps1) >= max_length:
                timestamps1 = []
                for i in range(0, bins):
                    fft_history = [[] for _ in range(bins)]
            
            # Update lines
            for i, line in enumerate(lines1):
                line.set_data(timestamps1, fft_history[i])
            
        elif 'AverageEMG' in decoded:
            fields = decoded.split(',')
            avg_data = float(fields[4].split(':')[1].strip())
            print('Average EMG: ', avg_data)
            timestamps2.append(len(timestamps2))
            avg_history.append(avg_data)

            if len(timestamps2) >= max_length:
                timestamps2 = []
                avg_history = []
    
    except Exception as e:
        print(f"Error: {e}")

    # Adjust plot limits dynamically
    #ax.relim()
    #ax.autoscale_view()
    #ax.set_xlim(0, len(timestamps))
    return lines1


def update_avg_plot(frame):
    global decoded, timestamps2, avg_history
    
    try:
        if decoded is None:
            return lines2
            
        # data, _ = sock.recvfrom(BUFFER_SIZE)
        # decoded = data.decode('utf-8')
            
        if 'AverageEMG' in decoded:
            fields = decoded.split(',')
            avg_data = float(fields[4].split(':')[1].strip())
            print('Average EMG: ', avg_data)
            timestamps2.append(len(timestamps2))
            avg_history.append(avg_data)

            if len(timestamps2) >= max_length:
                timestamps2 = []
                avg_history = []
            
            for line in lines2:
                line.set_data(timestamps2, avg_history)
    
    except Exception as e:
        print(f"Error: {e}")

    return lines2

def init_fft():
    for line in lines1:
        line.set_data([], [])
    return lines1

def init_avg():
    for line in lines2:
        line.set_data([], [])
    return lines1

# Start animation
ani_fft = FuncAnimation(fig, update_fft_plot, init_func=init_fft, interval=1, blit=True)
ani_avg = FuncAnimation(fig, update_avg_plot, init_func=init_avg, interval=1, blit=True)

plt.show()
