import socket
import time
import hid
import json
import math
import csv
import numpy as np
from scipy.interpolate import splprep, splev

# HID Device Configuration
VENDOR_ID = 0x044f
PRODUCT_ID = 0xb66d

# Socket Configuration
host = '127.0.0.1'
port = 65432

def decode_angle(data, calibration_offset, scaling):
    """
    Decodes raw angle data from the HID device.
    """
    angle = int.from_bytes(data, byteorder='little', signed=True)
    calibrated_angle = angle - calibration_offset
    scaled_angle = calibrated_angle / scaling
    return scaled_angle

def calibrate_angle(device):
    """
    Calibrates initial angles for steering, brake, and accelerator.
    """
    data = device.read(64)
    if data:
        initial_steering = data[42:45]
        initial_steering_angle = int.from_bytes(initial_steering, byteorder='little', signed=True)

        initial_brake = data[48:51]
        initial_brake_angle = int.from_bytes(initial_brake, byteorder='little', signed=True)

        initial_accelerator = data[45:48]
        initial_accelerator_angle = int.from_bytes(initial_accelerator, byteorder='little', signed=True)
        return initial_steering_angle, initial_brake_angle, initial_accelerator_angle
    return 0, 0, 0

# Initialize server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(5)
print(f"Server listening on {host}:{port}")

# Initialize HID device
device = hid.Device(VENDOR_ID, PRODUCT_ID)
print(f"Connected to {device.manufacturer} - {device.product}")

# Calibrate initial angles
calibration_offset_steering, calibration_offset_brake, calibration_offset_accelerator = calibrate_angle(device)

# Accept client connection
client_socket, addr = server_socket.accept()
print(f"Connection from {addr}")

# Build CSV header with all required fields
header = [
    "time",
    # Vehicle State:
    "x_pos","z_pos", "yaw_angle", "long_vel", "lat_vel", "yaw_rate",
    # Driver inputs
    "steering", "throttle", "brake"
]

# Initialize CSV file and write header
csv_file = open("../../runs/training/raw/session3/run1.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(header)

try:
    while True:
        # Read data from HID device
        data = device.read(64)
        if data:
            # Decode steering, brake, and throttle angles
            steering_angle = decode_angle(data[42:45], calibration_offset_steering, 20000)
            if steering_angle < -600:
                steering_angle += 838.8
            if steering_angle > 90:
                steering_angle = 90
            if steering_angle < -90:
                steering_angle = -90
            steering_angle = steering_angle / 90

            brake_angle = decode_angle(data[48:51], calibration_offset_brake, 10)
            brake = -brake_angle / 25.5

            accelerator_angle = decode_angle(data[45:48], calibration_offset_accelerator, 100)
            if accelerator_angle > 0:
                accelerator_angle -= 327.5 * 2
            throttle = -accelerator_angle / 655

        # Prepare and send control inputs to the client
        message = f"{steering_angle},{throttle},{brake}\n"
        client_socket.sendall(message.encode())

        # Receive vehicle state data from the client
        raw_data = client_socket.recv(1024).decode('utf-8').strip()
        try:
            fields = raw_data.split(',')
            if len(fields) < 6:
                raise ValueError(f"Incomplete data received: {fields}")

            # Extract vehicle state (x_pos, z_pos, yaw, long_vel, lat_vel, yaw_rate)
            x_pos = float(fields[0])
            z_pos = float(fields[1])
            yaw = -float(fields[2]) + 90
            long_vel = float(fields[3])
            lat_vel = float(fields[4])  # lateral velocity
            yaw_rate = float(fields[5])

            row = [time.time(), x_pos, z_pos, yaw, long_vel, lat_vel, yaw_rate, steering_angle, throttle, brake]

            csv_writer.writerow(row)
            csv_file.flush()

        except ValueError as ve:
            print(f"Data parsing error: {ve}. Raw data: '{raw_data}'")
        except Exception as e:
            print(f"Unexpected error: {e}")

        time.sleep(0.01)

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    client_socket.close()
    device.close()
    server_socket.close()
    csv_file.close()
    print("Server and connection closed.")
