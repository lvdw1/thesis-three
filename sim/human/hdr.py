#!/usr/bin/env python3
import socket
import time
import hid
import json
import math
import csv
import numpy as np
from scipy.interpolate import splprep, splev
import argparse

# HID Device Configuration
VENDOR_ID = 0x044f
PRODUCT_ID = 0xb66d

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
        initial_steering_angle = abs(int.from_bytes(initial_steering, byteorder='little', signed=True))

        initial_brake = data[48:51]
        initial_brake_angle = int.from_bytes(initial_brake, byteorder='little', signed=True)

        initial_accelerator = data[45:48]
        initial_accelerator_angle = int.from_bytes(initial_accelerator, byteorder='little', signed=True)
        return initial_steering_angle, initial_brake_angle, initial_accelerator_angle
    return 0, 0, 0

def process_hid_input(device, calibration_offsets):
    """
    Reads from the HID device and returns processed values for steering, throttle, and brake.
    Returns a tuple: (steering_angle, throttle, brake) or None if no data.
    """
    data = device.read(64)
    if data:
        # Decode steering angle
        steering_angle = decode_angle(data[42:45], calibration_offsets[0], 20000)
        if steering_angle < -600:
            steering_angle += 838.8
        if steering_angle > 90:
            steering_angle = 90
        if steering_angle < -90:
            steering_angle = -90
        steering_angle = steering_angle / 90

        # Decode brake angle
        brake_angle = decode_angle(data[48:51], calibration_offsets[1], 10)
        brake = -brake_angle / 25.5

        # Decode accelerator (for throttle)
        accelerator_angle = decode_angle(data[45:48], calibration_offsets[2], 100)
        if accelerator_angle > 0:
            accelerator_angle -= 327.5 * 2
        throttle = -accelerator_angle / 655
        if throttle < 0.0:
            throttle = 0.0
        elif throttle > 1.0:
            throttle = 1.0

        if np.abs(brake) != 0.0:
            throttle = 0.0

        return steering_angle, throttle*2.60, brake
    return None

def main():
    parser = argparse.ArgumentParser(description="HID control & data logging script")
    parser.add_argument("--csv", type=str,
                        help="Path to output CSV file (only required for drive mode)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Socket host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=65432,
                        help="Socket port (default: 65432)")
    parser.add_argument("--mode", type=str, choices=["debug", "drive", "training"], required=True,
                        help="Mode: 'debug' prints HID data to terminal; 'drive' runs TCP connection and logs CSV; 'training' runs TCP connection without logging")
    args = parser.parse_args()

    if args.mode == "debug":
        # Debug mode: No TCP connection or CSV writing; just print HID device values.
        try:
            device = hid.Device(VENDOR_ID, PRODUCT_ID)
            print(f"Connected to {device.manufacturer} - {device.product}")
            calibration_offsets = calibrate_angle(device)
            print("Calibration offsets:", calibration_offsets)
            while True:
                result = process_hid_input(device, calibration_offsets)
                if result:
                    steering_angle, throttle, brake = result
                    print(f"Steering: {steering_angle:.3f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}")
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in debug mode: {e}")
        finally:
            try:
                device.close()
            except:
                pass
            print("Device closed. Exiting debug mode.")

    elif args.mode == "drive":
        # Drive mode: Create a TCP connection and log CSV.
        if not args.csv:
            print("CSV output path required for drive mode (--csv ...)")
            return

        # Initialize server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((args.host, args.port))
        server_socket.listen(5)
        print(f"Server listening on {args.host}:{args.port}")

        # Initialize HID device
        device = hid.Device(VENDOR_ID, PRODUCT_ID)
        print(f"Connected to {device.manufacturer} - {device.product}")

        # Calibrate initial angles
        calibration_offsets = calibrate_angle(device)

        # Accept client connection
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        # Build CSV header
        header = [
            "time",
            "x_pos", "z_pos", "yaw_angle", "long_vel", "lat_vel", "yaw_rate",
            "steering", "throttle", "brake"
        ]
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        try:
            while True:
                # Process HID device input
                hid_result = process_hid_input(device, calibration_offsets)
                if hid_result:
                    steering_angle, throttle, brake = hid_result
                # Prepare and send control inputs to the client
                message = f"{steering_angle},{throttle},{brake}\n"
                client_socket.sendall(message.encode())

                # Receive vehicle state data from the client
                raw_data = client_socket.recv(1024).decode('utf-8').strip()
                try:
                    fields = raw_data.split(',')
                    if len(fields) < 6:
                        raise ValueError(f"Incomplete data received: {fields}")
                    # Extract vehicle state values
                    x_pos = float(fields[0])
                    z_pos = float(fields[1])
                    yaw = -float(fields[2]) + 90
                    long_vel = float(fields[3])
                    lat_vel = float(fields[4])
                    yaw_rate = float(fields[5])
                    row = [time.time(), x_pos, z_pos, yaw, long_vel, lat_vel, yaw_rate,
                           steering_angle, throttle, brake]
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
            
    elif args.mode == "training":
        # Training mode: Create a TCP connection but don't log to CSV.
        # Initialize server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((args.host, args.port))
        server_socket.listen(5)
        print(f"Training mode: Server listening on {args.host}:{args.port}")

        # Initialize HID device
        try:
            device = hid.Device(VENDOR_ID, PRODUCT_ID)
            print(f"Connected to {device.manufacturer} - {device.product}")

            # Calibrate initial angles
            calibration_offsets = calibrate_angle(device)
            print("Calibration offsets:", calibration_offsets)

            # Accept client connection
            print("Waiting for Unity client connection...")
            client_socket, addr = server_socket.accept()
            print(f"Connection established with {addr}")

            try:
                print("Training session started. Press Ctrl+C to stop.")
                while True:
                    # Process HID device input
                    hid_result = process_hid_input(device, calibration_offsets)
                    if hid_result:
                        steering_angle, throttle, brake = hid_result
                        # Print current control values in real-time
                        print(f"\rSteering: {steering_angle:.3f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}", end="")
                        
                        # Prepare and send control inputs to the client
                        message = f"{steering_angle},{throttle},{brake}\n"
                        client_socket.sendall(message.encode())

                        # Receive vehicle state data from the client but don't log it
                        try:
                            raw_data = client_socket.recv(1024).decode('utf-8').strip()
                            fields = raw_data.split(',')
                            if len(fields) < 6:
                                print(f"\nIncomplete data received: {fields}")
                                continue
                                
                            # We receive the data but don't write it to a CSV file
                            # Only used for maintaining the connection
                        except ValueError as ve:
                            print(f"\nData parsing error: {ve}. Raw data: '{raw_data}'")
                        except Exception as e:
                            print(f"\nUnexpected error: {e}")

                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("\nTraining session stopped by user.")
            except Exception as e:
                print(f"\nError during training: {e}")
        except Exception as e:
            print(f"Error initializing training mode: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            try:
                device.close()
            except:
                pass
            try:
                server_socket.close()
            except:
                pass
            print("Training session ended. All connections closed.")

if __name__ == "__main__":
    main()
