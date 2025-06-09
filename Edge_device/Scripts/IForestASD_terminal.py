import serial
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)  # changes CWD to the root

# Add the root to the sys.path for imports
sys.path.append(project_root)

from testjson import generate_schedule

print("Current Working Directory:", os.getcwd())  # Verify it's correct


import time
import pandas as pd
import numpy as np
from anomalyinserter import  process_data_iso_forest
from ML import Iso_Forest
import json
import threading
import csv
import asyncio
import joblib
import paho.mqtt.client as mqtt



FEATURE_COLUMNS = ["Temperature (°C)", "Humidity (%)", "Raw VOC", "IR Light", "Visible Light", "CO2 (ppm)",'hour','minute','second','sin_h','cos_h','sin_m','cos_m','sin_s','cos_s']
SERIAL_PORT = "COM3"   # Change to "/dev/ttyACM0" on Linux or another COM depending on the port
BAUD_RATE = 115200 
CSV_FILE = 'iso_test.csv'
RESULT_FILE = 'iso_result.csv'  # Path to the scaler file
IP_ADDRESS = "127.0.0.1"  # Replace with your MQTT broker IP address

#start the connection to the MQTT broker
def on_connect(client, userdata, flags, rc):
    print("CONNECTION ESTABLISHED")
    client.subscribe("thesis/anomalies/1")

    
async def main():
    print("Starting the Isolation forest detection")
    try:
        df = pd.DataFrame(columns=['Timestamp', 'Temperature (°C)', 'Humidity (%)', 
                                             'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)'])
        
        model = Iso_Forest()

        schedule = generate_schedule()

        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, open(CSV_FILE, mode="a", newline="") as file, open(RESULT_FILE, mode="a", newline="") as result_file:
            print(f"Listening on {SERIAL_PORT}... (Press Ctrl+C to stop)")
            writer = csv.writer(file, delimiter=";")
            result_writer = csv.writer(result_file, delimiter=";")



            client = mqtt.Client()
            client.on_connect = on_connect

            client.connect(IP_ADDRESS, 1883, 60)
            client.loop_start()
            if file.tell() == 0:
                writer.writerow(['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)', 'Is Anomaly', 'Anomaly (Prediction)'])
                
            if result_file.tell() == 0:  
                result_writer.writerow(['Anomaly Score', 'Is Anomaly', 'Anomaly (Prediciton)', 'Sample_score'])
            while(True):
                line = ser.readline().decode("utf-8").strip()  
                print(f"Received: {line}")
                if line:
                    is_anomaly, payload = await process_data_iso_forest(line, model, writer, df, result_writer, schedule)
                    if is_anomaly == True:  # If an anomaly is detected
                        client.publish("thesis/anomalies/1", json.dumps(payload))
                        print(f"Published anomaly data: {payload}")
                    file.flush()
                    result_file.flush()
                await asyncio.sleep(0.1)  # Small delay to minimize CPU usage

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nLogging stopped.")
    finally:
        try:
            file.close()
            result_file.close()
            ser.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())