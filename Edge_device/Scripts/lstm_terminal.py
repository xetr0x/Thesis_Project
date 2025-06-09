import serial
import pandas as pd
import numpy as np
import csv
import torch as torch
from torch import nn as nn
from sklearn.preprocessing import StandardScaler 
import asyncio
from anomalyinserter import  LSTM,load_model, generate_anomaly_schedule, process_data_lstm,TimeSeriesBuffer
import paho.mqtt.client as mqtt


SERIAL_PORT = "/dev/ttyACM0"  # Change to "/dev/ttyACM0" on Linux or another COM depending on the port
BAUD_RATE = 115200  
CSV_FILE = '../Resources/Thesis/Thesis_lstm.csv'
RESULT_FILE = '../Resources/Thesis/Thesis_lstm_anomaly.csv'

FEATURE_COLUMNS = ["Temperature (°C)", "Humidity (%)", "Raw VOC", "IR Light", "Visible Light", "CO2 (ppm)",'hour','minute','second','sin_h','cos_h','sin_m','cos_m','sin_s','cos_s']

def on_connect(client, userdata, flags, rc):
    print("CONNECTION ESTABLISHED")
    client.subscribe("thesis/anomalies/2")

async def main():
    try:
        df = pd.DataFrame(columns=['Timestamp', 'Temperature (°C)', 'Humidity (%)', 
                                             'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)'])
        model, scaler = load_model(i = 1)
        ts_buffer = TimeSeriesBuffer(seq_length=20)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        schedule = generate_anomaly_schedule(count = 122,period_hours = 5)
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, open(CSV_FILE, mode="a", newline="") as file, open(RESULT_FILE, mode="a", newline="") as result_file:
            print(f"Listening on {SERIAL_PORT}... (Press Ctrl+C to stop)")
            writer = csv.writer(file, delimiter=";")
            result_writer = csv.writer(result_file, delimiter=";")
            client = mqtt.Client()
            client.on_connect = on_connect

            client.connect("IP ADRESS", 1883, 60)
            client.loop_start()

            if file.tell() == 0:
                writer.writerow(['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)', 'Is Anomaly'])
                
            if result_file.tell() == 0:  
                result_writer.writerow(['Anomaly Score', 'Is Anomaly'])

            while(True):
                line = ser.readline().decode("utf-8").strip()  
                if line:
                    await process_data_lstm(line, model, scaler, writer,df, device, result_writer, schedule, ts_buffer, client)
                    result_file.flush()
                    file.flush()
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
            client.loop_stop()
            client.disconnect()
        except:
            pass
if __name__ == "__main__":
    asyncio.run(main())
