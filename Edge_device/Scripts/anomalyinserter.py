import pandas as pd
import numpy as np
import random
import torch as torch
from torch import nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime, timedelta, time
import paho.mqtt.client as mqtt

class autoencoder(nn.Module):
    def __init__(self, input_shape, encoding_dim):
        super(autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, encoding_dim),
            )

        self.decode = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, input_shape)
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch = len(x.size()) == 3
        if not batch:

            x = x.unsqueeze(0)
        batch_size = x.size(0)


        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0)) 
        
        out = self.fc(out[:, -1, :])
        return out

def create_anomaly(df, sensors=None, severity='medium'):
   
    anomalous_data = df.copy()
    
    all_sensors = ['Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)']
    
    if severity == 'low':
        num_sensors_range = (1, 2)
        magnitude_multiplier = 1.0
    elif severity == 'medium':
        num_sensors_range = (1, 3)
        magnitude_multiplier = 1.5
    elif severity == 'high':
        num_sensors_range = (2, 4)
        magnitude_multiplier = 2.0
    elif severity == 'critical':  
        num_sensors_range = (3, 6)
        magnitude_multiplier = 3.0
    else:
        num_sensors_range = (1, 2)
        magnitude_multiplier = 0.75
    if sensors is None:
        num_sensors_to_affect = random.randint(num_sensors_range[0], num_sensors_range[1])
        sensors_to_modify = random.sample(all_sensors, min(num_sensors_to_affect, len(all_sensors)))
    else:
        sensors_to_modify = [s for s in sensors if s in all_sensors]
    anomaly_details = {}
    
    for sensor in sensors_to_modify:
        original_value = anomalous_data.loc[0, sensor]
        
        if sensor == 'Temperature (°C)':
            change = random.choice([-1, 1]) * random.uniform(5, 15) * magnitude_multiplier
            new_value = original_value + change
            
        elif sensor == 'Humidity (%)':
            change = random.choice([-1, 1]) * random.uniform(15, 30) * magnitude_multiplier
            new_value = max(0, min(100, original_value + change))  
            
        elif sensor == 'Raw VOC':
            if random.random() > 0.5:
                
                new_value = original_value * random.uniform(1.5, 3) * magnitude_multiplier
            else:
                
                new_value = original_value * random.uniform(0.3, 0.7) / magnitude_multiplier
                
        elif sensor == 'IR Light' or sensor == 'Visible Light':
            if random.random() > 0.5:
                
                new_value = original_value * random.uniform(2, 4) * magnitude_multiplier
            else:
                new_value = original_value * random.uniform(0.2, 0.4) / magnitude_multiplier
                
        elif sensor == 'CO2 (ppm)':
            if random.random() > 0.7:
                new_value = original_value * random.uniform(0.4, 0.7) / magnitude_multiplier
            else:
                new_value = original_value * random.uniform(1.5, 2.5) * magnitude_multiplier
        
        anomalous_data.loc[0, sensor] = new_value
        
    return anomalous_data

def transform_all(df,device, scaler):
    df_new = df.copy()

    df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])

    df_new['hour'] = df_new['Timestamp'].dt.hour
    df_new['minute'] = df_new['Timestamp'].dt.minute
    df_new['second'] = df_new['Timestamp'].dt.second

    df_new['sin_h'] = np.sin(2* np.pi * df_new['hour']/24)
    df_new['cos_h'] = np.cos(2* np.pi * df_new['hour']/24)
    df_new['sin_m'] = np.sin(2* np.pi * df_new['minute']/60)
    df_new['cos_m'] = np.cos(2* np.pi * df_new['minute']/60)
    df_new['sin_s'] = np.sin(2* np.pi * df_new['second']/60)
    df_new['cos_s'] = np.cos(2* np.pi * df_new['second']/60)
    timestamps = df_new['Timestamp'].values
    
    numeric_df = df_new.copy()
    error_values = ['#NAMN', '#NAMN?', '#NAME?', '#DIV/0!', '#N/A', '#NULL!', '#NUM!', '#REF!', '#VALUE!','-inf','inf']
    
    for col in numeric_df.columns:
        if numeric_df[col].dtype == 'object' or numeric_df[col].astype(str).str.contains('#').any():
            numeric_df[col] = numeric_df[col].astype(str).replace(error_values, '1e9')
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    numeric_df = numeric_df.replace(np.inf, 1e9)
    numeric_df = numeric_df.replace(-np.inf, -1e9)
    numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))
    numeric_df = numeric_df.drop(['Timestamp', 'hour', 'minute', 'second'], axis=1, errors='ignore')

    data_scaled = scaler.transform(numeric_df)
    
    x_tensor = torch.FloatTensor(data_scaled).to(device)
    return x_tensor
    """Loads the trained models, simple duh!"""
def load_model(i = 1):
    if i == 1:
        MODEL_FILE = '../Resources/models_scalers/LSTM.pth'
        SCALER_FILE = '../Resources/models_scalers/LSTMscaler.sav'
        model = LSTM(12, 64, 2, 12)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        model.eval()
        scaler = joblib.load(SCALER_FILE)
    if i == 2:
        MODEL_FILE = '../Resources/models_scalers/autoe_model.pth'
        SCALER_FILE = '../Resources/models_scalers/autoencoderscaler.sav'
        model = autoencoder(12, 10)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        model.eval()
        scaler = joblib.load(SCALER_FILE)
    return model, scaler
    
    
def generate_anomaly_schedule(count, period_hours):
    now = datetime.now()
    end_time = now + timedelta(hours=period_hours)
    
    anomaly_times = []
    for i in range(count):
        seconds_offset = random.randint(0, period_hours * 3600)
        timestamp = now + timedelta(seconds=seconds_offset)
        if timestamp > end_time:
            timestamp = end_time
        
        severity = random.choices(
            ['low', 'medium', 'high', 'extreme'],
            weights=[0.3, 0.4, 0.2, 0.1],
            k=1
        )[0]
        
        if random.random() < 0.5: 
            anomaly_type = 'generic'
        else:
            specific_types = ['temperature_spike', 'humidity_drop', 
                             'air_quality_crisis', 'sensor_failure', 'light_anomaly']
            anomaly_type = random.choice(specific_types)
        
        anomaly_times.append({
            'timestamp': timestamp,
            'severity': severity,
            'type': anomaly_type,
            'executed': False
        })
    return sorted(anomaly_times, key=lambda x: x['timestamp'])

"""This to create a context window for the LSTM"""
class TimeSeriesBuffer:
    def __init__(self, seq_length=20):
        from collections import deque
        self.buffer = deque(maxlen=seq_length)
        self.seq_length = seq_length
        
    def add_data_point(self, data_point):
        """Add processed data point to buffer"""
        self.buffer.append(data_point)
        
    def is_full(self):
        """Check if buffer has enough data points for prediction"""
        return len(self.buffer) == self.seq_length
        
    def get_sequence(self):
        """Return current sequence as numpy array"""
        if not self.is_full():
            return None
        return np.array(list(self.buffer))
"""This fixes The time series in a manner that the LSTM can handle also it is different from 
    the other function since it doesnt scale the data"""
def fix_timeseries(df): #different from the other method since it doesn't scale data, this is specifically for LSTM
    df_new = df.copy()
    
    df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])
    df_new['hour'] = df_new['Timestamp'].dt.hour
    df_new['minute'] = df_new['Timestamp'].dt.minute
    df_new['second'] = df_new['Timestamp'].dt.second
    
    df_new['sin_h'] = np.sin(2 * np.pi * df_new['hour']/24)
    df_new['cos_h'] = np.cos(2 * np.pi * df_new['hour']/24)
    df_new['sin_m'] = np.sin(2 * np.pi * df_new['minute']/60)
    df_new['cos_m'] = np.cos(2 * np.pi * df_new['minute']/60)
    df_new['sin_s'] = np.sin(2 * np.pi * df_new['second']/60)
    df_new['cos_s'] = np.cos(2 * np.pi * df_new['second']/60)
    timestamps = df_new['Timestamp'].values
    
    numeric_df = df_new.copy()
    error_values = ['#NAMN', '#NAMN?', '#NAME?', '#DIV/0!', '#N/A', '#NULL!', '#NUM!', '#REF!', '#VALUE!','inf','-inf']
    
    for col in numeric_df.columns:
        if numeric_df[col].dtype == 'object' or numeric_df[col].astype(str).str.contains('#').any():
            numeric_df[col] = numeric_df[col].astype(str).replace(error_values, '1e9')
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    numeric_df = numeric_df.replace(np.inf, 1e9)
    numeric_df = numeric_df.replace(-np.inf, -1e9)
    numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))
    numeric_df = numeric_df.drop(['Timestamp', 'hour', 'minute', 'second'], axis=1, errors='ignore')
    return numeric_df


"""THis is for LSTM to process data as well as write it to a file, it works but not very elegantly"""
async def process_data_lstm(line, model, scaler, writer, df, device, result_writer,schedule, ts_buffer, client):

    print(f"Received: {line}")
    data = line.split(',')
    topic = "thesis/anomalies/1"
    if len(data) == 6:
        times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        """Capture data before any modifications""" 
        try:
            temperature = float(data[0])
            humidity = float(data[1])
            raw_voc = float(data[2])
            ir_light = float(data[3])
            visible_light = float(data[4])
            co2 = float(data[5])
        except ValueError as e:
            print(f"Error converting data: {e}, raw data: {data}")
            return
        """Add data to pandas dataframe""" 
        df.loc[0] = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2]
        
        current_time = datetime.now()
        inject_anomaly = False
        """Check if there is an anomaly due"""
        for anomaly in schedule:
            if not anomaly['executed'] and current_time >= anomaly['timestamp']:
                inject_anomaly = True
                anomaly_severity = anomaly['severity']
                anomaly_type = anomaly['type']
                anomaly['executed'] = True
                break
        """injecting anomaly if it is due""" 
        if inject_anomaly:
            if anomaly_type == 'generic':
                df = create_anomaly(df, severity=anomaly_severity)
            elif anomaly_type == 'temperature_spike':
                df = create_anomaly(df, sensors=['Temperature (°C)'], severity=anomaly_severity)
            elif anomaly_type == 'humidity_drop':
                df = create_anomaly(df, sensors=['Humidity (%)'], severity=anomaly_severity)
            elif anomaly_type == 'air_quality_crisis':
                df = create_anomaly(df, sensors=['Raw VOC', 'CO2 (ppm)'], severity=anomaly_severity)
            elif anomaly_type == 'sensor_failure':
                sensor = random.choice(['Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)'])
                df = create_anomaly(df, sensors=[sensor], severity=anomaly_severity)
            elif anomaly_type == 'light_anomaly':
                df = create_anomaly(df, sensors=['IR Light', 'Visible Light'], severity=anomaly_severity)
        
        """Grabs the data post modifications to write it to a file""" 
        temperature = df.loc[0, 'Temperature (°C)']
        humidity = df.loc[0, 'Humidity (%)']
        raw_voc = df.loc[0, 'Raw VOC']
        ir_light = df.loc[0, 'IR Light']
        visible_light = df.loc[0, 'Visible Light']
        co2 = df.loc[0, 'CO2 (ppm)']
        
        """Fixes the data, ie error handling and transforming timeseries to sin and cosine waves"""
        numeric_df = fix_timeseries(df)
         
        
        scaled_data = scaler.transform(numeric_df)
        """Adds datapoint to context window, and checks if the window is full"""
        ts_buffer.add_data_point(scaled_data[0])
        if ts_buffer.is_full():
                sequence = ts_buffer.get_sequence()
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(sequence_tensor)
                
                # Calculate reconstruction error (last point vs predicted)
                last_input = torch.FloatTensor(sequence[-1]).to(device)
                prediction = output[0]

                reconstruction_error = torch.mean((prediction - last_input) ** 2)
                
                anomaly_score = reconstruction_error.item()
                print(f"Anomaly score: {anomaly_score}")
                if anomaly_score > 0.26:
                    anomal = True
                    
                    payload = {"time": times, "temperature": temperature, "humidity":humidity, "voc":raw_voc, "ir_light":ir_light, "visible_light":visible_light, "co2":co2, "anomal":anomal}
                    data_out = json.dumps(payload)
                    client.publish(topic, data_out)
                else:
                    anomal = False
                row2 = [anomaly_score, anomal]
                row = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2, anomal]
                writer.writerow(row)
                result_writer.writerow(row2)
        else:
            x = torch.FloatTensor(scaled_data).to(device)
            
            with torch.no_grad():
                output = model(x)
                
            reconstruction_error = torch.mean((output - x) ** 2)
            anomaly_score = reconstruction_error.item()
            
            if anomaly_score > 1.26:

                anomal = True
                payload = {"time": times, "temperature": temperature, "humidity":humidity, "voc":raw_voc, "ir_light":ir_light, "visible_light":visible_light, "co2":co2, "anomal":anomal}
                data_out = json.dumps(payload)
                client.publish(topic, data_out)
            else:
                anomal = False
            result_writer.writerow([anomaly_score, anomal])
            print(f"Anomaly score: {anomaly_score}")
            row = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2, anomal]
            writer.writerow(row)
"""For transforming data to the autoencoder, it does almost the same as the lstm one but it does not have a context window"""
async def process_data(line, model, scaler, writer, df, device, result_writer,schedule,client):

    print(f"Received: {line}")
    data = line.split(',')
    topic = "thesis/anomalies/1" 
    if len(data) == 6:
        times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            temperature = float(data[0])
            humidity = float(data[1])
            raw_voc = float(data[2])
            ir_light = float(data[3])
            visible_light = float(data[4])
            co2 = float(data[5])
        except ValueError as e:
            print(f"Error converting data: {e}, raw data: {data}")
            return
            
        df.loc[0] = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2]
        
        current_time = datetime.now()
        inject_anomaly = False
        
        for anomaly in schedule:
            if not anomaly['executed'] and current_time >= anomaly['timestamp']:
                inject_anomaly = True
                anomaly_severity = anomaly['severity']
                anomaly_type = anomaly['type']
                anomaly['executed'] = True
                break
        
        if inject_anomaly:
            if anomaly_type == 'generic':
                df = create_anomaly(df, severity=anomaly_severity)
            elif anomaly_type == 'temperature_spike':
                df = create_anomaly(df, sensors=['Temperature (°C)'], severity=anomaly_severity)
            elif anomaly_type == 'humidity_drop':
                df = create_anomaly(df, sensors=['Humidity (%)'], severity=anomaly_severity)
            elif anomaly_type == 'air_quality_crisis':
                df = create_anomaly(df, sensors=['Raw VOC', 'CO2 (ppm)'], severity=anomaly_severity)
            elif anomaly_type == 'sensor_failure':
                sensor = random.choice(['Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)'])
                df = create_anomaly(df, sensors=[sensor], severity=anomaly_severity)
            elif anomaly_type == 'light_anomaly':
                df = create_anomaly(df, sensors=['IR Light', 'Visible Light'], severity=anomaly_severity)
            
        temperature = df.loc[0, 'Temperature (°C)']
        humidity = df.loc[0, 'Humidity (%)']
        raw_voc = df.loc[0, 'Raw VOC']
        ir_light = df.loc[0, 'IR Light']
        visible_light = df.loc[0, 'Visible Light']
        co2 = df.loc[0, 'CO2 (ppm)']
        x = transform_all(df, device, scaler)
        
        with torch.no_grad():
            output = model(x)
        reconstruction_error = torch.mean((output - x) ** 2)
        
        anomaly_score = reconstruction_error.item()
        anomaly_score = float(anomaly_score)
        if anomaly_score > 0.68:
            anomal = True
            row = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2, anomal]
            writer.writerow(row)
            print(f"Anomaly score: {anomaly_score}")
            payload = {"time": times, "temperature": temperature, "humidity":humidity, "voc":raw_voc, "ir_light":ir_light, "visible_light":visible_light, "co2":co2, "anomal": anomal}
            data_out = json.dumps(payload)
            client.publish(topic, data_out)
            row2 = [anomaly_score, anomal]
            result_writer.writerow(row2)
        else:
            anomal = False
            row = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2, anomal]
            writer.writerow(row)
            print(f"Anomaly score: {anomaly_score}")
            payload = {"time": times, "temperature": temperature, "humidity":humidity, "voc":raw_voc, "ir_light":ir_light, "visible_light":visible_light, "co2":co2, "anomal": anomal}
            data_out = json.dumps(payload)
            client.publish(topic, data_out)
            row2 = [anomaly_score, anomal]
            result_writer.writerow(row2)


async def process_data_iso_forest(line, model, writer, df,result_writer, schedule):

    print(f"Received: {line}")
    data = line.split(',')
    
    if len(data) == 6:
        times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            temperature = float(data[0])
            humidity = float(data[1])
            raw_voc = float(data[2])
            ir_light = float(data[3])
            visible_light = float(data[4])
            co2 = float(data[5])
        except ValueError as e:
            print(f"Error converting data: {e}, raw data: {data}")
            return
        print("Columns:", df.columns)
        print("Expected columns:", len(df.columns))
        df.loc[0] = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2]
        
        current_time = datetime.now()
        inject_anomaly = False
        
        for anomaly in schedule:
            if not anomaly['executed'] and current_time >= anomaly['timestamp']:
                inject_anomaly = True
                anomaly_severity = anomaly['severity']
                anomaly_type = anomaly['type']
                anomaly['executed'] = True
                break
        
        if inject_anomaly:
            if anomaly_type == 'generic':
                df = create_anomaly(df, severity=anomaly_severity)
            elif anomaly_type == 'temperature_spike':
                df = create_anomaly(df, sensors=['Temperature (°C)'], severity=anomaly_severity)
            elif anomaly_type == 'humidity_drop':
                df = create_anomaly(df, sensors=['Humidity (%)'], severity=anomaly_severity)
            elif anomaly_type == 'air_quality_crisis':
                df = create_anomaly(df, sensors=['Raw VOC', 'CO2 (ppm)'], severity=anomaly_severity)
            elif anomaly_type == 'sensor_failure':
                sensor = random.choice(['Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)'])
                df = create_anomaly(df, sensors=[sensor], severity=anomaly_severity)
            elif anomaly_type == 'light_anomaly':
                df = create_anomaly(df, sensors=['IR Light', 'Visible Light'], severity=anomaly_severity)
            anomal = True
        else:
            anomal = False
        
        temperature = df.loc[0, 'Temperature (°C)']
        humidity = df.loc[0, 'Humidity (%)']
        raw_voc = df.loc[0, 'Raw VOC']
        ir_light = df.loc[0, 'IR Light']
        visible_light = df.loc[0, 'Visible Light']
        co2 = df.loc[0, 'CO2 (ppm)']

        
        anomaly_score, is_anomaly, sample_score = model.predict(df)


        anomaly_score = float(anomaly_score)
        is_anomaly = bool(is_anomaly)
        print(f"Anomaly score: {anomaly_score}")
        print("data type of pred_data['anomaly_score']: ", type(is_anomaly))
        print("data type of pred_data['anomaly_score']: ", type(anomaly_score))

        row = [times, temperature, humidity, raw_voc, ir_light, visible_light, co2, anomal, is_anomaly]
        writer.writerow(row)

        row2 = [anomaly_score, anomal, is_anomaly, sample_score]
        result_writer.writerow(row2)

        payload = {
            "time": times,
            "temperature": temperature,
            "humidity": humidity,
            "voc": raw_voc,
            "ir_light": ir_light,
            "visible_light": visible_light,
            "co2": co2,
            "anomal": is_anomaly,
        }
        return is_anomaly, payload

