import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import msvcrt
import time
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)  # changes CWD to the root
sys.path.append(project_root)

def plot_graph(x_train, title, x_axis_label, y_axis_label, x_var, y_var):
    """graph for checking data"""
    window = plt.figure(figsize=(10, 5))
    window.suptitle(title)

    plot = window.add_subplot()
    regular = x_train[x_train['anomaly'] == 1]
    plot.scatter(regular[x_var], regular[y_var], label='Regular/non-anomalous')

    anomaly = x_train[x_train['anomaly'] == -1]
    plot.scatter(anomaly[x_var], anomaly[y_var], label='non-regular/anomaly')

    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)


def collect_data(start_index, end_index):

    n_rows = end_index - start_index
    nrows = int(n_rows)

    columns = ['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)']
    collected_dataframe = pd.read_csv(
            'Resources/Our_Dataset/data.csv',
            delimiter=';', skiprows=start_index,
            nrows=nrows, names=columns, header=0)  # ; seperates the columns
    print(collected_dataframe)
    return collected_dataframe

def collect_test_data():


    collected_dataframe = pd.read_csv(
            'iso_test.csv',
            delimiter=';',  header=0)  # ; seperates the columns
    print("expected columns? ", collected_dataframe.columns)
    columns = ['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)', 'Is Anomaly']
    collected_dataframe = collected_dataframe[columns]
    print(type(collected_dataframe))
    return collected_dataframe

# def data_size():
#     collected_dataframe = pd.read_csv(
#             'ML\data_fixed.csv',
#             delimiter=';')  # ; seperates the columns
    


def data_cleaner(dataframe_toclean, features):
    print("columns ", dataframe_toclean.columns)
    cleaned_dataframe = dataframe_toclean[features + ['Timestamp']].copy()

# # Changes Date and Time into Datetime column of date time type
#     cleaned_dataframe['Datetime'] = pd.to_datetime(
#         cleaned_dataframe['Date'] + ' ' +
#         cleaned_dataframe['Time'], format="%d/%m/%Y %H.%M.%S")
    cleaned_dataframe = cleaned_dataframe.replace('#NAMN?', -200)
# makes them all into valid values
    cleaned_dataframe[features] = cleaned_dataframe[features].replace(
        ',', '.', regex=True).astype(float)
    
    print("in function type: ", type(cleaned_dataframe))

    # cleaned_dataframe = cleaned_dataframe.dropna()

    return cleaned_dataframe


def split_timestamp(df):
    collected_dataframe = pd.read_csv(
            'ML/data_fixed.csv',
            delimiter=';')
  

# df.to_csv to writing



# make a delimeter with both ";" and " "



def check_for_key_press(): #change this when running on linux or mac since this is a microsoft specific library
    
    global inject_anomaly
    global running
    
    print("Key press detection active. Press 'a' to inject an anomaly, 'q' to quit.")
    
    while running:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            
            if key == 'a':
                inject_anomaly = True
            elif key == 'q':
                print("\nQuitting application...")
                running = False
                break
                
        time.sleep(0.1)
