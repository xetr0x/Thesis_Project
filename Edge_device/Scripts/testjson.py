import json
from anomalyinserter import create_anomaly, generate_anomaly_schedule
import pandas as pd

def save_schedule(schedule, filename = 'anomaly_schedule.json'):
    try:
        ser_schedule = []
        for anomaly in schedule:
            ser_anom = anomaly.copy()
            ser_anom['timestamp'] = anomaly['timestamp'].isoformat()
            ser_schedule.append(ser_anom)

        with open(filename, 'w') as f:
            json.dump(ser_schedule, f, indent=4)
    except Exception as e:
        print(f"Error: {e}")

def main():

    schedule = generate_anomaly_schedule(500, 24)
    save_schedule(schedule, filename = 'anomaly_schedule.json')


def generate_schedule():
    schedule = generate_anomaly_schedule(200, 24)
    save_schedule(schedule, filename = 'anomaly_schedule.json')
    return schedule

if __name__ == "__main__":
	main()
