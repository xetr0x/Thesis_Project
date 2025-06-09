import serial
import csv
import time

SERIAL_PORT = "COM4"  # Change to "/dev/ttyUSB0" on Linux/macOS
BAUD_RATE = 115200  # 
CSV_FILE = "pico_data.csv"

def main():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, open(CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter=";")


            if file.tell() == 0:
                writer.writerow(["Timestamp", "Temperature (°C)", "Humidity (%)", "Raw VOC", "IR Light", "Visible Light", "CO2 (ppm)"]) 

            print(f"Listening on {SERIAL_PORT}... (Press Ctrl+C to stop)")

            while True:
                line = ser.readline().decode("utf-8").strip()  
                print(f"Received: {line}")  

                if line:
                    data = line.split(",")  

                    if len(data) == 6:  
                        temperature, humidity, raw_voc, ir_light, visible_light, co2 = data
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
                        row = [timestamp, temperature, humidity, raw_voc, ir_light, visible_light, co2]
                        writer.writerow(row)  
                        file.flush()  
                        print(f"Logged: {row}")  

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nLogging stopped.")

if __name__ == "__main__":
    main()
