from datetime import datetime
import json
import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribe to the same topic we'll publish to
    client.subscribe("test/topic")


def on_message(msg):
    print(f"Received message: {msg.payload.decode()} on topic {msg.topic}")


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message


client.connect("IP_Adress", 1883, 60)


payload = {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "temperature": 23,
    "humidity": 40,
    "voc": 15,
    "ir_light": 100,
    "visible_light": 200,
    "co2": 200
}

json_payload = json.dumps(payload)

client.loop_start()


for i in range(1):
    client.publish("thesis/anomalies/2", json_payload)
    print(f"Published: {json_payload}")
    time.sleep(5)