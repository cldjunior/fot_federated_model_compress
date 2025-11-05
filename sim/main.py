import paho.mqtt.client as mqtt
import json
import tatu
import argparse
from time import sleep
import traceback

parser = argparse.ArgumentParser(description='Params sensors')
parser.add_argument('--name', action='store', dest='name', required=True)
parser.add_argument('--broker', action='store', dest='broker', required=True)
args = parser.parse_args()

def on_connect(mqttc, obj, flags, rc):
    topic = obj["topicPrefix"] + obj["deviceName"]
    print(f"[INFO] Connected with result code {rc}")
    print(f"[INFO] Subscribing to topic: {topic}")
    mqttc.subscribe(topic)
    print("Device's sensors:")
    for sensor in obj['sensors']:
        print(f"\t{sensor['name']}")
    print(f"[INFO] Device {obj['deviceName']} subscribed successfully.")

def on_message(mqttc, obj, msg):
    print(f"[INFO] Message received on topic {msg.topic}")
    if obj["topicPrefix"] in msg.topic:
        tatu.main(obj, msg)

def on_disconnect(mqttc, obj, rc):
    print(f"[ERROR] Disconnected from broker. Return code: {rc}")
    exit()

while True:
    try:
        with open('config.json') as f:
            data = json.load(f)

        data["deviceName"] = args.name
        data["mqttBroker"] = args.broker
        mqttBroker = args.broker
        mqttPort = int(data.get("mqttPort", 1883))
        mqttUsername = data.get("mqttUsername", "")
        mqttPassword = data.get("mqttPassword", "")
        deviceName = data["deviceName"]

        print(f"[INFO] Attempting connection to broker {mqttBroker}:{mqttPort} as {deviceName}")

        sub_client = mqtt.Client(deviceName + "_sub", protocol=mqtt.MQTTv31)
        sub_client.username_pw_set(mqttUsername, mqttPassword)
        sub_client.user_data_set(data)
        sub_client.on_connect = on_connect
        sub_client.on_message = on_message
        sub_client.on_disconnect = on_disconnect

        sub_client.connect(mqttBroker, mqttPort, 1800)
        sub_client.loop_forever()
    except Exception as e:
        print(f"[ERROR] Broker unreachable on {mqttBroker}:{mqttPort}. Exception: {e}")
        traceback.print_exc()
        sleep(5)

