import json
import os
import paho.mqtt.client as mqtt
import sys
import time
import logging

broker_address = sys.argv[1]
server_ip = sys.argv[2]
broker_port = 1883
topic = '#'
time_send = 30

output_dir = '/data'

# Carrega o mapeamento de sensor → device
sensor_to_device = {}
with open('/fot_gateway/association_hosts.json') as f:
    for line in f:
        assoc = json.loads(line)
        sensor_to_device[assoc['name']] = assoc['nome_device']

def on_connect(client, userdata, flags, rc):
    print("[MQTT] Conectado ao broker")
    client.subscribe(topic)

def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode())
        sensor = payload["header"]["sensor"]
        sensor_id = payload["header"]["device"]
        data = payload["data"][0]
        datetime_pub = payload["datetime_pub"]

        device = sensor_to_device.get(sensor_id)
        if not device:
            logging.warning(f"Sensor {sensor_id} não mapeado para nenhum device")
            return

        record = {
            "datetime_pub": datetime_pub,
            "sensor": sensor,
            "sensor_id": sensor_id,
            "data": data
        }

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{device}.json")
        with open(file_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        logging.error(f"Erro ao processar mensagem: {e}")

def run():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_address, broker_port)

    while True:
        try:
            client.loop_start()
            time.sleep(time_send)
            client.loop_stop()
            time.sleep(5)
        except Exception as err:
            logging.error(f"[✘] Erro no loop: {err}")

if __name__ == '__main__':
    logging.basicConfig(filename='./python.log', level=logging.INFO)
    logging.info("Server: " + server_ip)
    logging.info("Broker: " + broker_address)
    run()


