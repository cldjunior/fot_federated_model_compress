import os
import json
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Caminho do JSON
json_path = './sim/data_hosts.json'  # ajuste se estiver em outro local

# Carregar JSON e filtrar devicees
with open(json_path, 'r') as f:
    lines = f.readlines()
    devices = [json.loads(line) for line in lines if json.loads(line)["type"] == "device"]

# Criar pasta base
base_path = "./models"
os.makedirs(base_path, exist_ok=True)

# Criar subpastas para cada device
device_names = [s["name_iot"] for s in devices]
for name in device_names:
    device_path = os.path.join(base_path, name)
    os.makedirs(device_path, exist_ok=True)

# Carregar MNIST
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)

# Dividir em N partes iguais
n = len(device_names)
x_splits = np.array_split(x_train, n)
y_splits = np.array_split(y_train, n)

# Salvar para cada device
for i, name in enumerate(device_names):
    device_path = os.path.join(base_path, name)
    np.save(os.path.join(device_path, "x_train.npy"), x_splits[i])
    np.save(os.path.join(device_path, "y_train.npy"), y_splits[i])

print(f"✅ MNIST dividido entre {n} devicees e salvo em '{base_path}/scXX'")

