import os
import json
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from collections import Counter

# Caminho do JSON
json_path = './sim/data_hosts.json'

# Carregar JSON e filtrar devices
with open(json_path, 'r') as f:
    lines = f.readlines()
    devices = [json.loads(line) for line in lines if json.loads(line)["type"] == "device"]

device_names = [d["name_iot"] for d in devices]
n = len(device_names)

# Criar pastas
base_path = "./models"
os.makedirs(base_path, exist_ok=True)
for name in device_names:
    os.makedirs(os.path.join(base_path, name), exist_ok=True)

# Carregar MNIST
(x_all, y_all), _ = mnist.load_data()
x_all = x_all.astype("float32") / 255.0
x_all = x_all.reshape((-1, 28, 28, 1))

# Funções de partição Non-IID
def partition_dc01():
    idx = np.random.choice(len(y_all), 5000, replace=False)
    return x_all[idx], y_all[idx]

def partition_dc02():
    idx = np.where((y_all == 0) | (y_all == 1))[0]
    idx = np.random.choice(idx, 5000, replace=False)
    return x_all[idx], y_all[idx]

def partition_dc03():
    idx = np.where(y_all == 7)[0]
    idx = np.random.choice(idx, 5000, replace=False)
    return x_all[idx], y_all[idx]

def partition_dc04():
    counts = [int(5000 / (i + 1)) for i in range(10)]
    x_zipf, y_zipf = [], []
    for c, n in enumerate(counts):
        idx = np.where(y_all == c)[0]
        sel = np.random.choice(idx, min(n, len(idx)), replace=False)
        x_zipf.append(x_all[sel])
        y_zipf.append(y_all[sel])
    return np.concatenate(x_zipf), np.concatenate(y_zipf)

def partition_dc05():
    idx_0 = np.where(y_all == 0)[0]
    idx_other = np.where(y_all != 0)[0]
    idx_0 = np.random.choice(idx_0, 4500, replace=False)
    idx_other = np.random.choice(idx_other, 500, replace=False)
    idx = np.concatenate([idx_0, idx_other])
    return x_all[idx], y_all[idx]

# Aplicar partições para os 5 primeiros
partitions = [partition_dc01, partition_dc02, partition_dc03, partition_dc04, partition_dc05]

for i, name in enumerate(device_names):
    if i < 5:
        x_part, y_part = partitions[i]()
    else:
        # IID simples para os demais
        split_size = 5000
        idx = np.random.choice(len(y_all), split_size, replace=False)
        x_part, y_part = x_all[idx], y_all[idx]

    np.save(f"{base_path}/{name}/x_train.npy", x_part)
    np.save(f"{base_path}/{name}/y_train.npy", y_part)
    print(f"✅ {name}: {Counter(y_part)}")

print(f"\n✅ Dataset particionado com Non-IID nos 5 primeiros devices. Total: {n} devices.")


