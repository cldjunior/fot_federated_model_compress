import argparse
import os
import time
import numpy as np
import tensorflow as tf
import flwr as fl
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from flwr.common import NDArrays
from csv import writer
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=str, required=True)
parser.add_argument("--quant_type", type=str, default="float32", choices=["float32", "float16", "int8", "progressive", "heterogeneous"])
args = parser.parse_args()

DEVICE = args.device_id
QUANT_TYPE = args.quant_type.lower()

# Load and preprocess data
x = np.load(f"models/{DEVICE}/x_train.npy")
y = np.load(f"models/{DEVICE}/y_train.npy")
x = x.astype("float32") / 255.0
x = x.reshape((-1, 28, 28, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Quantization functions
def quantize_weights(weights, qtype):
    if qtype == "float16":
        return [w.astype(np.float16) for w in weights]
    elif qtype == "int8":
        return [((w / np.max(np.abs(w))) * 127).astype(np.int8) if w.dtype in [np.float32, np.float64] else w for w in weights]
    return weights

def get_quant_type_from_round(round_num):
    if round_num < 2:
        return "float32"
    elif round_num < 4:
        return "float16"
    return "int8"

QUANT_TYPE_MAP = {
    "dc01_compress": "float32", "dc02_compress": "float16", "dc03_compress": "int8",
    "dc04_compress": "float16", "dc05_compress": "int8"
}

# Create model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Save metrics with system info
def save_metrics(device_id, rnd, loss_b, acc_b, loss_a, acc_a, strategy, qtype, train_time, model_size, cpu, ram, ram_free):
    path = f"models/{device_id}/metrics_{device_id}.csv"
    columns = ["strategy", "round", "loss_before", "acc_before", "loss_after", "acc_after","quant", "train_time", "model_size_bytes", "cpu_percent", "ram_percent", "ram_available"]
    row = [strategy, rnd, loss_b, acc_b, loss_a, acc_a, qtype, train_time, model_size, cpu, ram, ram_free]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = writer(f)
        if write_header: w.writerow(columns)
        w.writerow(row)

# Plot accuracy and loss
def plot_metrics(device_id):
    csv_path = f"models/{device_id}/metrics_{device_id}.csv"
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    rounds = df["round"]
    plt.figure()
    plt.plot(rounds, df["loss_before"], label="Loss Before")
    plt.plot(rounds, df["loss_after"], label="Loss After")
    plt.title(f"Loss - {device_id}"); plt.legend(); plt.xlabel("Round"); plt.ylabel("Loss")
    plt.savefig(f"models/{device_id}/loss_{device_id}.png"); plt.close()

    plt.figure()
    plt.plot(rounds, df["acc_before"], label="Acc Before")
    plt.plot(rounds, df["acc_after"], label="Acc After")
    plt.title(f"Accuracy - {device_id}"); plt.legend(); plt.xlabel("Round"); plt.ylabel("Accuracy")
    plt.savefig(f"models/{device_id}/acc_{device_id}.png"); plt.close()

# Federated client
class FederatedClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = create_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters: NDArrays, config):
        
        self.model.set_weights(parameters)
        round_num = config.get("epoch_global", 0)
        strategy = config.get("strategy", "unknown")

        if QUANT_TYPE == "progressive":
            current_q = get_quant_type_from_round(round_num)
        elif QUANT_TYPE == "heterogeneous":
            current_q = QUANT_TYPE_MAP.get(DEVICE, "float32")
        else:
            current_q = QUANT_TYPE


        # Evaluate BEFORE training
        loss_b, acc_b = self.model.evaluate(x_test, y_test, verbose=0)

        # Local training
        start = time.time()
        self.model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
        train_time = time.time() - start

        # Evaluate AFTER training
        loss_a, acc_a = self.model.evaluate(x_test, y_test, verbose=0)

        # Quantize weights
        weights = quantize_weights(self.model.get_weights(), current_q)

        # System metrics
        model_size = sum([w.nbytes for w in self.model.get_weights()])
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        ram_free = psutil.virtual_memory().available

        # Save metrics
        os.makedirs(f"models/{DEVICE}", exist_ok=True)
        save_metrics(DEVICE, round_num, loss_b, acc_b, loss_a, acc_a, strategy, current_q, train_time, model_size, cpu, ram, ram_free)
        print(f"✅ Round {round_num}: métricas salvas com sucesso.")
        return weights, len(x_train), {"loss": loss_a, "accuracy": acc_a, "quant": current_q}

    def evaluate(self, parameters: NDArrays, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": acc}

if __name__ == "__main__":
    fl.client.start_client(server_address="172.27.27.4:8080", client=FederatedClient().to_client())
    plot_metrics(DEVICE)



