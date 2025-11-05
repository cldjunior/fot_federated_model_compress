import flwr as fl
import tensorflow as tf
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


import os

# Reduz verbosidade dos logs do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Apenas erros
os.environ["OMP_NUM_THREADS"] = "1"       # Limita uso de threads


# Limita número de threads usados pelo TensorFlow (reduz uso de CPU/memória)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# Número de épocas configurável
NUM_EPOCHS = 3


# Argumento do sensor
parser = argparse.ArgumentParser()
parser.add_argument("--sensor_id", required=True, help="ID do sensor (ex: sc01, sc02...)")
args = parser.parse_args()
sensor_id = args.sensor_id

# Caminho do dataset local
data_dir = os.path.join("models", args.sensor_id)
x_train = np.load(os.path.join(data_dir, "x_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))

# Pré-processamento: normalização e reshape
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# Modelo base
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = create_model()

# Função para salvar métricas no CSV
def save_metrics(sensor_id: str, rnd: int, loss_before: float, acc_before: float, loss_after: float, acc_after: float):
    csv_path = f"models/{sensor_id}/metrics_{sensor_id}.csv"
    df = pd.DataFrame([{ 
        "round": rnd,
        "loss_before": loss_before,
        "acc_before": acc_before,
        "loss_after": loss_after,
        "acc_after": acc_after
    }])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

# Geração de gráfico por sensor
def plot_metrics(sensor_id: str):
    csv_path = f"models/{sensor_id}/metrics_{sensor_id}.csv"
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    rounds = df['round']
    plt.figure()
    plt.plot(rounds, df['loss_before'], label="Loss Before")
    plt.plot(rounds, df['loss_after'], label="Loss After")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss - {sensor_id}")
    plt.savefig(f"models/{sensor_id}/loss_{sensor_id}.png")
    
    plt.figure()
    plt.plot(rounds, df['acc_before'], label="Accuracy Before")
    plt.plot(rounds, df['acc_after'], label="Accuracy After")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy - {sensor_id}")
    plt.savefig(f"models/{sensor_id}/acc_{sensor_id}.png")

# Cliente federado
class FederatedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        global model
        model.set_weights(parameters)

        # Avaliação antes do fit
        loss_before, acc_before = model.evaluate(x_test, y_test, verbose=0)

        model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=0)

        # Avaliação após o fit
        loss_after, acc_after = model.evaluate(x_test, y_test, verbose=0)

        save_metrics(sensor_id, config["epoch_global"], loss_before, acc_before, loss_after, acc_after)

        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return float(loss), len(x_test), {"accuracy": float(accuracy)}

# Inicialização do cliente
if __name__ == "__main__":
    #    fl.client.start_client(server_address="localhost:8080",client=FederatedClient().to_client())
    fl.client.start_client(server_address="172.27.27.3:8080",client=FederatedClient().to_client())
 
    plot_metrics(sensor_id)

#server_address="sfl01:8080"

