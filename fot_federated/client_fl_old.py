import flwr as fl
import tensorflow as tf
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from typing import Dict, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from csv import writer


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
parser.add_argument("--device_id", required=True, help="ID do device (ex: dc01, dc02...)")
args = parser.parse_args()
device_id = args.device_id

# Caminho do dataset local
data_dir = os.path.join("models", args.device_id)
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
#def save_metrics(device_id: str, rnd: int, loss_before: float, acc_before: float, loss_after: float, acc_after: float, strategy: str, train_time: float):
    # csv_path = f"models/{device_id}/metrics_{device_id}.csv"
    # df = pd.DataFrame([{ 
    #     "strategy": strategy,
    #     "round": rnd,
    #     "loss_before": loss_before,
    #     "acc_before": acc_before,
    #     "loss_after": loss_after,
    #     "acc_after": acc_after,
    #     "train_time": train_time
    # }])
    # if os.path.exists(csv_path):
    #     df.to_csv(csv_path, mode='a', header=False, index=False)
    # else:
    #     df.to_csv(csv_path, index=False)
def save_metrics(device_id: str, rnd: int, loss_before: float, acc_before: float, loss_after: float, acc_after: float, strategy: str, train_time: float):
    csv_path = f"models/{device_id}/metrics_{device_id}.csv"
    columns = ["strategy", "round", "loss_before", "acc_before", "loss_after", "acc_after", "train_time"]
    row = [strategy, rnd, loss_before, acc_before, loss_after, acc_after, train_time]

    # Cria arquivo com cabeçalho se não existir
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        csv_writer = writer(f)
        if write_header:
            csv_writer.writerow(columns)
        csv_writer.writerow(row)    
    
    
    
    

# Geração de gráfico por sensor
def plot_metrics(device_id: str):
    csv_path = f"models/{device_id}/metrics_{device_id}.csv"
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
    plt.title(f"Loss - {device_id}")
    plt.savefig(f"models/{device_id}/loss_{device_id}.png")
    
    plt.figure()
    plt.plot(rounds, df['acc_before'], label="Accuracy Before")
    plt.plot(rounds, df['acc_after'], label="Accuracy After")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy - {device_id}")
    plt.savefig(f"models/{device_id}/acc_{device_id}.png")

# Cliente federado
class FederatedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        global model
        model.set_weights(parameters)

        # Avaliação antes do fit
        loss_before, acc_before = model.evaluate(x_test, y_test, verbose=0)
        
        start_time = time.time()
        model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=0)
        train_time = time.time() - start_time
        
        # Avaliação após o fit
        loss_after, acc_after = model.evaluate(x_test, y_test, verbose=0)

        strategy = config.get("strategy", "unknown")
        round_num = config.get("epoch_global", -1)
        
        #save_metrics(device_id, config["epoch_global"], loss_before, acc_before, loss_after, acc_after, strategy)
        save_metrics(device_id, round_num, loss_before, acc_before, loss_after, acc_after, strategy, train_time)
         
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return float(loss), len(x_test), {"accuracy": float(accuracy)}

# Inicialização do cliente
if __name__ == "__main__":
    fl.client.start_client(server_address="172.27.27.4:8080", client=FederatedClient().to_client())
    plot_metrics(device_id)