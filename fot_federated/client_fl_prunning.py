import argparse
import os
import time
import numpy as np
import tensorflow as tf
import flwr as fl
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import gc
import tensorflow_model_optimization as tfmot

from flwr.common import NDArrays
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=str, required=True)
parser.add_argument("--quant_type", type=str, default="int8", choices=["float32", "float16", "int8", "progressive", "heterogeneous"])
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
        #return [((w / np.max(np.abs(w))) * 127).astype(np.int8) if w.dtype in [np.float32, np.float64] else w for w in weights]
        return weights
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
    
    # Cronograma de pruning
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )
    }    
    
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=(28, 28, 1)),
    #     tf.keras.layers.Conv2D(32, 3, activation="relu"),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(64, 3, activation="relu"),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(10, activation="softmax")
    # ])
    # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Apenas as camadas Dense serão podadas
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(128, activation="relu"), **pruning_params),
        tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(10, activation="softmax"), **pruning_params),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
 
    return model

# Save metrics with system info
def save_metrics(
    device_id, rnd, loss_b, acc_b, loss_a, acc_a, strategy, qtype,
    train_time, model_size, cpu, ram, ram_free, n_samples,
    precision, recall, f1, disk_usage, disk_free
):
    path = f"models/{device_id}/metrics_{device_id}.csv"
    columns = [
        "strategy", "round", "loss_before", "acc_before", "loss_after", "acc_after",
        "quant", "train_time", "model_size_bytes", "cpu_percent", "ram_percent", "ram_available",
        "n_samples", "precision", "recall", "f1_score", "disk_percent", "disk_available"
    ]
    row = [
        strategy, rnd, loss_b, acc_b, loss_a, acc_a, qtype, train_time,
        model_size, cpu, ram, ram_free, n_samples, precision, recall, f1,
        disk_usage, disk_free
    ]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = writer(f)
        if write_header:
            w.writerow(columns)
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


    # def fit(self, parameters: NDArrays, config):
    #     process = psutil.Process(os.getpid())
    #     rss = process.memory_info().rss / (1024 ** 2)
    #     print(f"[MEMORY] Uso de RAM início fit: {rss:.2f} MB")
    #     self.model.set_weights(parameters)
    #     round_num = config.get("epoch_global", 0)
    #     strategy = config.get("strategy", "unknown")

    #     if QUANT_TYPE == "progressive":
    #         current_q = get_quant_type_from_round(round_num)
    #     elif QUANT_TYPE == "heterogeneous":
    #         current_q = QUANT_TYPE_MAP.get(DEVICE, "float32")
    #     else:
    #         current_q = QUANT_TYPE

    #     # Evaluate BEFORE training
    #     print(f"[CHECK] Antes do evaluate pós-treinamento")
    #     loss_b, acc_b = self.model.evaluate(x_test, y_test, verbose=0)

    #     # Local training
    #     print(f"[CHECK] Antes do treinamento")
    #     start = time.time()
        
    #     #self.model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    #     #callback de pruning
    #     self.model.fit(
    #                     x_train,
    #                     y_train,
    #                     epochs=3,
    #                     batch_size=32,
    #                     verbose=0,
    #                     callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    #                 )
        
    #     train_time = time.time() - start

    #     # Evaluate AFTER training
    #     print(f"[CHECK] Depois do evaluate pós-treinamento")
    #     loss_a, acc_a = self.model.evaluate(x_test, y_test, verbose=0)
    #     print(f"[CHECK] Antes do Predict")
    #     y_pred = np.argmax(self.model.predict(x_test, verbose=0), axis=1)

    #     precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    #     recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    #     f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    #     print(f"[CHECK] Depois do predict")
    #     # Quantize weights
    #     weights = quantize_weights(self.model.get_weights(), current_q)

    #     # System metrics
    #     print(f"[MEMORY] Antes da conversão: {psutil.virtual_memory().percent}% usado")

    #     # Remove o wrapper de pruning antes de exportar o modelo
    #     self.model = tfmot.sparsity.keras.strip_pruning(self.model)

    

    #     converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
    #     converter._experimental_new_converter = True
        
    #     if current_q == "float16":
    #        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #        converter.target_spec.supported_types = [tf.float16]

    #     elif current_q == "int8":
    #        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #        def representative_dataset():
    #            for i in range(100):
    #                yield [x_train[i:i+1]]
    #        converter.representative_dataset = representative_dataset
    #        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #        converter.inference_input_type = tf.uint8
    #        converter.inference_output_type = tf.uint8

    #     # Conversão e salvamento
        
    #     tflite_model = converter.convert()
    #     print(f"[MEMORY] Depois da conversão: {psutil.virtual_memory().percent}% usado")
    #     tflite_path = f"models/{DEVICE}/model_round_{round_num}_{current_q}.tflite"
    #     with open(tflite_path, "wb") as f:
    #         f.write(tflite_model)

    #     # Tamanho real do modelo quantizado
    #     model_size = os.path.getsize(tflite_path)

    #     # Salvar modelo em .h5 apenas para referência
    #     #self.model.save(f"models/{DEVICE}/model_round_{round_num}.h5")


    #     # System metrics
        
    #     cpu = psutil.cpu_percent(interval=None)
    #     ram = psutil.virtual_memory().percent
    #     ram_free = psutil.virtual_memory().available
    #     disk = psutil.disk_usage("/").percent
    #     disk_free = psutil.disk_usage("/").free

    #     # Salvar métricas
    #     os.makedirs(f"models/{DEVICE}", exist_ok=True)
    #     save_metrics(
    #         DEVICE, round_num, loss_b, acc_b, loss_a, acc_a, strategy, current_q, train_time,
    #         model_size, cpu, ram, ram_free, len(x_train), precision, recall, f1, disk, disk_free
    #     )
    #     print(f"✅ Round {round_num}: métricas salvas com sucesso ({model_size / 1024:.2f} KB, tipo: {current_q})")
        
    #     gc.collect()
    #     tf.keras.backend.clear_session()
    #     time.sleep(1)
    #     print(f"[CHECK] Fim da rodada")
    #     return weights, len(x_train), {
    #         "loss": loss_a, "accuracy": acc_a, "quant": current_q,
    #         "precision": precision, "recall": recall, "f1_score": f1
    #     }
    def fit(self, parameters: NDArrays, config):
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 ** 2)
        print(f"[MEMORY] Uso de RAM início fit: {rss:.2f} MB")
        self.model.set_weights(parameters)
        round_num = config.get("epoch_global", 0)
        strategy = config.get("strategy", "unknown")

        if QUANT_TYPE == "progressive":
            current_q = get_quant_type_from_round(round_num)
        elif QUANT_TYPE == "heterogeneous":
            current_q = QUANT_TYPE_MAP.get(DEVICE, "float32")
        else:
            current_q = QUANT_TYPE

        print(f"[CHECK] Antes do evaluate pós-treinamento")
        loss_b, acc_b = self.model.evaluate(x_test, y_test, verbose=0)

        print(f"[CHECK] Antes do treinamento")
        start = time.time()
        self.model.fit(
            x_train,
            y_train,
            epochs=3,
            batch_size=32,
            verbose=0,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
        )
        train_time = time.time() - start

        print(f"[CHECK] Depois do evaluate pós-treinamento")
        loss_a, acc_a = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"[CHECK] Antes do Predict")
        y_pred = np.argmax(self.model.predict(x_test, verbose=0), axis=1)

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"[CHECK] Depois do predict")

        weights = quantize_weights(self.model.get_weights(), current_q)
        print(f"[MEMORY] Antes da conversão: {psutil.virtual_memory().percent}% usado")

        # Cria uma cópia limpa do modelo para exportação
        stripped_model = tfmot.sparsity.keras.strip_pruning(self.model)

        converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
        converter._experimental_new_converter = True

        if current_q == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif current_q == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            def representative_dataset():
                for i in range(100):
                    yield [x_train[i:i+1]]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        print(f"[MEMORY] Depois da conversão: {psutil.virtual_memory().percent}% usado")
        tflite_path = f"models/{DEVICE}/model_round_{round_num}_{current_q}.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        model_size = os.path.getsize(tflite_path)
        stripped_model.save(f"models/{DEVICE}/model_round_{round_num}.h5")

        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        ram_free = psutil.virtual_memory().available
        disk = psutil.disk_usage("/").percent
        disk_free = psutil.disk_usage("/").free

        os.makedirs(f"models/{DEVICE}", exist_ok=True)
        save_metrics(
            DEVICE, round_num, loss_b, acc_b, loss_a, acc_a, strategy, current_q, train_time,
            model_size, cpu, ram, ram_free, len(x_train), precision, recall, f1, disk, disk_free
        )
        print(f"✅ Round {round_num}: métricas salvas com sucesso ({model_size / 1024:.2f} KB, tipo: {current_q})")

        gc.collect()
        tf.keras.backend.clear_session()
        time.sleep(1)
        print(f"[CHECK] Fim da rodada")
        return weights, len(x_train), {
            "loss": loss_a, "accuracy": acc_a, "quant": current_q,
            "precision": precision, "recall": recall, "f1_score": f1
        }

    def evaluate(self, parameters: NDArrays, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": acc}

if __name__ == "__main__":
    fl.client.start_client(server_address="172.27.27.4:8080", client=FederatedClient().to_client())
    plot_metrics(DEVICE)



