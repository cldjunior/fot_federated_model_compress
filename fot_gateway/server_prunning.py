
import flwr as fl
import tensorflow as tf
import argparse
import os
import numpy as np
import csv
import tensorflow_model_optimization as tfmot

from flwr.server.strategy import FedAvg, FedProx, FedAdam
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox", "fedadam"])
args = parser.parse_args()

# Modelo base
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
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(28, 28, 1)),
    #     tf.keras.layers.Conv2D(32, 3, activation="relu"),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(64, 3, activation="relu"),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(10, activation="softmax")
    # ])
    
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

# Quantização inversa
def dequantize_weights(weights, level):
    if level == "int8" or level == "float16":
        return [w.astype(np.float32) for w in weights]
    return weights

# Salvamento de modelo
def save_model(model):
    model = tfmot.sparsity.keras.strip_pruning(model)
    os.makedirs("model_global", exist_ok=True)
    model.save("/model_global/model_final.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("/model_global/model_final.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Modelo salvo em H5 e TFLite.")

# Estratégia personalizada
class SaveModelStrategy:
    def __init__(self, base_strategy_class, model, max_rounds, **kwargs):
        self.global_model = model
        self.max_rounds = max_rounds
        self.metrics_path = "./metrics_global.csv"
        initial_parameters = ndarrays_to_parameters(model.get_weights())
        self.base = base_strategy_class(initial_parameters=initial_parameters, **kwargs)

        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "avg_accuracy", "avg_loss"])

    def aggregate_fit(self, server_round, results, failures):
        weights_list = []
        accs, losses = [], []

        for client_proxy, fit_res in results:
            parameters = fit_res.parameters
            metrics = fit_res.metrics
            quant = metrics.get("quant", "float32")
            local_weights = dequantize_weights(parameters_to_ndarrays(parameters), quant)
            weights_list.append(local_weights)
            accs.append(metrics.get("accuracy", 0.0))
            losses.append(metrics.get("loss", 0.0))

        if weights_list:
            avg_weights = [np.mean([w[i] for w in weights_list], axis=0) for i in range(len(weights_list[0]))]
            self.global_model.set_weights(avg_weights)

            if server_round >= self.max_rounds:
                save_model(self.global_model)

            # Salva métricas agregadas
            avg_acc = np.mean(accs) if accs else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"📊 Round {server_round} - Accuracy: {avg_acc:.4f}, Loss: {avg_loss:.4f}")
            with open(self.metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([server_round, avg_acc, avg_loss])

            return ndarrays_to_parameters(avg_weights), {}

        return None, {}

    def __getattr__(self, name):
        return getattr(self.base, name)

# Estratégia
def get_strategy(name: str, model, max_rounds: int):
    common_args = dict(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
        on_fit_config_fn=lambda rnd: {"epoch_global": rnd, "strategy": name}
    )

    if name == "fedprox":
        return SaveModelStrategy(FedProx, model, max_rounds, proximal_mu=0.1, **common_args)
    elif name == "fedadam":
        return SaveModelStrategy(FedAdam, model, max_rounds, **common_args)
    else:
        return SaveModelStrategy(FedAvg, model, max_rounds, **common_args)

# Inicialização
if __name__ == "__main__":
    NUM_ROUNDS =5
    model = create_model()
    print(f"Iniciando servidor federado com estratégia: {args.strategy}")
    strategy = get_strategy(args.strategy.lower(), model, NUM_ROUNDS)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
