
import os
import argparse
import tensorflow as tf
import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg, FedProx, FedAdam
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox", "fedadam"], help="Estratégia de federação")
args = parser.parse_args()

# Criação do modelo
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

# Salvamento final
def save_model(model):
    os.makedirs("model_global", exist_ok=True)
    model.save("model_global/model_final.h5")
    print("✅ Modelo salvo como model_final.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model_global/model_final.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Modelo salvo como model_final.tflite")

# Estratégia com suporte a quantizações
class SaveModelStrategy:
    def __init__(self, base_strategy_class, max_rounds, **kwargs):
        self.base = base_strategy_class(**kwargs)
        self.global_model = create_model()
        self.max_rounds = max_rounds

    def aggregate_fit(self, server_round, results, failures):
        # Conversão e normalização de pesos heterogêneos
        converted_results = []
        for r in results:
            weights = parameters_to_ndarrays(r.parameters)
            weights = [w.astype(np.float32) for w in weights]  # reescala se necessário
            converted_results.append((weights, r.num_examples))

        total_examples = sum(num for _, num in converted_results)
        agg_weights = [
            sum((w[i] * num for w, num in converted_results)) / total_examples
            for i in range(len(converted_results[0][0]))
        ]
        self.global_model.set_weights(agg_weights)
        print(f"✅ Pesos agregados na rodada {server_round}")

        # Salva modelo final
        if server_round == self.max_rounds:
            save_model(self.global_model)

        return ndarrays_to_parameters(agg_weights), {}

    def __getattr__(self, attr):
        return getattr(self.base, attr)

def get_strategy(name: str, max_rounds: int):
    common_kwargs = dict(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"epoch_global": rnd, "strategy": args.strategy.lower()}
    )
    if name == "fedprox":
        return SaveModelStrategy(FedProx, max_rounds, proximal_mu=0.1, **common_kwargs)
    elif name == "fedadam":
        return SaveModelStrategy(FedAdam, max_rounds, **common_kwargs)
    else:
        return SaveModelStrategy(FedAvg, max_rounds, **common_kwargs)

if __name__ == "__main__":
    print("Host agregador federado iniciando...")
    NUM_ROUNDS = 5
    strategy = get_strategy(args.strategy.lower(), NUM_ROUNDS)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
