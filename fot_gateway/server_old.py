import flwr as fl
import tensorflow as tf
import os
import argparse
import time
import pandas as pd 
from flwr.server.strategy import FedAvg, FedProx, FedAdam
from flwr.common import parameters_to_ndarrays

# Argumentos de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox", "fedadam"], help="Estratégia de federação")
args = parser.parse_args()

# Criação do modelo base
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

# Salvamento do modelo final
def save_model(model):
    os.makedirs("model_global", exist_ok=True)
    model.save("model_global/model_final.h5")
    print("✅ Modelo salvo como model_final.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model_global/model_final.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Modelo salvo como model_final.tflite")

# Estratégia com salvamento no final da última rodada
class SaveModelStrategy:
    def __init__(self, base_strategy_class, max_rounds, **kwargs):
        self.base = base_strategy_class(**kwargs)
        self.global_model = create_model()
        self.max_rounds = max_rounds

    # def aggregate_fit(self, server_round, results, failures):
    #     aggregated_weights, aggregated_metrics = self.base.aggregate_fit(server_round, results, failures)
    #     if aggregated_weights is not None:
    #         self.global_model.set_weights(parameters_to_ndarrays(aggregated_weights))
    #         print(f"✅ Pesos agregados na rodada {server_round}")
    #         if server_round == self.max_rounds:
    #             save_model(self.global_model)
    #     return aggregated_weights, aggregated_metrics
    def aggregate_fit(self, server_round, results, failures):
        start_time = time.time()
        aggregated_weights, aggregated_metrics = self.base.aggregate_fit(server_round, results, failures)
        duration = time.time() - start_time

        if aggregated_weights is not None:
            self.global_model.set_weights(parameters_to_ndarrays(aggregated_weights))
            print(f"✅ Pesos agregados na rodada {server_round}")
            if server_round == self.max_rounds:
                save_model(self.global_model)

        # Salvar métrica global
        accs = [res.metrics["accuracy"] for _, res in results if "accuracy" in res.metrics]
        acc_avg = sum(accs) / len(accs) if accs else 0

        df = pd.DataFrame([{
            "round": server_round,
            "strategy": args.strategy.lower(),
            "num_clients": len(results),
            "accuracy_global": acc_avg,
            "duration_round": duration
        }])
        os.makedirs("metrics_global", exist_ok=True)
        file_path = f"metrics_global/global_metrics.csv"
        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        return aggregated_weights, aggregated_metrics

    def __getattr__(self, attr):
        return getattr(self.base, attr)

# Função para selecionar a estratégia
def get_strategy(name: str, max_rounds: int):
    # Cria o modelo base para obter os parâmetros iniciais
    model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

    common_kwargs = dict(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"epoch_global": rnd, "strategy": args.strategy.lower()}
    )

    if name == "fedprox":
        return SaveModelStrategy(FedProx, max_rounds, proximal_mu=0.1, **common_kwargs)
    elif name == "fedadam":
        # Adiciona initial_parameters exigido por FedAdam
        common_kwargs["initial_parameters"] = initial_parameters
        return SaveModelStrategy(FedAdam, max_rounds, **common_kwargs)
    else:  # default to FedAvg
        return SaveModelStrategy(FedAvg, max_rounds, **common_kwargs)

# Inicialização
if __name__ == "__main__":
    print("Host agregador federado iniciando...")
    NUM_ROUNDS = 5
    strategy = get_strategy(args.strategy.lower(), NUM_ROUNDS)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )