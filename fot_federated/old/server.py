import flwr as fl
import tensorflow as tf
import os
from flwr.common import parameters_to_ndarrays



# Criação do modelo CNN para MNIST
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


# Salvamento do modelo em .h5 e .tflite
def save_model(model):
    os.makedirs("model_global", exist_ok=True)
    model.save("model_global/model_final.h5")
    print("✅ Modelo salvo como model_final.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model_global/model_final.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Modelo salvo como model_final.tflite")


# Estratégia personalizada com salvamento de modelo no final
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, max_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = create_model()
        self.max_rounds = max_rounds  # Armazena o número total de rodadas

    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            self.global_model.set_weights(parameters_to_ndarrays(aggregated_weights))
            print(f"✅ Pesos agregados na rodada {server_round}")
            if server_round == self.max_rounds:
                save_model(self.global_model)
        return aggregated_weights, aggregated_metrics



# Inicialização do servidor federado
if __name__ == "__main__":
    print("🚀 Servidor federado iniciando...")

    NUM_ROUNDS = 5

    strategy = SaveModelStrategy(
        max_rounds=NUM_ROUNDS,  # <-- novo
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"epoch_global": rnd},
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )



