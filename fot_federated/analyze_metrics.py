import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

input_dir = "./models"
output_dir = "./metrics_outputs"
os.makedirs(output_dir, exist_ok=True)

print("📥 Carregando métricas...")

all_data = []
for device in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, device, f"metrics_{device}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["client_id"] = device
            all_data.append(df)
        except Exception as e:
            print(f"Erro ao ler {path}: {e}")

if not all_data:
    print("❌ Nenhuma métrica encontrada.")
    exit()

df = pd.concat(all_data, ignore_index=True)

print("📊 Gerando métricas consolidadas...")

# Salvar CSV consolidado
df.to_csv(os.path.join(output_dir, "metrics_all_clients.csv"), index=False)

# Gráficos de barras para cada métrica
def plot_bar(metric, ylabel, title, filename, formatter=None):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="client_id", y=metric, estimator=np.mean, errorbar="sd")
    plt.ylabel(ylabel)
    plt.xlabel("Client")
    plt.title(title)
    if formatter:
        plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_bar("train_time", "Training Time (s)", "⏱️ Avg Training Time per Client", "plot_train_time_per_client.png")
plot_bar("acc_after", "Accuracy", "🎯 Accuracy After Training per Client", "plot_accuracy_after_per_client.png")
plot_bar("loss_after", "Loss", "📉 Loss After Training per Client", "plot_loss_after_per_client.png")


if "model_size" in df.columns:
    plot_bar("model_size", "Model Size (KB)", "💾 Model Size per Client", "plot_model_size_per_client.png")

if "memory_usage" in df.columns:
    plot_bar("memory_usage", "Memory (MB)", "🧠 Memory Usage per Client", "plot_memory_usage_per_client.png")

if "cpu_usage" in df.columns:
    plot_bar("cpu_usage", "CPU (%)", "⚙️ CPU Usage per Client", "plot_cpu_usage_per_client.png")

print("✅ Métricas consolidadas e gráficos salvos em:", output_dir)
