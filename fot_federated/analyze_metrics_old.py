import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_combine_metrics(base_path="./models"):
    paths = glob.glob(os.path.join(base_path, "dc*/metrics_dc*.csv"))
    df_all = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            client_id = os.path.basename(path).replace("metrics_", "").replace(".csv", "")
            df["client_id"] = client_id
            df_all.append(df)
        except Exception as e:
            print(f"Erro ao ler {path}: {e}")
    if df_all:
        return pd.concat(df_all, ignore_index=True)
    return pd.DataFrame()

def save_statistics(df, output_dir="metrics_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    stats = df.groupby("client_id").agg({
        "loss_before": ["mean", "std"],
        "acc_before": ["mean", "std"],
        "loss_after": ["mean", "std"],
        "acc_after": ["mean", "std"],
        "train_time": ["mean", "std"]
    }).reset_index()
    stats.columns = ["_".join(col).strip("_") for col in stats.columns.values]
    stats.to_csv(os.path.join(output_dir, "metrics_stats_by_client.csv"), index=False)
    return stats

def plot_metrics(df, output_dir="metrics_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Tempo médio por cliente
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="client_id", y="train_time", estimator=np.mean, ci="sd")
    plt.title("⏱️ Tempo Médio de Treinamento por Cliente")
    plt.ylabel("Tempo (s)")
    plt.xlabel("Cliente")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_train_time_per_client.png"))
    plt.close()

    # Acurácia após o treinamento
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="client_id", y="acc_after", estimator=np.mean, ci="sd")
    plt.title("🎯 Acurácia Média Após o Treinamento por Cliente")
    plt.ylabel("Acurácia")
    plt.xlabel("Cliente")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_accuracy_after_per_client.png"))
    plt.close()

if __name__ == "__main__":
    print("📥 Carregando métricas...")
    df = load_and_combine_metrics()

    if df.empty:
        print("❌ Nenhum arquivo encontrado em ./models/dc*/metrics_dc*.csv")
    else:
        print("📊 Gerando métricas consolidadas...")
        os.makedirs("metrics_outputs", exist_ok=True)

        df.to_csv("metrics_outputs/metrics_consolidated.csv", index=False)
        stats = save_statistics(df)
        plot_metrics(df)
        print("✅ Métricas consolidadas e gráficos salvos em: ./metrics_outputs/")
