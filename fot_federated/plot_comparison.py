import os
import pandas as pd
import matplotlib.pyplot as plt

def load_all_metrics(models_dir="models"):
    all_data = []
    for device_id in os.listdir(models_dir):
        path = os.path.join(models_dir, device_id)
        if os.path.isdir(path):
            csv_file = os.path.join(path, f"metrics_{device_id}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df["device_id"] = device_id
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else None

def plot_comparison(df):
    grouped = df.groupby(["strategy", "round"])[["loss_after", "acc_after"]].mean()

    # Plot Loss
    plt.figure()
    for strategy in df["strategy"].unique():
        strategy_data = grouped.loc[strategy]
        plt.plot(strategy_data.index, strategy_data["loss_after"], label=strategy)
    plt.xlabel("Round")
    plt.ylabel("Average Loss")
    plt.title("Loss After Training - Comparison by Strategy")
    plt.legend()
    plt.savefig("comparison_loss.png")
    print("📊 Gráfico salvo: comparison_loss.png")

    # Plot Accuracy
    plt.figure()
    for strategy in df["strategy"].unique():
        strategy_data = grouped.loc[strategy]
        plt.plot(strategy_data.index, strategy_data["acc_after"], label=strategy)
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.title("Accuracy After Training - Comparison by Strategy")
    plt.legend()
    plt.savefig("comparison_accuracy.png")
    print("📊 Gráfico salvo: comparison_accuracy.png")

if __name__ == "__main__":
    df = load_all_metrics()
    if df is not None:
        plot_comparison(df)
    else:
        print("Nenhum dado encontrado para gerar os gráficos.")
