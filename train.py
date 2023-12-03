import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np
import os
import pandas as pd
from utils import (
    normalizeLst,
    normalizeElem,
    denormalizeElem,
    meanError0,
    meanError1,
    computeCost,
    computeRMSE,
)

# Paramètres initiaux
theta0 = 0
theta1 = 0
learning_rate = 0.05
epochs = 5000
current_epoch = 0
display_step = 100
cost_history = []
rmse_history = []
training_threshold = 0.000001

# The algorithm : iterating towards local minimum
def gradientDescent(epochs, learning_rate, normedMileages, normedPrices):
    global theta0, theta1, current_epoch, cost_history, rmse_history
    for i in range(epochs):
        current_epoch += 1
        theta0 = theta0 - learning_rate * meanError0(
            normedMileages, normedPrices, theta0, theta1
        )
        theta1 = theta1 - learning_rate * meanError1(
            normedMileages, normedPrices, theta0, theta1
        )
        cost_history.append(computeCost(normedMileages, normedPrices, theta0, theta1))
        rmse_history.append(computeRMSE(normedMileages, normedPrices, theta0, theta1))


def load_data():
    filename = "data/data.csv"
    if not os.path.exists(filename):
        print("File not found", filename)
        exit()

    data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    df = pd.DataFrame(data, columns=["mileage", "price"])
    return df


def update(frame, axs, df, normedMileages, normedPrices):
    print("yo")
    global current_epoch
    print(f"Epoch {current_epoch}")

    # Nettoyage des graphiques
    axs[0, 0].cla()
    axs[0, 1].cla()


    # Mise à jour du graphique classique
    axs[0, 0].scatter(df["mileage"], df["price"])
    axs[0, 0].set_xlabel("Mileage")
    axs[0, 0].set_ylabel("Price")
    axs[0, 0].set_title(f"Price of Cars Epoch {current_epoch}")

    # Calculer la ligne de prédiction
    lineX = np.array([min(df["mileage"]), max(df["mileage"])])
    lineY = theta0 + theta1 * lineX
    axs[0, 0].plot(lineX, lineY, color="red")

def update_graphs(axs, df, normedMileages, normedPrices):
    global theta0, theta1, current_epoch
    print(f"Epoch {current_epoch}")

    # Nettoyage des graphiques
    classicPlot = axs[0, 0]
    normedPlot = axs[0, 1]
    costPlot = axs[1, 0]
    rmsePlot = axs[1, 1]
    classicPlot.cla()
    normedPlot.cla()
    costPlot.cla()
    rmsePlot.cla()
    normedPlot.set_title(f"Price of cars (normed) Epoch {current_epoch}")
    normedPlot.set_xlabel("Mileage")
    normedPlot.set_ylabel("Price")
    classicPlot.set_xlabel("Mileage")
    classicPlot.set_ylabel("Price")
    classicPlot.set_title(f"Price of cars Epoch {current_epoch}")
    classicPlot.scatter(df["mileage"], df["price"])
    normedPlot.scatter(normedMileages, normedPrices)


    # Calculate linear equation not normalized
    lineX = [float(min(df["mileage"])), float(max(df["mileage"]))]
    min_normed_mileage = min(normedMileages)
    max_normed_mileage = max(normedMileages)
    lineY = []
    for elem in lineX:
        elem = theta1 * normalizeElem(df["mileage"], elem) + theta0
        lineY.append(denormalizeElem(df["price"], elem))
    classicPlot.plot(
        lineX,
        lineY,
        color="red",
    )

    # Normed plot
    min_normed_mileage = min(normedMileages)
    max_normed_mileage = max(normedMileages)
    normedPlot.plot(
        [min_normed_mileage, max_normed_mileage],
        [
            theta0 + theta1 * min_normed_mileage,
            theta0 + theta1 * max_normed_mileage,
        ],
        color="red",
    )
    costPlot.set_xlabel("Epoch")
    costPlot.set_ylabel("Cost")
    costPlot.set_title("Cost Function Over Time")
    costPlot.plot(range(current_epoch), cost_history[:current_epoch], color="blue")

    rmsePlot.set_xlabel("Epoch")
    rmsePlot.set_ylabel("RMSE")
    rmsePlot.set_title("RMSE Over Time")
    rmsePlot.plot(range(current_epoch), rmse_history[:current_epoch], color="green")

def train(axs, df, normedMileages, normedPrices):
    for epoch in range(int(epochs / display_step)):
        if (len(cost_history) > 1 and cost_history[-1] > cost_history[-2] - training_threshold):
            print("Cost is not decreasing anymore, stopping training")
            break
        gradientDescent(display_step, learning_rate, normedMileages, normedPrices)
        update_graphs(axs, df, normedMileages, normedPrices)
        plt.gcf().canvas.draw_idle()   # Redessiner le graphique
        plt.gcf().canvas.flush_events()  # Traiter les événements de l'interface utilisateur

def show_results(df): 

    # Calcul et affichage du RMSE final et du pourcentage
    final_rmse = rmse_history[-1]

    def RMSE_percent(cost):
        RMSE = 100 * (1 - cost**0.5)
        return RMSE

    def MSE_percent(cost):
        MSE = 100 * (1 - cost)
        return MSE

    print("theta finaux normalisés")
    print(theta0)
    print(theta1)
    print("theta finaux")
    print(denormalizeElem(df["price"], theta0))
    print(denormalizeElem(df["price"], theta1))
    print("cost")
    print(cost_history[-1])
    print("rmse")
    print(rmse_history[-1])
    rmse_percentage = RMSE_percent(cost_history[-1])
    mse_percentage = MSE_percent(cost_history[-1])
    print("RMSE en pourcentage : {:.2f}%".format(rmse_percentage))
    print("MSE en pourcentage : {:.2f}%".format(mse_percentage))


def main():
    df = load_data()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    classicPlot = axs[0, 0]
    normedPlot = axs[0, 1]
    costPlot = axs[1, 0]
    rmsePlot = axs[1, 1]
    classicPlot.scatter(df["mileage"], df["price"])
    classicPlot.set_xlabel("Mileage")
    classicPlot.set_ylabel("Price")
    classicPlot.set_title("Price of cars")

    normedMileages = normalizeLst(df["mileage"])
    normedPrices = normalizeLst(df["price"])

    normedPlot.scatter(normedMileages, normedPrices)
    normedPlot.set_xlabel("Mileage")
    normedPlot.set_ylabel("Price")
    normedPlot.set_title("Price of cars (Normed)")

    costPlot.set_xlabel("Epoch")
    costPlot.set_ylabel("Cost")
    costPlot.set_title("Cost Function Over Time")
    costPlot.plot(range(current_epoch), cost_history[:current_epoch], color="blue")

    rmsePlot.set_xlabel("Epoch")
    rmsePlot.set_ylabel("RMSE")
    rmsePlot.set_title("RMSE Over Time")
    fig.subplots_adjust(hspace=0.3)  # Add space between rows

    ax_button = plt.axes([0.05, 0.9, 0.1, 0.075])
    btn = Button(ax_button, 'Start training')

    def on_button_clicked(event):
        btn.label.set_text("Training...") 
        btn.color = 'gray' 
        btn.hovercolor = 'gray'
        btn.active = False
        normedMileages = normalizeLst(df["mileage"])
        normedPrices = normalizeLst(df["price"])
        train(axs, df, normedMileages, normedPrices)
        show_results(df)

    btn.on_clicked(on_button_clicked)
    plt.show()



if __name__ == "__main__":
    main()
