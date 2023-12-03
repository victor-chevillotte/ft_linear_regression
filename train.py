import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np
import os
import pandas as pd
from utils import (
    normalizeLst,
    meanError0,
    meanError1,
    computeCost,
    computePrecision
)
from messages import (
    debug,
    title,
    normal,
    success,
    verbose,
    error
)
# Paramètres initiaux
theta0 = 0
theta1 = 0
learning_rate = 0.01
epochs = 5000
current_epoch = 0
display_step = 100
cost_history = []
precision_history = []
training_threshold = 0.00000001


# The algorithm : iterating towards local minimum
def gradientDescent(epochs, learning_rate, normedMileages, normedPrices, df):
    global theta0, theta1, current_epoch, cost_history, precision_history
    for i in range(epochs):
        current_epoch += 1
        theta0 = theta0 - learning_rate * meanError0(
            normedMileages, normedPrices, theta0, theta1
        )
        theta1 = theta1 - learning_rate * meanError1(
            normedMileages, normedPrices, theta0, theta1
        )
        cost_history.append(computeCost(normedMileages, normedPrices, theta0, theta1))
        theta0_denorm = (
            theta0 * (max(df["price"]) - min(df["price"]))
            + min(df["price"])
            - theta1
            * min(df["mileage"])
            * (max(df["price"]) - min(df["price"]))
            / (max(df["mileage"]) - min(df["mileage"]))
        )
        theta1_denorm = (
            theta1
            * (max(df["price"]) - min(df["price"]))
            / (max(df["mileage"]) - min(df["mileage"]))
        )
        precision = computePrecision(df["mileage"], df["price"], theta0_denorm, theta1_denorm)
        
        precision_history.append(100 - round(precision * 100, 2))

def load_data():
    filename = "data/data.csv"
    if not os.path.exists(filename):
        error("File not found", filename)
        exit()

    data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    df = pd.DataFrame(data, columns=["mileage", "price"])
    return df


def update_graphs(axs, df, normedMileages, normedPrices):
    global theta0, theta1, current_epoch
    print(f"Epoch {current_epoch}", end="\r", flush=True)

    # Nettoyage des graphiques
    classicPlot = axs[0, 0]
    normedPlot = axs[0, 1]
    costPlot = axs[1, 0]
    precisionPlot = axs[1, 1]
    classicPlot.cla()
    normedPlot.cla()
    costPlot.cla()
    precisionPlot.cla()
    normedPlot.set_title(f"Price of cars (normed) Epoch {current_epoch}")
    normedPlot.set_xlabel("Mileage")
    normedPlot.set_ylabel("Price")
    classicPlot.set_xlabel("Mileage")
    classicPlot.set_ylabel("Price")
    classicPlot.set_title(f"Price of cars Epoch {current_epoch}")
    classicPlot.scatter(df["mileage"], df["price"])
    normedPlot.scatter(normedMileages, normedPrices)

    # Denormalize theta values
    # TODO : explicit formula for denormalization

    theta0_denorm = (
        theta0 * (max(df["price"]) - min(df["price"]))
        + min(df["price"])
        - theta1
        * min(df["mileage"])
        * (max(df["price"]) - min(df["price"]))
        / (max(df["mileage"]) - min(df["mileage"]))
    )
    theta1_denorm = (
        theta1
        * (max(df["price"]) - min(df["price"]))
        / (max(df["mileage"]) - min(df["mileage"]))
    )
    classicPlot.plot(
        [min(df["mileage"]), max(df["mileage"])],
        [
            theta0_denorm + theta1_denorm * min(df["mileage"]),
            theta1_denorm * max(df["mileage"]) + theta0_denorm,
        ],
        color="C1",
        label="f(x) = {0}*x + {1}".format(
            round(theta1_denorm, 4), round(theta0_denorm, 4)
        ),
    )
    classicPlot.legend()

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
        label="f(x) = {0}*x + {1}".format(round(theta0, 4), round(theta1, 4)),
    )
    normedPlot.legend()
    costPlot.set_xlabel("Epoch")
    costPlot.set_ylabel("Cost")
    costPlot.set_title("Cost Function Over Time")
    costPlot.plot(range(current_epoch), cost_history[:current_epoch], color="blue")

    precisionPlot.set_xlabel("Epoch")
    precisionPlot.set_ylabel("Precision %")
    precisionPlot.set_title("Precision Over Time")
    precisionPlot.plot(range(current_epoch), precision_history[:current_epoch], color="green")


def train(axs, df, normedMileages, normedPrices):
    success("Starting training !")
    for epoch in range(int(epochs / display_step)):
        if (
            len(cost_history) > 1
            and cost_history[-1] > cost_history[-2] - training_threshold
        ):
            normal("Cost function is not decreasing anymore (< e^-7), stopping training")
            break
        gradientDescent(display_step, learning_rate, normedMileages, normedPrices, df)
        update_graphs(axs, df, normedMileages, normedPrices)
        plt.gcf().canvas.draw_idle()  # Redessiner le graphique
        plt.gcf().canvas.flush_events()  # Traiter les événements de l'interface utilisateur

def store_results():
    title("Storing results...")
    global theta0, theta1, cost_history, precision_history
    try :
        with open("data/results.txt", "w") as f:
            f.write(f"theta0 : {theta0}\n")
            f.write(f"theta1 : {theta1}\n")
        success("Results stored !")
    except Exception as e:
        error("Error while storing results", e)


def show_results(axs, df, btn):
    global theta0, theta1, cost_history, precision_history
    btn.label.set_text("Training finished !")
    success("Training finished !")
    title("Final thetas :")
    normal(theta0)
    normal(theta1)

    theta0_denorm = (
        theta0 * (max(df["price"]) - min(df["price"]))
        + min(df["price"])
        - theta1
        * min(df["mileage"])
        * (max(df["price"]) - min(df["price"]))
        / (max(df["mileage"]) - min(df["mileage"]))
    )
    theta1_denorm = (
        theta1
        * (max(df["price"]) - min(df["price"]))
        / (max(df["mileage"]) - min(df["mileage"]))
    )
    title("Final denormalized thetas :")
    normal(theta0_denorm)
    normal(theta1_denorm)

    title("Final cost :")
    normal(cost_history[-1])

    title("Final precision :")
    error = computePrecision(df["mileage"], df["price"], theta0_denorm, theta1_denorm) 
    normal(f"{100 - round(error * 100, 2)} % (average error : {round(error * 100, 2)} %)")
    store_results()
    # Plot cost and precision
    costPlot = axs[1, 0]
    precisionPlot = axs[1, 1]

    # Annotate final theta values on the classic and normalized plots
    cost_text = f"Final cost: {round(cost_history[-1], 4)}"
    costPlot.annotate(cost_text, xy=(0.5, 0.8), xycoords='axes fraction', ha='center', fontsize=9, color='blue')
    precision_text = f"Final precision: {round(100 - round(error * 100, 2), 2)}%"
    precisionPlot.annotate(precision_text, xy=(0.5, 0.8), xycoords='axes fraction', ha='center', fontsize=9, color='green')
    plt.show()

def main():
    df = load_data()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    classicPlot = axs[0, 0]
    normedPlot = axs[0, 1]
    costPlot = axs[1, 0]
    precisionPlot = axs[1, 1]
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

    precisionPlot.set_xlabel("Epoch")
    precisionPlot.set_ylabel("Precision %")
    precisionPlot.set_title("Precision Over Time")
    fig.subplots_adjust(hspace=0.3)  # Add space between rows

    ax_button = plt.axes([0.05, 0.93, 0.2, 0.05])
    btn = Button(ax_button, "Start training")

    def on_button_clicked(event):
        btn.label.set_text("Training...")
        btn.color = "gray"
        btn.hovercolor = "gray"
        btn.active = False
        normedMileages = normalizeLst(df["mileage"])
        normedPrices = normalizeLst(df["price"])
        train(axs, df, normedMileages, normedPrices)
        show_results(axs, df, btn)

    btn.on_clicked(on_button_clicked)
    plt.show()


if __name__ == "__main__":
    main()
