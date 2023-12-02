import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd

theta0 = 0
theta1 = 0
learning_rate = 0.01
epochs = 5000
current_epoch = 0
display_step = 100


def load_data():
    # Vérifier si le fichier existe
    filename = "data/data.csv"
    if not os.path.exists(filename):
        print("File not found", filename)
        exit()

    # Charger les données avec NumPy
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    if data.size == 0:
        raise ValueError("Data is empty")
    df = pd.DataFrame(data, columns=["mileage", "price"])
    return df


# Goal function
def estimatePrice(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)


# Theta0 formula : sum of errors
def meanError0(x, y, theta0, theta1):
    m = len(x)
    adjustedSum = 0
    for i in range(m):
        adjustedSum += estimatePrice(theta0, theta1, x[i]) - y[i]
    return adjustedSum / m


# Theta1 formula : sum of errors multiplied by x
def meanError1(x, y, theta0, theta1):
    m = len(x)
    adjustedSum = 0
    for i in range(m):
        adjustedSum += (estimatePrice(theta0, theta1, x[i]) - y[i]) * x[i]
    return adjustedSum / m


# The algorithm : iterating towards local minimum
def gradientDescent(epochs, learning_rate, miles, prices):
    global theta0, theta1, current_epoch
    for i in range(epochs):
        current_epoch += 1
        theta0 = theta0 - learning_rate * meanError0(miles, prices, theta0, theta1)
        theta1 = theta1 - learning_rate * meanError1(miles, prices, theta0, theta1)


# Normalization (all values between 0 and 1) for calculations
def normalize(lst):
    normalized = []
    for value in lst:
        normalized.append((value - min(lst)) / (max(lst) - min(lst)))
    return normalized

def denormalize(theta0, theta1, mileage_mean, mileage_std, price_mean, price_std):
    theta1_denorm = theta1 * price_std / mileage_std
    theta0_denorm = theta0 * price_std + price_mean - theta1_denorm * mileage_mean
    return theta0_denorm, theta1_denorm

def main():
    global theta0, theta1, learning_rate
    df = load_data()
    fig, axs= plt.subplots(2, 2, figsize=(10, 10))
    classicPlot = axs[0, 0]
    normedPlot = axs[0, 1]

    classicPlot.scatter(df["mileage"], df["price"])
    classicPlot.set_xlabel("Mileage")
    classicPlot.set_ylabel("Price")
    classicPlot.set_title("Price of cars")

    normedMileages = normalize(df["mileage"])
    normedPrices = normalize(df["price"])

    normedPlot.scatter(normedMileages, normedPrices)
    normedPlot.set_xlabel("Mileage")
    normedPlot.set_ylabel("Price")
    normedPlot.set_title("Price of cars (Normed)")
    plt.show()
    
    def update(frame):
        global theta0, theta1, display_step, current_epoch
        classicPlot.cla()  
        normedPlot.cla() 
        normedPlot.set_title(f"Price of cars (normed) Epoch {current_epoch}")
        normedPlot.set_xlabel("Mileage")
        normedPlot.set_ylabel("Price")
        classicPlot.set_xlabel("Mileage")
        classicPlot.set_ylabel("Price")
        classicPlot.set_title(f"Price of cars Epoch {current_epoch}")
        classicPlot.scatter(df["mileage"], df["price"])
        normedPlot.scatter(normedMileages, normedPrices)
        gradientDescent(display_step, learning_rate, normedMileages, normedPrices)

        # Calculer les moyennes et les écarts-types
        mileage_mean = np.mean(df["mileage"])
        mileage_std = np.std(df["mileage"])
        price_mean = np.mean(df["price"])
        price_std = np.std(df["price"])

        # Calculer les valeurs dénormalisées
        theta0_denorm, theta1_denorm = denormalize(
            theta0, theta1, mileage_mean, mileage_std, price_mean, price_std
        )

        # Utiliser les valeurs dénormalisées pour tracer la droite

        min_mileage = min(df["mileage"])
        max_mileage = max(df["mileage"])
        classicPlot.plot(
            [min_mileage, max_mileage],
            [theta0_denorm + theta1_denorm * min_mileage, theta0_denorm + theta1_denorm * max_mileage],
            color="red",
        )
        min_normed_mileage = min(normedMileages)
        max_normed_mileage = max(normedMileages)

        normedPlot.plot(
            [min_normed_mileage, max_normed_mileage],
            [theta0 + theta1 * min_normed_mileage, theta0 + theta1 * max_normed_mileage],
            color="red",
        )

    fig, axs= plt.subplots(2, 2, figsize=(10, 10))
    classicPlot = axs[0, 0]
    normedPlot = axs[0, 1]
    ani = FuncAnimation(fig=fig, func=update, frames=int(epochs / display_step), interval=20, repeat=False)
    plt.show()

    print("theta finaux")
    print(theta0)
    print(theta1)
    theta0_denorm, theta1_denorm = denormalize(
            theta0,
            theta1,
            min(df["mileage"]),
            max(df["mileage"]),
            min(df["price"]),
            max(df["price"]),
        )
    print("theta denorm")
    print(theta0_denorm)
    print(theta1_denorm)



if __name__ == "__main__":
    main()
