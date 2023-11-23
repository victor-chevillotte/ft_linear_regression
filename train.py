import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd

theta0 = 0
theta1 = 0
learning_rate = 0.01
epochs = 100
current_epoch = 0


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
        current_epoch = i
        theta0 = theta0 - learning_rate * meanError0(miles, prices, theta0, theta1)
        print(theta0)
        theta1 = theta1 - learning_rate * meanError1(miles, prices, theta0, theta1)

# Normalization (all values between 0 and 1) for calculations
def normalize (lst):
    normalized = []
    for elem in lst:
        normalized.append( (elem - min(lst)) / (max(lst) - min(lst)) )
    return normalized

def main():
    global theta0, theta1, learning_rate
    df = load_data()

    # Création du DataFrame correctement

    # print(df["mileage"])  # Afficher les données de kilométrage

    # Affichage des données
    plt.scatter(df["mileage"], df["price"])
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Price of cars")
    # plt.show()

    # Initialisation des paramètres

    def update(frame):
        global theta0, theta1, current_epoch, epochs
        # display new regression line with new theta0 and theta1
        plt.title(f"Epoch {current_epoch}")
        print("================")
        print(theta0)
        print(theta1)
        min_v = min(df["mileage"])
        max_v = max(df["mileage"])
        line2 = plt.plot(
            [min_v, max_v], [theta0 + theta1 * min_v, theta0 + theta1 * max_v]
        )
        return (scat, line2)

    fig = plt.figure()
    scat = plt.scatter(df["mileage"], df["price"])
    
	# Normalize vectors
    normedMileages = normalize(df["mileage"])
    normedPrices = normalize(df["price"])
    ani = FuncAnimation(fig=fig, func=update, interval=5)
    plt.show()
    
    gradientDescent(epochs, learning_rate, normedMileages, normedPrices)
    print(theta0)
    print(theta1)
    
    # Affichage du modèle ajusté


if __name__ == "__main__":
    main()
