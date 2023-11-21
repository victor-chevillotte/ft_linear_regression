import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

theta0 = 0
theta1 = 0
learning_rate = 0.01

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


def estimate_price(mileage):
	global theta0, theta1
	return theta0 + theta1 * mileage


def calculate_theta0(learning_rate, m, mileage, price):
	return learning_rate * (1 / m) * estimate_price(mileage) - price


def calculate_theta1(learning_rate, m, mileage, price):
	return learning_rate * (1 / m) * (estimate_price(mileage) - price) * mileage


def cost_function(m, mileage, price):
	return (1 / (2 * m)) * sum((estimate_price(mileage) - price) ** 2)


def main():
	df = load_data()

	# Création du DataFrame correctement

	print(df["mileage"])  # Afficher les données de kilométrage

	# Affichage des données
	plt.scatter(df["mileage"], df["price"])
	plt.xlabel("Mileage")
	plt.ylabel("Price")
	plt.title("Price of cars")
	plt.show()

	# Initialisation des paramètres
   
	epochs = 1000
	m = len(df["mileage"])
	
	plt.scatter(df["mileage"], df["price"])

	for epoch in range(epochs):
		h = estimate_price(df["mileage"])
		cost = cost_function(m, h, df["price"])

		theta0 = calculate_theta0(learning_rate, m, df["mileage"], df["price"])
		theta1 = calculate_theta1(learning_rate, m, df["mileage"], df["price"])

		if epoch % 100 == 0:
			print(f"Epoch {epoch}, Cost: {cost}")
		#display_plot(df, theta0, theta1)
		plt.plot(df["mileage"], h, color="red")
		plt.pause(0.01)
		plt.clf()
		plt.scatter(df["mileage"], df["price"])
		plt.xlabel("Mileage")
		plt.ylabel("Price")
		plt.title("Price of cars")
		plt.show()


	# Affichage du modèle ajusté
	
	
	


if __name__ == "__main__":
	main()
