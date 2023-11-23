import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
	print("theta0", theta0)
	print("theta1", theta1)
	print("mileage", mileage)
	return (theta0 + theta1 * mileage)


def calculate_theta0(mileage, price):
	m = mileage.size
	print("size", m)
	adjustedSum = 0
	for i in range(m):
		print("************")
		print(estimate_price(mileage[i]))
		print(mileage[i])
		print(price[i])
		adjustedSum += estimate_price(mileage[i]) - price[i]
	print("adjustedSum", adjustedSum)
	return adjustedSum / m


def calculate_theta1(mileage, price):
	m = mileage.size
	adjustedSum = 0
	for i in range(m):
		adjustedSum += (estimate_price(mileage[i]) - price[i]) * mileage[i]
	return adjustedSum / m


def cost_function(m, mileage, price):
	return (1 / (2 * m)) * sum((estimate_price(mileage) - price) ** 2)


def main():
	global theta0, theta1, learning_rate
	df = load_data()

	# Création du DataFrame correctement

	print(df["mileage"])  # Afficher les données de kilométrage

	# Affichage des données
	plt.scatter(df["mileage"], df["price"])
	plt.xlabel("Mileage")
	plt.ylabel("Price")
	plt.title("Price of cars")
	#plt.show()

	# Initialisation des paramètres


	
	def update(frame):
		# display new regression line with new theta0 and theta1
		plt.title(f"Epoch {frame+1}")
		#print("================")
		min_v = min(df["mileage"])
		max_v = max(df["mileage"])
		#print("***********")
		#print(theta0)
		#print(theta1)
		#print(theta0 + theta1 * min_v)
		#print(theta0 + theta1 * max_v)
		#print("------------")
		line2 = plt.plot(
			[min_v, max_v], [theta0 + theta1 * min_v, theta0 + theta1 * max_v]
		)
		return (scat, line2)


	fig = plt.figure()
	scat = plt.scatter(df["mileage"], df["price"])
	#ani = FuncAnimation(fig=fig, func=update, interval=5)
	#plt.show()

	epochs = 1
	m = len(df["mileage"])
	for epoch in range(epochs):
		h = estimate_price(df["mileage"])
		cost = cost_function(m, h, df["price"])

		for i in range(20):
			res = calculate_theta0(df["mileage"], df["price"])
			print(res)
			theta0 = theta0 - learning_rate * calculate_theta0(df["mileage"], df["price"])
			theta1 = theta1 - learning_rate * calculate_theta1(df["mileage"], df["price"])

		if epoch % 100 == 0:
			print(f"Epoch {epoch}, Cost: {cost}")


	

	# Affichage du modèle ajusté


if __name__ == "__main__":
	main()
