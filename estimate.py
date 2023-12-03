import pandas as pd
import matplotlib.pyplot as plt
from messages import (
	debug,
	title,
	normal,
	success,
	verbose,
	error
)

def get_thetas ():
	filename = "data/results.csv"

	try :
		df = pd.read_csv(filename)
	except FileNotFoundError:
		error("File with thetas not found. Please run train.py first.")
		normal("Price estimation is not possible without thetas. Price Car : 0$")
		title("Exiting...")
		exit()
	#return theta0, theta1
	return df["theta0"][0], df["theta1"][0]


def estimate():
	#load thetas
	theta0, theta1 = get_thetas()
	#ask user for mileage
	mileage = input("Enter mileage : ")
	if (not mileage.isdigit()):
		error("Mileage must be a positive number")
		exit()
	print("\nBased on current predictions, a car with a mileage of", mileage, "kilometers would be worth :")
	result = theta0 + (theta1 * float(mileage))
	if (result < 0):
		print("0$")
	else:
		print(f"{result:.2f}$")


if __name__ == "__main__":
    estimate()
