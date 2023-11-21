import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
	# Load data
	
	try :
		# Check if file exists 
		with open('data/data.csv') as f:
			pass
		# Check if file format
		if not 'csv' in 'data/data.csv':
			raise IOError('File is not csv')
	except IOError as e:
		print('File not found data/data.csv')
		exit()

	data = pd.read_csv('data/data.csv', header=None)
	data = data.values
	print(data)
	return data

def main():
	data = load_data()
	plt.scatter(data[:,0], data[:,1])
	plt.show()

if __name__ == '__main__':
	main()