import numpy as np

# Normalization (all values between 0 and 1) for calculations
def normalizeLst(lst):
    normalized = []
    for value in lst:
        normalized.append((value - min(lst)) / (max(lst) - min(lst)))
    return normalized

def	normalizeElem(list, elem):
	return ((elem - min(list)) / (max(list) - min(list)))

def	denormalizeElem(list, elem):
	return ((elem * (max(list) - min(list))) + min(list))


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

def computeRMSE(x, y, theta0, theta1):
    return np.sqrt(computeCost(x, y, theta0, theta1))




def computeCost(x, y, theta0, theta1):
    m = len(x)
    total_error = 0.0
    for i in range(m):
        total_error += (theta0 + theta1 * x[i] - y[i]) ** 2
    return total_error / (2 * m)