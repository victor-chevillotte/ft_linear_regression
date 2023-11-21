
def estimate_price(mileage, theta0, theta1):
	return theta0 + theta1 * mileage

def calculate_theta0(learning_rate, m, mileage, price):
	return learning_rate * (1/m) * estimate_price(mileage) - price

def calculate_theta1(learning_rate, m, mileage, price):
	return learning_rate * (1/m) * (estimate_price(mileage) - price) * mileage

def cost_function(m, mileage, price):
	return (1/(2*m)) * sum((estimate_price(mileage) - price)**2)