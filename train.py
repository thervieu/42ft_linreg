import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(x, theta0, theta1):
    return theta0 + (theta1 * x)

def error(y_hat, y):
	return y_hat - y

def train(x, y, theta0, theta1):
	pred = predict(x, theta0, theta1)
	errors = error(pred, y)
	rtn0 = 0.01 * np.sum(errors) / len(errors)
	rtn1 = 0.01 * np.sum(errors * x) / len(errors)
	return (theta0 - rtn0, theta1 - rtn1)

def mse_(x, y, theta0, theta1):
    pred = predict(x, theta0, theta1)
    return np.sum(error(pred, y)**2) / len(y)

#get data
data = pd.read_csv('data.csv')
kms = data['km']
prices = data['price']

#normalize
max_km = kms.max()
max_price = prices.max()
kms = kms / max_km
prices = prices / max_price

#train
theta0 = 0.0
theta1 = 0.0
mses = []
for epoch in range(10000):
    theta0, theta1 = train(kms, prices, theta0, theta1)
    mse = mse_(kms, prices, theta0, theta1)
    mses.append(mse)
    if epoch % 1000 == 0:
        print('epoch {:04d}: avg_error: {}'.format(epoch, mse))

theta0 = theta0 * max_price
theta1 = theta1 * (max_price / max_km)

print('\nnew thetas are\n{}\n{}'.format(theta0, theta1))
np.savetxt('thetas.csv', [theta0, theta1])
plt.plot(mses)
plt.show()