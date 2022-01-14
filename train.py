import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(x, theta0, theta1):
    return theta0 + (theta1 * x)

def error(y_hat, y):
	return y_hat - y

def cost(x, theta0, theta1, y):
	y_hat = predict(x, theta0, theta1)
	return (1/(2*len(y))) * (np.sum(y_hat - y)**2)

def train(x, y, theta0, theta1):
	y_hat = predict(x, theta0, theta1)
	errors = error(y_hat, y)
	rtn0 = 0.1 * np.sum(errors) / len(errors)
	rtn1 = 0.1 * np.sum(errors * x) / len(errors)
	return (theta0 - rtn0, theta1 - rtn1)

def mse_(x, y, theta0, theta1):
    y_hat = predict(x, theta0, theta1)
    return np.sum(error(y_hat, y)**2) / len(y)

if os.path.isfile('data.csv') is False:
    print('please add data.csv file')
    sys.exit()


# get data
data = pd.read_csv('data.csv')
kms = data['km']
prices = data['price']

# normalize
max_km = kms.max()
max_price = prices.max()
kms = kms / max_km
prices = prices / max_price

# train
theta0 = 0.0
theta1 = 0.0
mses = []
theta0train = []
theta1train = []
for epoch in range(1001):
    theta0, theta1 = train(kms, prices, theta0, theta1)
    theta0train.append(theta0)
    theta1train.append(theta1)
    mse = mse_(kms, prices, theta0, theta1)
    mses.append(mse)
    if epoch % 100 == 0:
        print('epoch {:04d}: mean_squared_error: {}'.format(epoch, mse))
        print('theta0 = |{}|\ntheta1 = |{}|\n'.format(theta0, theta1))

theta0 = theta0 * max_price
theta1 = theta1 * (max_price / max_km)

# print and plot
print('\nnew thetas are\n{}\n{}'.format(theta0, theta1))
np.savetxt('thetas.csv', [theta0, theta1])
plt.plot(mses)

plt.legend(['mse'])
plt.title('Graph of the mse depending on the epochs')
plt.xlabel("epochs")
plt.ylabel("mean_squared_errors")

plt.figure()

costs = []
theta1true = -000.6
theta0s = np.linspace(-0.5, 2.6, 1000)
for theta in theta0s:
    costs.append(cost(kms, theta, theta1true, prices))
costtrain = []
for theta in theta0train:
    costtrain.append(cost(kms, theta, theta1true, prices))
plt.plot(theta0s, costs)
plt.scatter(theta0train, costtrain, c='orange')

plt.figure()

costs = []
theta0true = 1.017
theta1s = np.linspace(-1.7, 0.5, 1000)
for theta in theta1s:
    costs.append(cost(kms, theta0true, theta, prices))
costtrain = []
for theta in theta1train:
    costtrain.append(cost(kms, theta0true, theta, prices))
plt.plot(theta1s, costs)
plt.scatter(theta1train, costtrain, c='orange')

plt.show()
