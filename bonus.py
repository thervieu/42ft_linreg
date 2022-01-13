import sys
import csv
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.path.isfile('thetas.csv') is False:
    print('please train first')
    sys.exit()

with open('thetas.csv', newline='') as csvfile:
    thetas = np.array(list((csv.reader(csvfile)))).reshape(-1,1)

data = pd.read_csv('data.csv')
kms = data['km']
prices = data['price']


def predict(x, theta0, theta1):
    return theta0 + x * theta1

thetas = thetas.astype('float64')
x = np.linspace(int(np.floor(kms.min())), int(np.floor(kms.max())))

y = thetas[0] + (x.astype('float64') * thetas[1])
plt.scatter(kms, prices, c='b')
plt.plot(x, y, c='orange')

plt.show()
