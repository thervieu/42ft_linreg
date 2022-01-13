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

if float(thetas[0][0]) == 0.0:
    print('please train first')
    sys.exit()

data = pd.read_csv('data.csv')
kms = data['km']
prices = data['price']

thetas = thetas.astype('float64')
x = np.linspace(int(np.floor(kms.min())), int(np.floor(kms.max())))
y = float(thetas[0][0]) + (x.astype('float64') * float(thetas[1][0]))

plt.scatter(kms, prices, c='b')
plt.plot(x, y, c='orange')

plt.legend(['data_set', 'lin_reg'])
plt.title('Graph of the lin_reg created with the data_set')
plt.xlabel("kms")
plt.ylabel("price")

plt.show()
