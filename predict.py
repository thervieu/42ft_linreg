import csv
import sys
import os.path
import numpy as np

if os.path.isfile('thetas.csv') is False:
    np.savetxt('thetas.csv', np.asarray([0.0, 0.0]))

with open('thetas.csv', newline='') as csvfile:
    thetas = np.array(list((csv.reader(csvfile))))

while True:
    print("Please write the kms ")
    print(">> ", end='')
    txt_input = input()
    if txt_input.isnumeric() is True:
        nb_input = int(txt_input)
        break
    else:
        print("That's not a positive number.")
        continue

print('theta0 = ', float(thetas[0][0]))
print('theta1 = ', float(thetas[1][0]),)
print('nb_input = ', nb_input, '\n')
print('prediction = theta0 + (nb_input*theta1)')
print('prediction = {}'.format(float(thetas[0][0]) + (nb_input*float(thetas[1][0]))))
