# this code simply plots data points from dataset to get a general ided of data looks like
#use python 3.6 or above 

import numpy as np
import matplotlib.pyplot as plt

list_of_dataset= ['clusterincluster.dat','halfkernel.dat','twogaussians.dat','twospirals.dat']
for x in list_of_dataset:
    fp = open(x)
    fp.readline()
    dataset = np.loadtxt(fp)

    # split data at x and y axis of graph per requirement
    X = dataset[:, 1]
    y  = dataset[:, 2]

    plt.plot(X,y,'ro',color='blue')
    plt.title(str(x))
    plt.show()