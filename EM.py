from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as pl

dataset_list = ['clusterincluster.dat', 'twogaussians.dat','twospirals.dat','halfkernel.dat']

for x in dataset_list:
    fp = open(x)
    fp.readline()
    dataset = np.loadtxt(fp)
    X = dataset[:, 1:]
    y = dataset[:, 0] #Actual labels

    clustering_EM = GaussianMixture(n_components=2).fit(X)
    labels = clustering_EM.predict(X) #Predicted labels
    for a, b, c, d in zip(X[:, 0], X[:, 1], labels, y):
        if c == 1:
            if d == 1:
                pl.scatter(a, b, facecolors='none', edgecolors='#ab3533')
            else:
                pl.scatter(a, b, facecolors='none', edgecolors='#729da8')
        else:
            if d == 1:
                pl.scatter(a, b, c='#ab3533', marker='+')
            else:
                pl.scatter(a, b, c='#729da8', marker='+')

    pl.title("EM clustering for dataset " + x + "with K = 2")
    pl.show()