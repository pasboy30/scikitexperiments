from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as pl

dataset_list = ['twogaussians.dat', 'clusterincluster.dat','twospirals.dat','halfkernel.dat']

for x in dataset_list:
    fp = open(x)
    fp.readline()
    dataset = np.loadtxt(fp)
    X = dataset[:, 1:]
    y = dataset[:, 0] #Actual labels

    clustering = KMeans(n_clusters=2).fit(X)
    labels = clustering.predict(X) #Predicted labels
    centroids = np.array(clustering.cluster_centers_)
    for a,b,c,d in zip(X[:,0], X[:,1], labels,y):
        if c == 1:
            if d ==1:
                pl.scatter(a,b,facecolors='none',edgecolors='#ab3533')
            else:
                pl.scatter(a,b,facecolors='none',edgecolors='#729da8')
        else:
            if d == 1:
                pl.scatter(a,b,c='#ab3533',marker='+')
            else:
                pl.scatter(a,b,c='#729da8',marker='+')

    pl.scatter(centroids[:,0],centroids[:,1] ,marker='x', c='#030303' )
    pl.title("K-Means clustering for dataset " + x)
    pl.show()
