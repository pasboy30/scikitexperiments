#This code use Naive Bayes for classification 
#Use python 3.6 or above 

import numpy as np
import pylab as pl
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

list_of_datasets = ['clusterincluster.dat', 'halfkernel.dat', 'twogaussians.dat', 'twospirals.dat']

for ds in list_of_datasets:
    # import some data to play with
    fp = open(ds)
    fp.readline()
    dataset = np.loadtxt(fp)

    X = dataset[:, 1:]
    y = dataset[:, 0]
    h = .02  # step size in the mesh
    cmap_light = ListedColormap(['#FCCC92','#AAFFB6'])
    cmap_bold = ListedColormap(['#F98902',  '#04E824'])

    kf = KFold(n_splits=10, shuffle=True)
    navieB = GaussianNB()
    count = 1
    final_results = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Initiating classification
        navieB.fit(X_train, y_train)
        P = navieB.predict(X_test)
        results = confusion_matrix(np.array(y_test), np.array(P))
        final_results += results
        # print ("Correctness in  prediction:", np.array(y_test) == np.array(P))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = navieB.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure()
        pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

        pl.xlim(xx.min(), xx.max())
        pl.ylim(yy.min(), yy.max())
        pl.title("Naive Bayesian Classification for 2 classes \n Dataset: " + str(ds) + "\n Split number:" + str(count))
        # pl.show()
        pl.savefig("BAYES" + str(ds) + "split" + str(count) + '.png')
        count += 1
    print ("Confusion matrix (Aggregated) for Data-set:" + str(ds) + " is: \n" + str(final_results))
    TP = final_results[0][0]
    TN = final_results[1][1]
    FP = final_results[1][0]
    FN = final_results[0][1]
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    specificity = TN/(TN+FP)
    sensitivity = TP/(TP+FN)
    if(ds == 'twogaussians.dat'):
        accuracy = ((TP+TN)/400)*100
    else:
        accuracy = ((TP + TN) / 1000)*100
    print("PPV : "+ str(PPV))
    print("NPV : " + str(NPV))
    print("specificity : " + str(specificity))
    print("sensitivity : " + str(sensitivity))
    print("accuracy : " + str(accuracy)+"%")