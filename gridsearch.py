import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


data_set = ['twospirals.dat','twogaussians.dat','clusterincluster.dat','halfkernel.dat']

for data in data_set:

    fp = open(data)
    fp.readline()
    dataset = np.loadtxt(fp)
    X = dataset[:, 1:]
    y = dataset[:, 0]

    c_range = range(1,6)
    gamma_range = [0.1,0.2,0.3,0.4,0.5]

    parameters = dict(C = c_range,gamma=gamma_range)

    svc = svm.SVC()
    grid = GridSearchCV(svc,parameters,cv=10,scoring='accuracy')
    grid.fit(X,y)
    dict_nos ={}
    mean_scores =grid.cv_results_['mean_test_score']
    threshold = 0
    print("---------------------------------")
    print("*** GRID SEARCH RESULTS for " + data +" ***")
    print("-> They are in the ordered pair (x,y) where \n - x is value of C \n - y is value for gamma")
    print("----------------------------------")
    for x in c_range:
        for y in gamma_range:
            dict_nos[threshold] = "(" + str(x) + "," + str(y) +")"
            print("(" + str(x) + "," + str(y) +") =" + str(mean_scores[threshold]))
            threshold += 1
            if(threshold == 25):
                break;
    print("----------------------------------")
    print("--> best value obtained from grid search is ... ")
    threshold = 0
    max = np.amax(mean_scores)
    print(max)
    for x in mean_scores:
        threshold += 1
        if x == max:
            print ("Available for following (C,gamma) " + dict_nos[threshold-1])