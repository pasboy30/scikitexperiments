import numpy as np
from scipy import interp
import matplotlib.pyplot as pl
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold


data_set = ['twospirals.dat','twogaussians.dat','clusterincluster.dat','halfkernel.dat']

for data_sel in data_set:
    print('--------------------------------------------------------')
    print('Dataset :' + data_sel)
    print('--------------------------------------------------------')
    #import some data-set to play with
    fp = open(data_sel)
    fp.readline()
    dataset = np.loadtxt(fp)

    X = dataset[:, 1:]
    y = dataset[:, 0]

    kf = KFold(n_splits=10, shuffle=True)
    classifier = svm.SVC(kernel='rbf',probability=True , C = 1.0 , gamma='auto')
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    count = 0
    data_plot_list = []
    lis_of_colors = ['#02ff7c','#b7ff01','#fff700','#ffb300','#ff4d00', '#fff308', '#ffcd94', '#ffd0fa', '#fff89d', '#ffdde2']
    for train_index, test_index in kf.split(X):
        count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        probas_= classifier.fit(X_train, y_train)
        probas_ = probas_.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1],pos_label=2)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        print("AUC for fold no " + str(count) + " is " + str(roc_auc))
        temp_list = []
        temp_list.append(fpr)
        temp_list.append(tpr)
        temp_list.append(roc_auc)
        data_plot_list.append(temp_list)
    fold_count = 0
    for x in data_plot_list:
        fold_count += 1
        pl.plot(x[0], x[1], label='ROC curve for fold count ' + str(fold_count) + ' (area = %0.4f)' % x[2], c = lis_of_colors[fold_count-1])
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('FP Rate')
        pl.ylabel('TP Rate')
        pl.title('ROC Curve for dataset ' + data_sel)
        pl.legend(loc="lower right")
        if fold_count == 9:
            pl.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
    pl.show()