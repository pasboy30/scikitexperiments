from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import calinski_harabaz_score

dataset_list = ['twogaussians.dat', 'twospirals.dat', 'halfkernel.dat','clusterincluster.dat']
scores_k = []
scores_em= []
for x in dataset_list:
    fp = open(x)
    fp.readline()
    dataset = np.loadtxt(fp)
    X = dataset[:, 1:]
    y = dataset[:, 0]  # Actual labels
    for K in range(2,11):
        temp_k = []
        k_means = KMeans(n_clusters=K).fit(X)
        labels_k_means = k_means.predict(X)
        em = GaussianMixture(n_components=K).fit(X)
        labels_em = em.predict(X)
        k_means_score = calinski_harabaz_score(X,labels_k_means)
        em_score = calinski_harabaz_score(X,labels_em)
        temp_k.append(K)
        temp_k.append(k_means_score)
        scores_k.append(temp_k)
        temp_k = []
        temp_k.append(K)
        temp_k.append(em_score)
        scores_em.append(temp_k)

    print("Calinski Harbaz score for the dataset " + x + " for K - Means")
    for z in scores_k:
        print(str(z))
    print("Calinski Harbaz score for the dataset " + x + " for EM")
    for z in scores_em:
        print(str(z))


