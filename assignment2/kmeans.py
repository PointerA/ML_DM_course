import random
import math
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

class kmeans():
    def __init__(self, k, init_func):
        assert init_func == "random" or init_func == "distance" or init_func == "plus"
        self.k = k
        self.init_func = init_func
        self.centers = None

    def fit(self, X, y, max_epochs=100, tol=1e-3):
        self.centers = eval("generate_" + self.init_func)(X, self.k)
        last_clusters = None
        last_dist = None
        sum_dists, nmis = [], []

        for epoch in range(max_epochs):
            pred = []
            sum_dist = 0.
            clusters = [[] for _ in range(self.k)]
            #update the label of each point
            for i in range(X.shape[0]):
                idx, dist = nearest_dist(X[i], self.centers)
                clusters[idx].append(i)
                sum_dist += dist
                pred.append(idx)
            #update the label of center points
            for i in range(self.k):
                if clusters[i]:
                    self.centers[i] = np.mean(X[clusters[i]], axis=0)
            
            nmi = normalized_mutual_info_score(y, pred)
            sum_dists.append(sum_dist)
            nmis.append(nmi)

            if (last_clusters is not None and last_clusters == clusters) or\
               (last_dist is not None and math.fabs(sum_dist - last_dist) <= tol):
                sum_dists.append(sum_dist)
                nmis.append(nmi)
                print("epoch %d : sum_dist : %f , nmi : %f, k : %d"
                  % (epoch, sum_dist, nmi, self.k))
                break
            else:
                last_clusters, last_dist = clusters, sum_dist
        return sum_dists, nmis
   

def generate_random(X, k):
    idex = list(range(X.shape[0]))
    idxs = np.random.choice(idex,k,replace=False)
    return X[idxs]

def generate_distance(X, k):
    random_idx = np.random.randint(X.shape[0])
    centers = [X[random_idx]]
    d = np.zeros(X.shape[0])

    for pos in range(1, k):
        for i in range(X.shape[0]):
            _, d[i] = nearest_dist(X[i], centers[:pos])
        newcenter = np.argmax(d)
        centers.append(X[newcenter])
    return centers

def generate_plus(X, k):
    random_idx = np.random.randint(X.shape[0])
    centers = [X[random_idx]]
    d = np.zeros(X.shape[0])

    for pos in range(1, k):
        for i in range(X.shape[0]):
            _, d[i] = nearest_dist(X[i], centers[:pos])
        newcenter = np.random.choice(np.where(d>np.quantile(d,0.75,interpolation='higher'))[0])
        centers.append(X[newcenter])
    return centers


def nearest_dist(x, centers):
    minDist, idx = math.inf, -1
    for i, center in enumerate(centers):
        curDist = dist(x, center)
        if curDist < minDist:
            minDist = curDist
            idx = i
    return idx, minDist


def dist(point1, point2):
    return np.sum(np.square(point1 - point2))

