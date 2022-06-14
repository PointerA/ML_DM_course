# coding=utf-8
import heapq
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import copy
import matplotlib.pyplot as plt

class cf:
    def __init__(self, k, data, test):
        self.k = k
        self.data = data
        self.test = test
        self.pridicts = copy.deepcopy(test)

    def count_sim(self, i_, j_):
        r,c = self.data.shape
        sims = np.zeros(c)
        #print(np.count_nonzero(self.data, axis=0)[j_])
        for j in range(c):
            if j == j_ or self.data[i_][j]==0:
                sims[j] = -1
                continue
            #皮尔逊相关系数
            fenzi = 0
            fenmu1 = 0
            fenmu2 = 0
            count1 = np.count_nonzero(self.data[:,j_])
            count2 = np.count_nonzero(self.data[:,j])
            if(count2 == 0 or count1 == 0):
                sims[j] = -1
                continue
            tempj_ = self.data[:,j_]-np.mean(self.data[:,j_])*r/count1
            tempj = self.data[:,j]-np.mean(self.data[:,j])*r/count2
            tempj_ = tempj_.reshape(-1,1)
            tempj = tempj.reshape(-1,1)
            #print(tempj.shape)
            for i in range(r):
                if(self.data[i][j] and self.data[i][j_]):
                    fenzi += tempj_[i][0]*tempj[i][0]
                    fenmu1 += np.square(tempj_[i][0])
                    fenmu2 += np.square(tempj[i][0])
            if(fenmu1==0 or fenmu2==0):
                sims[j] = -1
                continue
            sims[j] = fenzi/((np.sqrt(fenmu1))*(np.sqrt(fenmu2)))
        return sims

    def item_base(self):
        r,c = self.pridicts.shape
        for j in range(c):
            for i in range(r):
                if(self.pridicts[i][j]):
                    sims = self.count_sim(i,j)
                    kSimIndex = heapq.nlargest(self.k, range(len(sims)), sims.take)
                    fenzi = 0
                    fenmu = 0
                    for similar in kSimIndex:
                        if(sims[similar] > 0):
                            fenzi += self.data[i][similar] * sims[similar]
                            fenmu += sims[similar]
                    if(fenmu == 0):
                        self.pridicts[i][j] = 3
                    else:
                        self.pridicts[i][j] = fenzi/fenmu
                        '''
                        if(self.pridicts[i][j] - int(self.pridicts[i][j])>=0.5):
                            self.pridicts[i][j] = int(self.pridicts[i][j]) + 1
                        else:
                            self.pridicts[i][j] = int(self.pridicts[i][j])
                    if(self.pridicts[i][j] > 5):
                        self.pridicts[i][j] = 5
                    elif(self.pridicts[i][j] < 1):
                        self.pridicts[i][j] = 1
                        '''
    
    def rmse(self):
        r,c = self.pridicts.shape
        count = np.count_nonzero(self.pridicts)
        return np.sqrt(r*c*mean_squared_error(self.pridicts, self.test)/count)
    
def readData(file, proportion=0.8):
    df=pd.read_csv(file)
    userNum = df['userId'].max()
    movieNum = df['movieId'].max()
    matrix = [[0]*movieNum for _ in range(userNum)]
    dfRow = df.shape[0]
    for i in range(dfRow):
        matrix[df['userId'][i]-1][df['movieId'][i]-1] = df['rating'][i]
    trainUserNum = int(userNum * proportion)
    trianMovieNum = int(movieNum * proportion)
    matrix = np.array(matrix)
    testMatrix = matrix[trainUserNum:,trianMovieNum:]
    testMatrix = copy.deepcopy(testMatrix)
    for i in range(trainUserNum, userNum):
        for j in range(trianMovieNum, movieNum):
            matrix[i][j] = 0
    #print(np.sum(matrix))
    return np.array(matrix), np.array(testMatrix)


data, test = readData("./dataset/ratings2.csv")
print("read data already")
ks = range(1,21)
rmses = []
min_rmse = 9999
min_k = 0
for k in ks:
    item_base_cf = cf(k, data, test)
    item_base_cf.item_base()
    rmse = item_base_cf.rmse()
    rmses.append(rmse)
    if(rmse < min_rmse):
        min_rmse = rmse
        min_k = k

plt.plot(ks, rmses)
plt.xlabel("k")
plt.ylabel("RMSE")
plt.savefig("res4.png", format='png') 
plt.close()
print("the best k value is ",min_k)
print("the minimum rmse is ",min_rmse)
print("finished...")