import numpy as np
from numpy.core.multiarray import ndarray


def loadData():
    x = np.array(range(10)) + 1
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80,
                  7.05, 8.90, 8.70, 9.00, 9.05])
    return x,y

class simpltTree:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.getSplit()
    def getSplit(self):
        bestErr = np.inf
        for split in self.x[:-1]:
            err = self.error(split)
            if err < bestErr:
                bestErr = err
                bestSplit = split
        self.split = bestSplit
        self.err = bestErr
    def error(self,s):
        y1=self.y[np.where(self.x<=s)]
        y2=self.y[np.where(self.x>s)]
        return y1.var()*y1.shape[0]+y2.var()*y2.shape[0]
    def predict(self,x):
        y1=self.y[np.where(self.x<=self.split)]
        y2=self.y[np.where(self.x>self.split)]
        return np.where(x<=self.split,y1.mean(),y2.mean())

class Tree:
    def __init__(self, x, y,n):
        f=np.zeros(y.shape)
        residual = y-f
        trees=[]
        for i in range(n):
            TS = simpltTree(x,residual)
            f += TS.predict(x)
            residual = y - f
            trees.append(TS)
        self.trees = trees
        self.err = residual.var()*residual.shape[0]

if __name__ == '__main__':
    x,y = loadData()
    T = Tree(x,y,6)