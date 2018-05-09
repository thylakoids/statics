import numpy as np 

from matplotlib import pyplot as plt 

def loadData():
    '''
    Parameters:
        null
    Return:
        x - feature
        y - label
    '''
    x = np.array([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    y = np.array([1.0 ,1.0,-1.0,-1.0,1.0])[:,np.newaxis]
    return x,y

def showData(x,y):
    '''
    Parameters:
        x - feature
        y - label
    Return:
        null
    '''
    for col,i in zip(['red','blue'],[1,-1]):
        data1 = x[y[:,0]==i,]
        plt.plot(data1[:,0],data1[:,1],'.',color=col,markerSize=10)
    plt.show()
def showStump(x,y,dim,threshVal):
    for col,i in zip(['red','blue'],[1,-1]):
        data1 = x[y[:,0]==i,]
        plt.plot(data1[:,0],data1[:,1],'.',color=col,markerSize=10)
    dims = 1-dim
    mindims=x[:,dims].min();maxdims=x[:,dims].max()
    mindim=x[:,dim].min();maxdim=x[:,dim].max()
    if dims==0:
    	plt.plot([mindims,maxdims],[threshVal,threshVal],'-')
    	mindims=x[:,dims].min();maxdims=x[:,dims].max()
    else:
    	plt.plot([threshVal,threshVal],[mindims,maxdims],'-')
    plt.show()
def stumpClassify(x,dim,threshVal,threshIneq):
    '''
    Parameters:
        x - feature
        dim - which dimention
        threshVal - hyperplane
        threshIneq - label for hyperplane, lower than or greater than is positive
    Returen:
        retArray - classification result
    '''
    retArray = np.ones((x.shape[0],1))
    if threshIneq == 'lt':
        retArray[x[:,dim]>=threshVal] = -1
    else:
        retArray[x[:,dim]<threshVal] = -1
    return retArray
def error(true,predict,w):
	return 1.0*sum(w[true*predict==-1])
def buildStump(x,y,w):
	epsilon = 1e-10
	'''
	Parameters:
		x - feature
		y - label
	Teturn:
		dim - which dimentation
		threshVal - where
		threshIneq - which side
	'''
	errBest = 1
	dimBest = 0
	threshValBest =0
	threshIneqBest = 'lt'
	nsampel,nfeature = x.shape
	for dim in range(nfeature): #dim

		index = x[:,dim].argsort(axis=0)
		sorty = y[index]
		sortx = x[index,dim]
		sortyPlus = np.concatenate([sorty[1:,:],np.ones((1,1))],axis=0)
		betweens = np.where((sorty*sortyPlus)[:-1,]==-1)[0]
		threshVals = [np.mean(sortx[[between,between+1]]) for between in betweens]
		threshVals.append(sortx.min()-epsilon)
		for threshVal in threshVals: #threshVal
			for threshIneq in ['lt','gt']: #threshIneq
				err = error(y,stumpClassify(x,dim,threshVal,threshIneq),w)
				if err<errBest:
					errBest = err
					dimBest = dim
					threshValBest = threshVal
					threshIneqBest =threshIneq
	return dimBest,threshValBest,threshIneqBest,errBest
def adaBoost(x,y):
	'''
	to do : convert buildstumps as a class
	'''
	w=1.0/y.shape[0]*np.ones(y.shape)

if __name__ == "__main__":
    x,y = loadData()
    dim,threshVal,threshIneq,errBest=buildStump(x,y,1.0/y.shape[0]*np.ones(y.shape))
    print dim,threshVal,threshIneq
    showStump(x,y,dim,threshVal)
