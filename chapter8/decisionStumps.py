import time
import numpy as np 

from matplotlib import pyplot as plt 
# decision stump (weak learner)

def loadData():
    '''
    Parameters:
        null
    Return:
        x - feature
        y - label
    '''
    x = np.mat([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    y = np.mat([1.0 ,1.0,-1.0,-1.0,1.0]).transpose()
    return x,y

def showData(x,y):
    '''
    Parameters:
        x - feature
        y - label
    Return:
        null
    '''
    ymax=y.max()
    ymin=y.min()


    for i in range(y.shape[0]):
		if y[i]>0:
			plt.plot(x[i,0],x[i,1],'.',color=(y[i,0]/ymax,0,0),markerSize=10)
		if y[i]<=0:
			plt.plot(x[i,0],x[i,1],'.',color=(0,0,y[i,0]/ymin),markerSize=10)


    # for col,i in zip(['red','blue'],[1,-1]):
    #     data1 = x[np.where(y==i)[0],]
    #     plt.plot(data1[:,0],data1[:,1],'.',color=col,markerSize=10)
class decisonStump:
	def __init__(self,xnD,y,w):
		'''
		Parameters:
			xnD-mat(m,n)
			y-mat(m,1)  -1 or 1
		Return:
			plane={
			t=,# thresh
			s=,# side =-1 or 1
			f=,# feature
			}
		'''
		errBest = 2

		nsample,nfeature = xnD.shape
		for feature in range(nfeature): #feature
			x = xnD[:,feature]
			index = x.A[:,0].argsort(axis=0) #array
			sortx=x[index] 
			sorty=y[index] 

			margin = [sortx[i+1,0]-sortx[i,0] for i in range(nsample-1)]
			clip = [sorty[i+1,0]*sorty[i,0] for i in range(nsample-1)]
			
			betweens = set()
			for i in range(nsample-1):
				if clip[i] ==-1:
					if margin[i]>0:
						betweens.add(i)
					else:
						goodMarginIndex=np.where(np.array(margin)!=0)[0] #the location of margin>0
						try:
							betweens.add(goodMarginIndex[(goodMarginIndex-i)>0].min()) 
						except ValueError as e:
							pass
						try:
							betweens.add(goodMarginIndex[(goodMarginIndex-i)<0].max())
						except ValueError as e:
							pass

			ts = [0.5*(sortx[i+1,0]+sortx[i,0]) for i in betweens]
			ts.append(x.min()-1e-8)
			ss = ['lt','gt']
			for t in ts:
				for s in ss:
					err = self.error(y,self.stumpClassify(xnD,feature,t,s),w)
					if err<errBest:
						errBest=err
						Best = {'threshVal':t,'threshIneq':s,'feature':feature}
		self.feature=Best['feature']
		self.threshVal=Best['threshVal']
		self.threshIneq=Best['threshIneq']
		self.preArray=self.predict(xnD)
		self.err=errBest
		self.xnD=x
		self.y=y
	def predict(self,x):
	    retArray = np.mat(np.ones((x.shape[0],1)))
	    if self.threshIneq == 'lt':
	        retArray[x[:,self.feature]>=self.threshVal] = -1
	    else:
	        retArray[x[:,self.feature]<self.threshVal] = -1
	    self.retArray=retArray
	    return retArray
	def printstump(self):
		print [self.feature,self.threshVal,self.threshIneq,self.err]
	@staticmethod
	def stumpClassify(x,feature,threshVal,threshIneq):
	    '''
	    Parameters:
	        x - feature
	        feature - which feature
	        threshVal - hyperplane
	        threshIneq - label for hyperplane, lower than or greater than is positive
	    Returen:
	        retArray - classification result
	    '''
	    retArray = np.mat(np.ones((x.shape[0],1)))
	    if threshIneq == 'lt':
	        retArray[x[:,feature]>=threshVal] = -1
	    else:
	        retArray[x[:,feature]<threshVal] = -1
	    return retArray
	@staticmethod
   	def error(true,pre,w):
   		err= 1.0*sum(w[np.multiply(true,pre)==-1].transpose())
   		try:
   			err=err[0,0]
   		except TypeError as e:
   			pass
   		return err

class adaBoost:
	'''
	to do : convert buildstumps as a class
	'''
	def __init__(self,x,y):
		nsampel = x.shape[0] #const
		errTol=1e-8
		err=1
		Tmax=100
		T=0 #in the loop

		
		w = np.mat(1.0/y.shape[0]*np.ones(y.shape)) #init W,change in loop
		hs=[]#stumps
		alphas=[]#alphas
		errs=[]
		alphaH=np.mat(np.zeros(y.shape))
		while err>=errTol and T<Tmax:

			h1 = decisonStump(x,y,w)
			# h1.printstump()
			w1=w #update w
			for i in range(nsampel):
				if h1.preArray[i]==y[i]:
					w1[i]=0.5/(1-h1.err)*w[i]
				else:
					w1[i]=0.5/h1.err*w[i]
			w=w1
			try:
				alpha = 0.5*np.log((1-h1.err)/h1.err)
			except ZeroDivisionError:
				hs.append(h1)
				alphas.append(1)
				errs.append(0)
				break
			hs.append(h1)
			alphas.append(alpha)
			alphaH+=np.multiply(alpha,h1.predict(x))
			err=1.0*np.sum(np.multiply(y,np.sign(alphaH))==-1,axis=0)/nsampel
			errs.append(err[0,0])
			T+=1
			if h1.err>=0.5:
			 	break

		self.hs=hs
		self.alphas=alphas
		self.x=x
		self.y=y
		self.errs=errs
	def predict(self,x):
		alphaH=np.mat(np.zeros((x.shape[0],1)))
		for i in range(len(self.hs)):
			alphaH+=np.multiply(self.alphas[i],self.hs[i].predict(x))
		return alphaH
	def showpredict(self):
			lx=np.linspace(self.x[:,0].min(),self.x[:,0].max(),10)
			ly=np.linspace(self.x[:,1].min(),self.x[:,1].max(),10)
			X,Y=np.meshgrid(lx,ly)
			data = zip(X.flatten(),Y.flatten())
			data = np.mat(data)
			showData(data,self.predict(data))
if __name__ == "__main__":
	for i in range(10):
	    x=np.mat(np.random.randn(12,2))
	    y=np.mat(np.sign(np.random.randn(12,1)))
	    plt.subplot(121)
	    showData(x,y)
	    H=adaBoost(x,y)
	    plt.subplot(122)
	    H.showpredict()
	    plt.show()

