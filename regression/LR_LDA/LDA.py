#encode=utf-8
'''
show that \beta in proportional to cov^{-1}*(\mu_1-\mu_2)
'''
import matplotlib.pyplot as plt
import numpy as np 
from functionplot import fcontour
## generate data for 2 classes
x=np.zeros((0,2))#0
y=np.zeros((0,1))#1
plt.figure(1)




N=[100,5000]
codeY=[0,1]
i=0
for mu in [-10,10]:
    mu1=[mu,mu]
    sigma1=[[10,0],[0,3]]
    R = np.random.multivariate_normal(mu1,sigma1,N[i])
    plt.plot(R[:,0],R[:,1],'+')
    plt.xlim(-17,17)
    plt.ylim(-17,17)
    x = np.vstack((x,R))
    y0=np.zeros((N[i],1))

    y0[:,]=codeY[i]
    y = np.vstack((y,y0))
    i+=1


## regression

x_regress = np.hstack((np.ones((x.shape[0],1)),x))
b=np.dot(np.linalg.pinv(x_regress),y)
def lr(x,y,b):
    xt=x.reshape(-1,1)
    yt=y.reshape(-1,1)
    X = np.hstack((np.ones((xt.shape[0],1)),xt,yt))
    return np.dot(X,b).reshape(x.shape)
fcontour(lambda x,y:lr(x,y,b),[-17,17])


##LDA
m1 = x[:N[0],].mean(axis=0)
m2 = x[N[0]:,].mean(axis=0)


s1 = np.dot((x[:100,]-x[:100,].mean(axis=0)).T,(x[:100,]-x[:100,].mean(axis=0)))
s2 = np.dot((x[100:,]-x[100:,].mean(axis=0)).T,(x[100:,]-x[100:,].mean(axis=0)))
s = (s1+s2)/(sum(N)-2)

d=np.dot(np.linalg.inv(s),m1-m2)


## they are identical 
print d
print b 
print 1/(b[1:,0].T/d)

import pdb; pdb.set_trace()  # breakpoint ae83f1d9 //

plt.show()




