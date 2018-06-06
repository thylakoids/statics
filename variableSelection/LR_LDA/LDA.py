'''
show that \beta in proportional to cov^{-1}*(\mu_1-\mu_2)
i.e. Σ̂ B β is in the direction (μ̂ 2 − μ̂ 1 )
'''

import numpy as np 

## generate data for 2 classes
x=np.zeros((0,2))
y=np.zeros((0,2))


i=0
for mu in [-10,10]:
    mu1=[mu,mu]
    sigma1=[[4,0],[0,3]]
    R = np.random.multivariate_normal(mu1,sigma1,100)
    x = np.vstack((x,R))
    y0=np.zeros((100,2))
    y0[:,i]=1
    y = np.vstack((y,y0))
    i+=1

## regression

x_regress = np.hstack((np.ones((x.shape[0],1)),x))
b=np.dot(np.linalg.pinv(x_regress),y)



##LDA
m1 = x[:100,].mean(axis=0)
m2 = x[100:,].mean(axis=0)


s1 = np.dot((x[:100,]-x[:100,].mean(axis=0)).T,(x[:100,]-x[:100,].mean(axis=0)))
s2 = np.dot((x[100:,]-x[100:,].mean(axis=0)).T,(x[100:,]-x[100:,].mean(axis=0)))
s = (s1+s2)/(200-2)

d=np.dot(np.linalg.inv(s),m1-m2)


## they are identical 
print d
print b 

print 1/(b[1:,0].T/d)




