import numpy as np 
alpha=[0,0,0]
b=0
X=np.array([[3, 3, 1], [4, 3, 1], [1, 1, -1]])
x=X[:,0:-1]
y=X[:,-1]
g=np.dot(x,x.T)
def cal(i):
    return  y[i]*(np.dot(alpha*y,g[i,:])+b)

def update(i):
    global alpha,b
    print "alpha: " + str(alpha) + " b: " + str(b)
    alpha[i]+=1
    b+=y[i]
def check():
    flag = False
    for i in range(len(x)):
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        print "RESULT: alpha: " + str(alpha) + " b: " + str(b)
    return flag
if __name__ == "__main__":
    while check():
        pass