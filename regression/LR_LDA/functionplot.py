import matplotlib.pyplot as plt 
import numpy as np 

def fcontour(fun,plotrange=[-5,5],MeshDensity=50):
    if len(plotrange)==2:
        plotrange.extend(plotrange)
        
    x=np.linspace(plotrange[0],plotrange[1],MeshDensity)
    y=np.linspace(plotrange[2],plotrange[3],MeshDensity)
    X,Y = np.meshgrid(x,y) 
    Z = fun(X,Y) # the fun must be vectorized
    #todo

    plt.figure()
    plt.contourf(X,Y,Z,cmap=plt.cm.RdBu)
    plt.colorbar()


def z_func(x,y):
    return (1-(x**2+y**3))*np.exp(-(x**2+y**2)/2)
def main():
    fcontour(z_func)

if __name__=='__main__':
    main()