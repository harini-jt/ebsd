# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:05:40 2021

@author: harini


%timeit _= ndimage.median_filter(Eulers[0,:,:],3)
193 ms ± 14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit _ = medianFilter(Eulers, data10, 3)
327 ms ± 14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""                
import matplotlib.pyplot as plt
import numpy as np
import linecache
from numba import jit, njit,vectorize,stencil
import time
import numba as nb
import math
from scipy import ndimage
filename = 'Fe 1100 Map Data 2.ctf' 

#loading data from file...
def read_data(filename):
    f = open(filename) 
    x=int(float(linecache.getline(filename,5)[7:]))
    y=int(float(linecache.getline(filename,6)[7:]))
    x1=float(linecache.getline(filename,7)[6:])
    y1=float(linecache.getline(filename,8)[6:])
    Phases=int(linecache.getline(filename,13)[7:])
    skipRows=14+Phases

    data1=np.transpose(np.loadtxt(filename, skiprows=skipRows))
    print(y,x)
    data0=data1[0].reshape(y,x)
    Eulers=np.zeros((3,y,x))
    Eulers[0][:][:]=data1[5].reshape(y,x)
    Eulers[1][:][:]=data1[6].reshape(y,x)
    Eulers[2][:][:]=data1[7].reshape(y,x)
    f.close()
    return(Eulers,data0,x,y,x1,y1)


@njit
def medianFilter(Eulers, data0, w):
    for i in range(w,Eulers.shape[1]-w):
        for j in range(w,Eulers.shape[2]-w):
            if data0[i,j]==0:
                block0 = Eulers[0,i-w:i+w+1, j-w:j+w+1]
                m0 = np.median(block0)
                block1 = Eulers[1,i-w:i+w+1, j-w:j+w+1]
                m1 = np.median(block1)
                block2 = Eulers[2,i-w:i+w+1, j-w:j+w+1]
                m2 = np.median(block2) 
                Eulers[0,i,j] = m0
                Eulers[1,i,j] = m1
                Eulers[2,i,j] = m2
    return Eulers
tic = time.time()
Eulers,data10,x,y,x1,y1 = read_data(filename)
# Eulers[0,:,:] = ndimage.median_filter(Eulers[0,:,:],3)
# Eulers[1,:,:]= ndimage.median_filter(Eulers[1,:,:],3)
# Eulers[2,:,:] = ndimage.median_filter(Eulers[2,:,:],3)
Eulers=medianFilter(Eulers, data10, 3)  

@njit()
def theta(p1,p,p2,q1,q,q2):
    x1 = (math.cos(p1)*math.cos(p2) - math.cos(p)*math.sin(p1)*math.sin(p2))*(math.cos(q1)*math.cos(q2) - math.cos(q)*math.sin(q1)*math.sin(q2))
    x2 = (math.cos(p2)*math.sin(p1) + math.cos(p)*math.cos(p1)*math.sin(p2))*(math.cos(q2)*math.sin(q1) + math.cos(q)*math.cos(q1)*math.sin(q2))
    x3 = (math.sin(p)*math.sin(p2))*(math.sin(q)*math.sin(q2))
    x4 = (-1*math.cos(p1)*math.sin(p2) - math.cos(p)*math.cos(p2)*math.sin(p1))*(-1*math.cos(q1)*math.sin(q2) - math.cos(q)*math.cos(q2)*math.sin(q1))
    x5 = (math.cos(p)*math.cos(p1)*math.cos(p2) - math.sin(p1)*math.sin(p2))*(math.cos(q)*math.cos(q1)*math.cos(q2) - math.sin(q1)*math.sin(q2))
    x6 = (math.cos(p2)*math.sin(p))*(math.cos(q2)*math.sin(q))
    x7 = math.sin(p)*math.sin(p1)*(math.sin(q)*math.sin(q1)) 
    x8 = (-1*math.cos(p1)*math.sin(p))*(-1*math.cos(q1)*math.sin(q))
    x9 = math.cos(p) * math.cos(q)
    f = 0.5*(x1+x2+x3+x4+x5+x6+x7+x8+x9-1)
    if f>1 or f<-1: f=0
    return (abs((math.acos(f)))*180/np.pi)


@njit(fastmath=True)
def main():
    kam=np.zeros((y,x), dtype=np.float64)
    e1 = Eulers[0,:,:]*math.pi/180
    e2 = Eulers[1,:,:]*math.pi/180
    e3 = Eulers[2,:,:]*math.pi/180
    for i in range(y-2):
        for j in range(x-2):
            for m in range(3):
                for n in range(3):
                    if(m==1 and n==1):
                        pass
                    elif(m+n)%2 ==0:
                        kam[i+1,j+1] += theta(e1[i+1,j+1],e2[i+1,j+1],e3[i+1,j+1],e1[i+m,j+n],e2[i+m,j+n],e3[i+m,j+n])/np.sqrt(2)
                    else:
                        kam[i+1,j+1] += theta(e1[i+1,j+1],e2[i+1,j+1],e3[i+1,j+1],e1[i+m,j+n],e2[i+m,j+n],e3[i+m,j+n])
            
    return kam/8

    

kam= main()
toc = time.time()
print(f"Program Execution Time : {(toc-tic)} seconds")
plt.imshow(kam)
plt.colorbar()

#Execution time of the main() function execution
#%time _= main()
#%timeit _= main()
