# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:13:20 2019
CIGRE microgrid benchmark
@author: cdgua
"""
import numpy as np
from microgrid import mgrid
import matplotlib.pyplot as plt
from scipy.stats import mode
import tikzplotlib

mg = mgrid()
npar = mg.npar
mg.MC.nd = np.int32(1e5) #Cantidad de muestras
nsets = 5
bounds = np.array([1e-7,0.15])*np.ones((2*npar,2))
np.random.seed(0)
X = (bounds[:,1]-bounds[:,0])*np.random.rand(nsets,2*npar)+bounds[:,0]

for k in range(nsets):
    print("Iteration ",k)
    mg.savename = 'mgrid1_data'+str(k)
    xi = X[k,:npar]
    zita = X[k,npar:]
    mg.Montecarlo(xi, zita, savedata=True)
    

""" 
samp_size = [500, 1000, 5000, 10000, 50000, 100000]

res_to = []
for siz in samp_size:
    res = []
    np.random.seed(1234)
    for k in range(100):
        index = np.random.choice(np.arange(100000), size=siz)
        wt = w[index]
        dvt = dv[index,:]
        dpqt = dpq[index,:]
        res.append(objfun(wt,dvt,dpqt))
    res_to.append([np.mean(res),np.std(res)]) 
    """


