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
import pandas as pd

def objfun(w,dp,dv,dpq,imax):
    nd = w.size
    res = np.sum(dp**2)/(nd-1) #Power losses around 0
    res += np.sum((w-1.)**2)/(nd-1) #w around 1
    res += np.sum((dv[:,0]-1.)**2)/(nd-1) #dv min around 1
    res += np.sum((dv[:,1]-1.)**2)/(nd-1) #dv max around 1
    res += np.sum(dpq[:,0]**2)/(nd-1) #dp around 0
    res += np.sum(dpq[:,1]**2)/(nd-1) #dq around 0
    res += np.sum(imax**2)/(nd-1) #imax around 0
    return res

mg = mgrid()
npar = mg.npar
mg.MC.nd = np.int32(1e5) #Cantidad de muestras
nsets = 5

df = pd.DataFrame(columns=['microGridType','ParamSet','SampleSize','Mean','StdDev'])
idx = 0
samp_size = [100, 250, 500, 1000, 5000, 10000, 50000, 100000]
gridNames = ['mgrid1','mgrid2']
np.random.seed(1234)
for gridName in gridNames:
    print(gridName)
    for set in range(nsets):
        print("Iteration ",set)
        name = gridName+'_data'+str(set)+'.npz'
        data = np.load(name)
        #xi = data['params'][:npar]
        #zita = data['params'][npar:]
        for siz in samp_size:
            res = []
            for k in range(100):
                index = np.random.choice(np.arange(100000), size=siz)
                wt = data['w'][index]
                dpt = data['dp'][index]
                dvt = data['dv'][index,:]
                dpqt = data['dpq'][index,:]
                imaxt = data['imax'][index,:]
                res.append(objfun(wt,dpt,dvt,dpqt,imaxt))
            row = [gridName,set,siz,np.mean(res),np.std(res)]
            df.loc[idx] = row
            idx = idx+1

df.to_csv('SampleSizeComparison.csv')