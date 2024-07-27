# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:07:44 2020

@author: cristianguarnizo
"""

import numpy as np
import os
from microgrid import mgrid
import pandas as pd

mg = mgrid()
npar = mg.npar
mg.MC.nd = np.int32(1e2) #Cantidad de muestras
Sizes = [100, 200, 500, 1000]
optNames = ['GA','PSO','BO_EI','BO_LCB','BO_PI']
microGrid = ['mg1','mg2']
df = pd.DataFrame(columns=['microGridType','OptMethod','Size','Iteration','PowerLossDev','FreqDev','DVmin','DVmax','DP','DQ','Imax','PowerLossMean','ObjFunc'])
idx = 0
for micro in microGrid:
    mg = mgrid()
    if micro == 'mg2':
        mg.microgrid1()
    for name in optNames:
        print(f"{micro} - {name}")
        npar = mg.npar
        for size in Sizes:
            path = 'ResultsSize'+str(size)+'/'+name+'_'+micro
            mg.MC.nd = np.int32(size)
            if name[:2]=='BO':
                path = 'ResultsSize'+str(size)+'/'+name[:2]+'_'+micro+name[2:]
            BestSol = np.load(path+'_BestSol.npy')
            #Simulate each set of params
            for k in range(BestSol.shape[0]):
                xi = BestSol[k,:npar]
                zita = BestSol[k,npar:]
                mg.Montecarlo(xi, zita, graficar=False, flagres=False)
                row = [micro,name,size,k]
                row.extend(list(mg.optmetrics))
                row.extend([mg.resDev])
                print(row)
                df.loc[idx] = row
                idx = idx+1

df.to_csv('DataAllMethods.csv')