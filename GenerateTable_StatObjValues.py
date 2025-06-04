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
mg.MC.nd = np.int32(500) #Cantidad de muestras
microGrid = ['mg2']
mgDic = {'mg1': mg.mg1,'mg2': mg.mg2, 'mg3': mg.mg3}
df = pd.DataFrame(columns=['microGridType','Iteration','PowerLossDev','FreqDev','DVmin','DVmax','DP','DQ','Imax','PowerLossMean','ObjFunc'])
nsets = 100
idx = 0
for micro in microGrid:
    mgDic[micro]()
    mg.initilizeMG()
    npar = mg.NumC
    print('Microgrid: ', micro,' Num. Params: ', npar)
    bounds = np.array([1e-7,0.15])*np.ones((2*npar,2))
    np.random.seed(0)
    X = (bounds[:,1]-bounds[:,0])*np.random.rand(nsets,2*npar)+bounds[:,0]

    #Simulate different params
    for k in range(nsets):
        mg.Montecarlo(X[k,:npar], X[k,npar:], makefig=False, flagres=False)
        row = [micro,k]
        row.extend(list(mg.optmetrics))
        row.extend([mg.resDev])
        print(row)
        df.loc[idx] = row
        idx = idx+1
    print(df[df['microGridType']==micro].describe())
    mg.Lines = []
    mg.Impedances = []
    mg.Converters = []
    mg.DemandP = []
    mg.DemandL = []
    mg.DemandType = []

#df.to_csv('StatObjValues.csv')

#print(X[54,:])