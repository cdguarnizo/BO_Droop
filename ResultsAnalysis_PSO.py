# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:07:44 2020

@author: cristianguarnizo
"""

import numpy as np
import matplotlib.pyplot as plt
#import tikzplotlib as tp
import os
from microgrid import mgrid
mg = mgrid()
mg.microgrid1()
mg.path = 'Results/Figs/GA_mg2/'
path = 'Results/GA_mg2'
npar = mg.npar
#GPyOpt
BestVal = np.load(path+'_BestVal.npy')
print(BestVal)

plt.figure()
plt.boxplot(BestVal.reshape((10,1)),labels=["PSO"])

# EI
ind = np.argmin(BestVal)
print(BestVal[ind])
BestSol = np.load(path+'_BestSol.npy')
res = np.load(path+'_Results.npy')

plt.figure()
#fig, ax = plt.subplots()
indpar = np.arange(1,npar+1)
newticks = ["x"+str(i) for i in indpar]
print(indpar)
plt.scatter(indpar,BestSol[[ind],:npar].reshape(-1,), s=60, marker ='o', facecolors='none', edgecolor='black', label ='EI')
plt.xticks(indpar,newticks[:npar])
plt.legend(loc='best', shadow=True, ncol=1)
plt.title('Xi')
plt.grid()
plt.show()
#tp.save("xi_comp.tex")


plt.figure()
#fig, ax = plt.subplots()
indpar = np.arange(1,npar+1)
plt.scatter(indpar,BestSol[[ind],npar:].reshape(-1,), s=60, marker ='o', facecolors='none', edgecolor='black', label ='EI')
plt.xticks([1,2,3,4,5],["x1","x2","x3","x4","x5"])
#plt.legend(loc='upper left', shadow=True, ncol=1)
plt.title('Zita')
plt.grid()
plt.show()
#tp.save("zita_comp.tex")

## plot histograms
os.makedirs(mg.path, exist_ok=True)
xi = BestSol[[ind],:npar]
zita = BestSol[[ind],npar:]
print('Mejores parametros: ',BestSol[[ind],:])
mg.Montecarlo(xi,zita,graficar=False, flagres=False)
print(mg.optmetrics)

## Plot w simulaiton
#from microgrid import mgrid
#mg = mgrid()
    
mg.converters[:,2] = BestSol[[ind],:npar]
mg.converters[:,3] = BestSol[[ind],npar:]
mg.updateYd(4)
mae, stdw = mg.dynamic_sim(mg.converters, nd = 2000, pw_flag=True)
#tp.save("01.tex")
print(mae,stdw)

'''
mg.converters[:,2] = BestSol_MPI[[ind],:5]
mg.converters[:,3] = BestSol_MPI[[ind],5:]
mae, stdw =mg.dynamic_sim(mg.converters, nd = 2000, pw_flag=True)
#tp.save("02.tex")
print(mae,stdw)

mg.converters[:,2] = BestSol_LCB[[ind],:5]
mg.converters[:,3] = BestSol_LCB[[ind],5:]
mae, stdw =mg.dynamic_sim(mg.converters, nd = 2000, pw_flag=True)
#tp.save("03.tex")
print(mae,stdw)
'''