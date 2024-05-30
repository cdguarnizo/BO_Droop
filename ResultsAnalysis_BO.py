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
mg.path = 'Results/Figs/EI_mg2/' #where to save figures
path = 'Results/BO_mg2' #results' name
npar = 3

#GPyOpt
BestVal_EI = np.load(path+'_EI_BestVal.npy')
BestVal_MPI = np.load(path+'_PI_BestVal.npy')
BestVal_LCB = np.load(path+'_LCB_BestVal.npy')

print(BestVal_EI, BestVal_MPI, BestVal_LCB)

plt.figure()
plt.boxplot(np.block([BestVal_EI.reshape((10,1)),BestVal_MPI.reshape((10,1)),\
                      BestVal_LCB.reshape((10,1))]),labels=["EI","MPI","LCB"])

# EI
ind = np.argmin(BestVal_EI)
print(BestVal_EI[ind])
BestSol_EI = np.load(path+'_EI_BestSol.npy')
res = np.load(path+'_EI_Results.npy')

# MPI
ind2 = np.argmin(BestVal_MPI)
print(BestVal_MPI[ind2])
BestSol_MPI = np.load(path+'_PI_BestSol.npy')
res2 = np.load(path+'_PI_Results.npy')

# LCB
ind3 = np.argmin(BestVal_LCB)
print(BestVal_LCB[ind3])
BestSol_LCB = np.load(path+'_LCB_BestSol.npy')
res3 = np.load(path+'_EI_Results.npy')

plt.figure()
#fig, ax = plt.subplots()
indpar = np.arange(1,npar+1)
plt.scatter(indpar,BestSol_EI[[ind],:npar].reshape(-1,), s=60, marker ='o', facecolors='none', edgecolor='black', label ='EI')
plt.scatter(indpar,BestSol_MPI[[ind2],:npar].reshape(-1,), s=180, marker ='x', color ='black', label='MPI')
plt.scatter(indpar,BestSol_LCB[[ind3],:npar].reshape(-1,), s=180, marker ='+', color ='black', label ='LCB')
plt.xticks([1,2,3,4,5],["x1","x2","x3","x4","x5"])
plt.legend(loc='best', shadow=True, ncol=1)
plt.title('Xi')
plt.grid()
plt.show()
#tp.save("xi_comp.tex")


plt.figure()
#fig, ax = plt.subplots()
indpar = np.arange(1,npar+1)
plt.scatter(indpar,BestSol_EI[[ind],npar:].reshape(-1,), s=60, marker ='o', facecolors='none', edgecolor='black', label ='EI')
plt.scatter(indpar,BestSol_MPI[[ind2],npar:].reshape(-1,), s=180, marker ='x', color ='black', label='MPI')
plt.scatter(indpar,BestSol_LCB[[ind3],npar:].reshape(-1,), s=180, marker ='+', color ='black', label ='LCB')
plt.xticks([1,2,3,4,5],["x1","x2","x3","x4","x5"])
#plt.legend(loc='upper left', shadow=True, ncol=1)
plt.title('Zita')
plt.grid()
plt.show()
#tp.save("zita_comp.tex")

## Mean and Std EI
res_mu = np.mean(res,0)
res_std = np.std(res,0)
index = np.arange(res.shape[1])
plt.figure()
plt.plot(index, res_mu,'k-')
#plt.plot(index,res_mu+res_std,'b--')
#plt.plot(index,res_mu-res_std,'b--')
plt.show()

## Mean and Std MPI
res2_mu = np.mean(res2,0)
res2_std = np.std(res2,0)
index = np.arange(res.shape[1])
plt.plot(index, res2_mu,'k--')
#plt.plot(index,res2_mu+res2_std,'r--')
#plt.plot(index,res2_mu-res2_std,'r--')
plt.show()

## Mean and Std LCB
res3_mu = np.mean(res3, 0)
res3_std = np.std(res3, 0)
index = np.arange(res.shape[1])
plt.plot(index, res3_mu, 'k:')
#plt.plot(index,res3_mu+res3_std,'g--')
#plt.plot(index,res3_mu-res3_std,'g--')
plt.legend(['EI','MPI','LCB'])
plt.grid()
plt.show()
#tp.save("BOiter.tex")

## plot histograms
os.makedirs(mg.path, exist_ok=True)
xi = BestSol_EI[[ind],:npar]
zita = BestSol_EI[[ind],npar:]
mg.Montecarlo(xi,zita,graficar=False, flagres=False)
print(mg.optmetrics)


## Plot w simulation
#from microgrid import cigre
#mg = cigre()
    
mg.converters[:,2] = BestSol_EI[[ind],:npar]
mg.converters[:,3] = BestSol_EI[[ind],npar:]
mg.updateYd(0)
print('Mejores parametros: ',BestSol_EI[[ind],:])
mae, stdw = mg.dynamic_sim(mg.converters, nd = 500, pw_flag=True)
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