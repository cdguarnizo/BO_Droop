# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:07:44 2020

@author: cristianguarnizo
"""

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tp


#GPyOpt
#BestVal_EI = np.load('BestVal_EI_STD_3p_50i.npy')
BestVal_EI = np.load('BestVal_BO_SK_EI_STD_EV5.npy')
#BestVal_EI_ARD = np.load('BestVal_EI_ARD_3p_50i_MC500_LI0P001_LS0P09_new.npy')
BestVal_MPI = np.load('BestVal_BO_SK_PI_STD_EV5.npy')
#BestVal_MPI_ARD = np.load('BestVal_MPI_ARD_3p_50i_MC500_LI0P001_LS0P09_new.npy')
BestVal_LCB = np.load('BestVal_BO_SK_LCB_STD_EV5.npy')
#BestVal_LCB_ARD = np.load('BestVal_LCB_ARD_3p_50i_MC500_LI0P001_LS0P09_new.npy')

print(BestVal_EI, BestVal_MPI, BestVal_LCB)

plt.figure()
plt.boxplot(np.block([BestVal_EI.reshape((10,1)),BestVal_MPI.reshape((10,1)),\
                      BestVal_LCB.reshape((10,1))]),labels=["EI","MPI","LCB"])

#plt.boxplot(np.block([BestVal_EI.reshape((10,1)),BestVal_EI_ARD.reshape((10,1)),BestVal_MPI.reshape((10,1)),BestVal_MPI_ARD.reshape((10,1)),\
#                      BestVal_LCB.reshape((10,1)),BestVal_LCB_ARD.reshape((10,1))]),labels=["EI","EI-ARD","MPI","MPI-ARD","LCB","LCB-ARD"])

#tp.save("PerformComp_Boxplot.tex")
#tp.save('CompareARD.tikz')

#ScikitOpt
#BestVal_SK_EI = np.load('BestVal_BO_SK_EI_STD_2.npy')
#BestVal_SK_PI = np.load('BestVal_BO_SK_PI_STD_2.npy')
#BestVal_SK_LCB = np.load('BestVal_BO_SK_LCB_STD_2.npy')
#
#plt.figure()
#plt.boxplot(np.block([BestVal_SK_EI.reshape((10,1)),BestVal_SK_PI.reshape((10,1)),BestVal_SK_LCB.reshape((10,1))]),labels=["EI","PI","LCB"])

# EI

ind = np.argmin(BestVal_EI)
print(BestVal_EI[ind])
BestSol_EI = np.load('BestSol_BO_SK_EI_STD_EV5.npy')
res = np.load('Results_BO_SK_EI_EV5.npy')


# MPI
ind2 = np.argmin(BestVal_MPI)
print(BestVal_MPI[ind2])
BestSol_MPI = np.load('BestSol_BO_SK_PI_STD_EV5.npy')
res2 = np.load('Results_BO_SK_PI_EV5.npy',allow_pickle=True)
#res2 = np.delete(res2, [6,7], 0)
#res2 = np.concatenate(res2).astype(None)
#res2 = res2.reshape((8,53))

# LCB
ind3 = np.argmin(BestVal_LCB)
print(BestVal_LCB[ind3])
BestSol_LCB = np.load('BestSol_BO_SK_LCB_STD_EV5.npy')
res3 = np.load('Results_BO_SK_LCB_EV5.npy')

plt.figure()
#fig, ax = plt.subplots()
indpar = np.arange(1,6)
plt.scatter(indpar,BestSol_EI[[ind],:5].reshape(-1,), s=60, marker ='o', facecolors='none', edgecolor='black', label ='EI')
plt.scatter(indpar,BestSol_MPI[[ind2],:5].reshape(-1,), s=180, marker ='x', color ='black', label='MPI')
plt.scatter(indpar,BestSol_LCB[[ind3],:5].reshape(-1,), s=180, marker ='+', color ='black', label ='LCB')
plt.xticks([1,2,3,4,5],["x1","x2","x3","x4","x5"])
plt.legend(loc='best', shadow=True, ncol=1)
plt.title('Xi')
plt.grid()
plt.show()
#tp.save("xi_comp.tex")


plt.figure()
#fig, ax = plt.subplots()
indpar = np.arange(1,6)
plt.scatter(indpar,BestSol_EI[[ind],5:].reshape(-1,), s=60, marker ='o', facecolors='none', edgecolor='black', label ='EI')
plt.scatter(indpar,BestSol_MPI[[ind2],5:].reshape(-1,), s=180, marker ='x', color ='black', label='MPI')
plt.scatter(indpar,BestSol_LCB[[ind3],5:].reshape(-1,), s=180, marker ='+', color ='black', label ='LCB')
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
from microgrid import cigre
mg = cigre()
xi = BestSol_EI[[ind],:5]
zita = BestSol_EI[[ind],5:]
mg.Montecarlo_ret(xi,zita, graficar=True, flagres=False)
print(mg.resDev)

'''
## Plot w simulaiton
from microgrid import cigre
mg = cigre()
    
mg.converters[:,2] = BestSol_EI[[ind],:5]
mg.converters[:,3] = BestSol_EI[[ind],5:]
mae, stdw = mg.dynamic_sim(mg.converters, nd = 2000, pw_flag=True)
#tp.save("01.tex")
print(mae,stdw)


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