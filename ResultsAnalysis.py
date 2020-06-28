# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:07:44 2020

@author: cristianguarnizo
"""

import numpy as np
import matplotlib.pyplot as plt

#GPyOpt
BestVal_EI = np.load('BestVal_EI_STD.npy')
BestVal_EI_ARD = np.load('BestVal_EI_STD_ARD.npy')
BestVal_MPI = np.load('BestVal_MPI_STD.npy')
BestVal_MPI_ARD = np.load('BestVal_MPI_ARD_STD.npy')
BestVal_LCB = np.load('BestVal_LCB_STD.npy')
BestVal_LCB_ARD = np.load('BestVal_LCB_STD_ARD.npy')

plt.figure()
plt.boxplot(np.block([BestVal_EI.reshape((10,1)),BestVal_EI_ARD.reshape((10,1)),BestVal_MPI.reshape((10,1)),BestVal_MPI_ARD.reshape((10,1)),\
                      BestVal_LCB.reshape((10,1)),BestVal_LCB_ARD.reshape((10,1))]),labels=["EI","EI_ARD","MPI","MPI_ARD","LCB","LCB_ARD"])

#ScikitOpt
BestVal_SK_EI = np.load('BestVal_BO_SK_EI_STD_2.npy')
BestVal_SK_PI = np.load('BestVal_BO_SK_PI_STD_2.npy')
BestVal_SK_LCB = np.load('BestVal_BO_SK_LCB_STD_2.npy')

plt.figure()
plt.boxplot(np.block([BestVal_SK_EI.reshape((10,1)),BestVal_SK_PI.reshape((10,1)),BestVal_SK_LCB.reshape((10,1))]),labels=["EI","PI","LCB"])

plt.figure()
BestSol_EI = np.load('BestSol_EI_STD.npy')
plt.boxplot(BestSol_EI[[4,6],:])

#Results = np.load('Results_EI_STD.npy')
#
#plt.figure()
#plt.boxplot(Results.T)
#
#Results2 = np.load('Results_EI_STD_ARD.npy')
#plt.figure()
#plt.boxplot(Results2.T)
#
#Results3 = np.load('Results_MPI_STD_ARD.npy')
#plt.figure()
#plt.boxplot(Results3.T)
#
#Results4 = np.load('Results_MPI_STD.npy')
#plt.figure()
#plt.boxplot(Results4.T)

