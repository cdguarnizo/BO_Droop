# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:13:20 2019
CIGRE microgrid benchmark
@author: cdgua
"""
import numpy as np
import os
from microgrid import mgrid

mg = mgrid()
mg.microgrid1()
npar = mg.npar
mg.MC.nd = np.int32(1e3) #Cantidad de muestras
if npar ==5:
    name = 'GA_mg1'
else:
    name = 'GA_mg2'
path = 'Results/'
os.makedirs(path, exist_ok=True)

def objfun(x):
    xi = x[:npar].reshape((1,npar))
    zita = x[npar:].reshape((1,npar))
    mg.Montecarlo(xi, zita, False, False)
    return mg.resDev

from random import random

def GA(f, psize, linf, lsup, maxiter=100, X0 = None):
    D = len(linf) # number of decision variables
    if X0 is None:
        X = (lsup-linf)*np.random.rand(psize,D)+linf
    else:
        X = X0.copy()
    aptitud = np.apply_along_axis(f, 1, X)
    fiter = np.zeros((maxiter,))
    for iter in range(maxiter):
        #Parents Roulette selection
        ValoresRul = 1.0/aptitud #Invert fitness values
        indSort = np.argsort(ValoresRul)
        S = sum(ValoresRul)
        r = np.random.rand()*S
        sumasPar = np.cumsum(ValoresRul)
        ind = np.argwhere(sumasPar>=r)[0][0]

        sinInd = list(range(psize))
        sinInd.remove(ind)
        S2 = np.sum(ValoresRul[sinInd])
        r = np.random.rand()*S2
        sumasPar = np.cumsum(ValoresRul[sinInd])
        ind2=np.argwhere(sumasPar>=r)[0][0]

        #Parents crossover
        alpha = np.random.rand()
        hijo = alpha*X[ind,:]+(1-alpha)*X[ind2,:]

        #Children Mutation
        if np.random.rand()>0.5:
            cuantos_par = np.random.randint(X.shape[1]+1)
            indHijoMutar = np.random.permutation(X.shape[1])[:cuantos_par]
            hijo[indHijoMutar] += np.random.randn(cuantos_par)

        hijo = np.clip(hijo,linf,lsup)
        #Insert children in population
        fHijo = f(hijo)
        if fHijo < aptitud[indSort[0]]:
            X[indSort[0],:] = hijo
            aptitud[indSort[0]] = fHijo
        
        indBest = np.argmin(aptitud)
        fiter[iter] = aptitud[indBest]
        gb = X[indBest,:]
        gbval = aptitud[indBest]

    return {'best_x': gb, 'best_f': gbval, 'fval': fiter}

# and compute a baseline to beat with hyperparameter optimization 
bounds = np.array([1e-7,0.15])*np.ones((2*npar,2))

max_iter = 30
BestSol = []
BestVal = []
Results = []
for i in range(10):
    print('GA iteration', i)
    np.random.seed(i)
    X = (bounds[:,1]-bounds[:,0])*np.random.rand(10,2*npar)+bounds[:,0]
    cigre_opt = GA(objfun,10, bounds[:,0], bounds[:,1], max_iter, X0 = X)
    Results.append(cigre_opt['fval'])
    BestSol.append(cigre_opt['best_x'])
    BestVal.append(cigre_opt['best_f'])
np.save(path+name+'_Results',Results)
np.save(path+name+'_BestVal',BestVal)
np.save(path+name+'_BestSol',BestSol)