# -*- coding: utf-8 -*-

import numpy as np
import os
from microgrid import mgrid

mg = mgrid()
#mg.microgrid1()
npar = mg.npar
mg.MC.nd = np.int32(1000) #Cantidad de muestras
if npar ==5:
    name = 'PSO_mg1'
else:
    name = 'PSO_mg2'

path = 'ResultsSize'+str(mg.MC.nd)+'/'
os.makedirs(path, exist_ok=True)

def objfun(x):
    xi = x[:npar].reshape((1,npar))
    zita = x[npar:].reshape((1,npar))
    mg.Montecarlo(xi, zita, False, False)
    return mg.resDev

from random import random

def PSO(f, psize, linf, lsup, maxiter=100, X0 = None):
    D = len(linf) # number of decision variables
    c1, c2 = 1.5, 1.5
    if X0 is None:
        X = (lsup-linf)*np.random.rand(psize,D)+linf
    else:
        X = X0.copy()
    fval = np.apply_along_axis(f, 1, X)
    
    Pb = X.copy()
    gb = X[fval.argmin(),:]
    gbval = fval.min()
    Fb = fval.copy()

    V = np.zeros((psize, D)) #inicializar matriz de velocidades
    for iter in range(maxiter):
        V = c1*np.random.rand(psize,D)*(Pb-X)+c2*np.random.rand(psize,D)*(gb-X)
        X = X + V
        X = np.clip(X, linf, lsup)
        F = np.apply_along_axis(f,1,X) #evaluar particula
        #Actualizar mejores locales
        idx = F<Fb
        Pb[idx,:] = X[idx,:]
        Fb[idx] = F[idx]
        #Actualizar mejores globales
        indmin = fval.argmin()
        if fval[indmin]<gbval:
            gb = X[indmin,:] #Actualizar mejor global
            gbval = fval[indmin] #Actualizar mejor global
        #Actualizar mejores locales
        idx = F<Fb
        Pb[idx,:] = X[idx,:]
        Fb[idx] = F[idx]

    return {'best_x': gb, 'best_f': gbval, 'fval': fval}

# and compute a baseline to beat with hyperparameter optimization 
bounds = np.array([1e-7,0.15])*np.ones((2*npar,2))

max_iter = 30
n_pob = 10
BestSol = []
BestVal = []
Results = []
for i in range(10):
    print('PSO iteration ', i)
    np.random.seed(i)
    X = (bounds[:,1]-bounds[:,0])*np.random.rand(10,2*npar)+bounds[:,0]
    cigre_opt = PSO(objfun, n_pob, bounds[:,0], bounds[:,1], max_iter, X0 = X)
    Results.append(cigre_opt['fval'])
    BestSol.append(cigre_opt['best_x'])
    BestVal.append(cigre_opt['best_f'])
np.save(path+name+'_Results',Results)
np.save(path+name+'_BestVal',BestVal)
np.save(path+name+'_BestSol',BestSol)