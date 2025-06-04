# -*- coding: utf-8 -*-

import numpy as np
import os
from microgrid import mgrid

mg = mgrid()
methodName = 'PSO'
def objfun(x):
    xi = x[:mg.NumC].reshape((1,mg.NumC))
    zita = x[mg.NumC:].reshape((1,mg.NumC))
    mg.Montecarlo(xi, zita, scales, False, False)
    return mg.resDev

def PSO(f, psize, linf, lsup, maxiter=100, X0 = None):
    D = len(linf) # number of decision variables
    c1, c2 = 2.0, 2.0
    wstart, wf, wend = 1.2, 0.5, 0.4
    wdec = (wstart-wend)/(maxiter*wf)
    w = wstart
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
        V = w*V+c1*np.random.rand(psize,D)*(Pb-X)+c2*np.random.rand(psize,D)*(gb-X)
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
        if iter < np.floor(maxiter*wf):
            w = w-wdec

    return {'best_x': gb, 'best_f': gbval, 'fval': fval}

#microNames = ['mg1', 'mg2', 'mg3']
microNames = ['mg3']
sampleSizes = [50,100,200,500,1000]
#sampleSizes = [500]
# and compute a baseline to beat with hyperparameter optimization 
for micro in microNames:
    if micro == 'mg1':
        mg.__init__()
        scales = np.array([2.372127,
                    8.370188e-4,
                    0.003252,
                    0.004098,
                    2.462032,
                    6.179947,
                    9.559357])
    elif micro == 'mg2':
        mg.mg2()
        scales = np.array([2.557772,
                    7.267406e-3,
                    2.678419e-11,
                    1.356699e-11,
                    3.615616,
                    2.142949e-8,
                    4.319097])
    else:
        mg.mg3()
        scales = np.array([13.754020,
                    3.723757e-3,
                    0.002352,
                    1.396108e-3,
                    8.451084,
                    1.049735,
                    9.766920])
    
    bounds = np.array([1e-7,0.15])*np.ones((2*mg.NumC,2))
    name = methodName+'_'+micro

    for size in sampleSizes:
        mg.MC.nd = np.int32(size) #Cantidad de muestras
        path = 'ResultsSize'+str(mg.MC.nd)+'/'
        os.makedirs(path, exist_ok=True)

        max_iter = 30
        BestSol = []
        BestVal = []
        Results = []
        for i in range(10):
            print(f'{name} size: {size}, iter: {i}')
            np.random.seed(i)
            X = (bounds[:,1]-bounds[:,0])*np.random.rand(10,2*mg.NumC)+bounds[:,0]
            cigre_opt = PSO(objfun,10, bounds[:,0], bounds[:,1], max_iter, X0 = X)
            print(cigre_opt['best_x'], cigre_opt['best_f'])
            Results.append(cigre_opt['fval'])
            BestSol.append(cigre_opt['best_x'])
            BestVal.append(cigre_opt['best_f'])
        np.save(path+name+'_Results',Results)
        np.save(path+name+'_BestVal',BestVal)
        np.save(path+name+'_BestSol',BestSol)