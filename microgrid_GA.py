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
def objfun(x):
    xi = x[:mg.NumC].reshape((1, mg.NumC))
    zita = x[mg.NumC:].reshape((1, mg.NumC))
    mg.Montecarlo(xi, zita, None, False, False)
    return mg.resDev

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
    name = 'GA_'+micro

    for size in sampleSizes:
        mg.MC.nd = np.int32(size) #Cantidad de muestras
        path = 'ResultsSize'+str(mg.MC.nd)+'/'
        os.makedirs(path, exist_ok=True)

        max_iter = 30
        BestSol = []
        BestVal = []
        Results = []
        for i in range(10):
            print(f'{name} GA size: {size}, iter: {i}')
            np.random.seed(i)
            X = (bounds[:,1]-bounds[:,0])*np.random.rand(10,2*mg.NumC)+bounds[:,0]
            cigre_opt = GA(objfun,10, bounds[:,0], bounds[:,1], max_iter, X0 = X)
            print(cigre_opt['best_x'], cigre_opt['best_f'])  
            Results.append(cigre_opt['fval'])
            BestSol.append(cigre_opt['best_x'])
            BestVal.append(cigre_opt['best_f'])
        np.save(path+name+'_Results',Results)
        np.save(path+name+'_BestVal',BestVal)
        np.save(path+name+'_BestSol',BestSol)