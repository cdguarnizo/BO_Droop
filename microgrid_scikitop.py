# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:13:20 2020
CIGRE microgrid benchmark
@author: cdgua
"""
import numpy as np
import os
from microgrid import mgrid

mg = mgrid()
mg.microgrid1()
mg.MC.nd = np.int32(1e3) #Cantidad de muestras
npar = 3

def objfun(x):
    x = np.reshape(x, (1,2*npar))
    xi = x[0,:npar].reshape((1,npar))
    zita = x[0,npar:].reshape((1,npar))
    mg.Montecarlo(xi, zita, False, False)
    return mg.resDev

# and compute a baseline to beat with hyperparameter optimization 
bounds = [ (1e-7,0.15) for _ in range(2*npar) ]

from skopt import gp_minimize
# https://scikit-optimize.github.io/st8able/modules/generated/skopt.gp_minimize.html
# We can build the surrogate model using
# GP: Gaussian Process
# RF: Random Forest
# ET: ExtraTreesRegressor
# GBR: GradientBoostingRegressor
   
max_iter = 30
# seed: EI - 0, LCB-10, PI-20 
#x =  (0.15-1e-7)*np.random.rand(10) + 1e-7
#objfun(x)
path = 'Results/'
os.makedirs(path, exist_ok=True)

names = ['EI','LCB','PI']
for name in names:
    BestSol = []
    BestVal = []
    Results = []
    for i in range(10):
        print(name,' iteration: ',i)
        cigre_opt = gp_minimize(objfun,      # the function to minimize
                                bounds,      # the bounds on each dimension of x
                                acq_func=name,      # the acquisition function
                                n_calls=max_iter,   # the number of evaluations of f
                                n_random_starts=5,  # the number of random initialization points
                                random_state=i,
                                noise="gaussian")   # the random seed
        Results.append(cigre_opt['func_vals'])
        BestSol.append(cigre_opt['x'])
        BestVal.append(cigre_opt['fun'])
    np.save(path+'BO_mg2_'+name+'_Results',Results)
    np.save(path+'BO_mg2_'+name+'_BestVal',BestVal)
    np.save(path+'BO_mg2_'+name+'_BestSol',BestSol)