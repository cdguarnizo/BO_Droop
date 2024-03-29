# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:13:20 2020
CIGRE microgrid benchmark
@author: cdgua
"""
import numpy as np
from microgrid import cigre


mg = cigre()
mg.MC.nd = np.int32(1e3) #Cantidad de muestras

def objfun(x):
    x = np.reshape(x, (1,10))
    xi = x[0,:5].reshape((1,5))
    zita = x[0,5:].reshape((1,5))
    mg.Montecarlo_ret(xi, zita, False, False)
    return mg.resDev

# and compute a baseline to beat with hyperparameter optimization 
bounds = [ (1e-7,0.15) for _ in range(10) ]

from skopt import gp_minimize
# https://scikit-optimize.github.io/st8able/modules/generated/skopt.gp_minimize.html
# We can build the surrogate model using
# GP: Gaussian Process
# RF: Random Forest
# ET: ExtraTreesRegressor
# GBR: GradientBoostingRegressor
   
max_iter = 30
BestSol = []
BestVal = []
Results = []
# seed: EI - 0, LCB-10, PI-20 
for i in range(10):
    print('Iteration ',i)
    cigre_opt = gp_minimize(objfun,      # the function to minimize
                            bounds,      # the bounds on each dimension of x
                            acq_func="LCB",      # the acquisition function
                            n_calls=max_iter,         # the number of evaluations of f
                            n_random_starts=3,  # the number of random initialization points
                            random_state=i+10,
                            noise="gaussian")   # the random seed
    Results.append(cigre_opt['func_vals'])
    BestSol.append(cigre_opt['x'])
    BestVal.append(cigre_opt['fun'])
np.save('Results_BO_SK_LCB_STD',Results)
np.save('BestVal_BO_SK_LCB_STD',BestVal)
np.save('BestSol_BO_SK_LCb_STD',BestSol)