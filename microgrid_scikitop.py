# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:13:20 2020
CIGRE microgrid benchmark
@author: cdgua
"""
import numpy as np
from microgrid import cigre


mg = cigre()
#xi = 0.05 + 0.15*np.ones((1,5)) #np.random.rand(1,5)
#zita = 0.05 + 0.15*np.ones((1,5)) #np.random.rand(1,5)
xi = np.array([[0.0636,0.0806,0.0639,0.0762,0.1425]])
zita = np.array([[0.4596,0.2963,0.2338,0.4481,0.4753]])
mg.MC.nd = np.int(1e4) #Cantidad de muestras

def objfun(x):
    x = np.reshape(x, (1,10))
    xi = x[0,:5].reshape((1,5))
    zita = x[0,5:].reshape((1,5))
    mg.Montecarlo(xi, zita, False, False)
    return mg.res

# and compute a baseline to beat with hyperparameter optimization 
bounds = [ (0.05,0.5) for _ in range(10) ]

from skopt import gp_minimize
# https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
# We can build the surrogate model using
# GP: Gaussian Process
# RF: Random Forest
# ET: ExtraTreesRegressor
# GBR: GradientBoostingRegressor
   
max_iter = 30
BestSol = []
BestVal = []
Results = []
for i in range(10):
    print(i)
    cigre_opt = gp_minimize(objfun,      # the function to minimize
                            bounds,      # the bounds on each dimension of x
                            acq_func="LCB",      # the acquisition function
                            n_calls=max_iter,         # the number of evaluations of f
                            n_random_starts=3,  # the number of random initialization points
                            random_state=i,
                            noise="gaussian")   # the random seed
    Results.append(cigre_opt['func_vals'])
    BestSol.append(cigre_opt['x'])
    BestVal.append(cigre_opt['fun'])
np.save('Results_BO_SK_LCB_STD_2',Results)
np.save('BestVal_BO_SK_LCB_STD_2',BestVal)
np.save('BestSol_BO_SK_LCB_STD_2',BestSol)