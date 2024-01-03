# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:13:20 2019
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
mg.MC.nd = np.int(5000) #Monte Carlo's sample size

import GPyOpt
# https://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/index.ipynb
# https://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_reference_manual.ipynb
def objfun(x):
    xi = x[0,:5].reshape((1,5))
    zita = x[0,5:].reshape((1,5))
    mg.Montecarlo_ret(xi, zita, False, False)
    return mg.res

boundxi = (0.05,0.5)
bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': bound},
     {'name': 'var_2', 'type': 'continuous', 'domain': bound},
     {'name': 'var_3', 'type': 'continuous', 'domain': bound},
     {'name': 'var_4', 'type': 'continuous', 'domain': bound},
     {'name': 'var_5', 'type': 'continuous', 'domain': bound},
     {'name': 'var_6', 'type': 'continuous', 'domain': bound},
     {'name': 'var_7', 'type': 'continuous', 'domain': bound},
     {'name': 'var_8', 'type': 'continuous', 'domain': bound},
     {'name': 'var_9', 'type': 'continuous', 'domain': bound},
     {'name': 'var_10', 'type': 'continuous', 'domain': bound}]

max_iter = 100
BestSol = []
BestVal = []
Results = []
for i in range(10):
    print(i)
    cigre_opt = GPyOpt.methods.BayesianOptimization(f=objfun,
                                                    domain=bounds,
                                                    model_type='GP',
                                                    acquisition_type='LCB',
                                                    ARD=False,
                                                    exact_feval=False,
                                                    initial_design_numdata=1)
    cigre_opt.run_optimization(max_iter)
    Results.append(cigre_opt.Y_best)
    BestSol.append(cigre_opt.x_opt)
    BestVal.append(cigre_opt.Y_best[-1])
np.save('Results_LCB_STD',Results)
np.save('BestVal_LCB_STD',BestVal)
np.save('BestSol_LCB_STD',BestSol)
