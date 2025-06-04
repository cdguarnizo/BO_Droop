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
def objfun(x):
    x = np.reshape(x, (1,2*npar))
    xi = x[0,:npar].reshape((1,npar))
    zita = x[0,npar:].reshape((1,npar))
    mg.Montecarlo(xi, zita, scales, False, False, saturate=True)
    return mg.resDev

# and compute a baseline to beat with hyperparameter optimization 


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
'''
bounds2 = np.array(bounds)
names = ['EI','LCB','PI']
for name1 in names:
    BestSol = []
    BestVal = []
    Results = []
    for i in range(10):
        print(name1,' iteration: ',i)
        np.random.seed(i)
        X = (bounds2[:,1]-bounds2[:,0])*np.random.rand(10,2*npar)+bounds2[:,0]
        cigre_opt = gp_minimize(objfun,      # the function to minimize
                                bounds,      # the bounds on each dimension of x
                                acq_func=name1,      # the acquisition function
                                n_calls=max_iter+10,   # the number of evaluations of f
                                n_initial_points=0,  # the number of random initialization points
                                x0 = X.tolist(),
                                random_state=i, # the random seed
                                noise="gaussian")  
        Results.append(cigre_opt['func_vals'])
        BestSol.append(cigre_opt['x'])
        BestVal.append(cigre_opt['fun'])
    np.save(path+name+'_'+name1+'_Results',Results)
    np.save(path+name+'_'+name1+'_BestVal',BestVal)
    np.save(path+name+'_'+name1+'_BestSol',BestSol)
'''
microNames = ['mg3']
sampleSizes = [50,100,200,500,1000]
#sampleSizes = [500]
names = ['EI','LCB','PI']
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
    elif micro=='mg2':
        mg.microgrid1()
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

    npar = mg.NumC
    bounds = np.array([1e-7,0.15])*np.ones((2*npar,2))
    bounds2 = [ (1e-7,0.15) for _ in range(2*npar) ]

    name = 'BO_'+micro

    for size in sampleSizes:
        mg.MC.nd = np.int32(size) #Cantidad de muestras
        path = 'ResultsSize'+str(mg.MC.nd)+'/'
        os.makedirs(path, exist_ok=True)

        max_iter = 30
        
        for name1 in names:
            BestSol = []
            BestVal = []
            Results = []
            for i in range(10):
                print(f'{name} size: {size}, iter: {i}')
                np.random.seed(i)
                X = (bounds[:,1]-bounds[:,0])*np.random.rand(10,2*npar)+bounds[:,0]
                cigre_opt = gp_minimize(objfun,      # the function to minimize
                                    bounds2,      # the bounds on each dimension of x
                                    acq_func=name1,      # the acquisition function
                                    n_calls=max_iter+10,   # the number of evaluations of f
                                    n_initial_points=0,  # the number of random initialization points
                                    x0 = X.tolist(),
                                    random_state=i, # the random seed
                                    noise="gaussian") 
                print(cigre_opt['x'], cigre_opt['fun'])
                Results.append(cigre_opt['func_vals'])
                BestSol.append(cigre_opt['x'])
                BestVal.append(cigre_opt['fun'])
            np.save(path+name+'_'+name1+'_Results',Results)
            np.save(path+name+'_'+name1+'_BestVal',BestVal)
            np.save(path+name+'_'+name1+'_BestSol',BestSol)