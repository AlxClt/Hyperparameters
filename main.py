# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:59:04 2018

@author: Alexis

Bayesian hyperparameter optimization using hyperopt

WORK IN PROGRESS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from models import piecewise
import time



# =============================================================================
# Simple cases
# =============================================================================


#gaussian mixture
f, g, h = norm(loc=-3, scale = 1), norm(loc=0, scale = 1), norm(loc=4, scale = 1)
def gaussian_mixture(x):
    return(1-0.2*f.pdf(x)+0.35*g.pdf(x)+0.45*h.pdf(x))
  
#random piecewise linear function
def piecewise_linear(x):
    return(piecewise(breaks=15).call(x))

#plot the functions to minimize
x=np.linspace(-5,5,5000)

fig=plt.figure()
ax = fig.add_subplot(121)
ax.plot(x, piecewise_linear(x),color='g')
ax.set_title('Piecewise linear function')
ax = fig.add_subplot(122)
ax.plot(x, gaussian_mixture(x),color='r')
ax.set_title('Gaussian mixture model')
plt.show()


#the piecewise linear function looks difficult to optimize correctly

#use hyperopt to find the min

from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials

#We will make use of the trials objects to carry information along the process
def objective1(x):
    return {
        'loss': gaussian_mixture(x),
        'x':x,
        'status': STATUS_OK,
        'eval_time': time.time()
        }

def objective2(x):
    return {
        'loss': piecewise_linear(x),
        'x':x,
        'status': STATUS_OK,
        'eval_time': time.time()
        }
    
trials_tpe_1 = Trials()
trials_rnd_1 = Trials()

best_tpe_1 = fmin(objective1,
    space=hp.uniform('x', -5, 5),
    algo=tpe.suggest,
    max_evals=2000,
    trials=trials_tpe_1)

best_rnd_1 = fmin(objective1,
    space=hp.uniform('x', -5, 5),
    algo=rand.suggest,
    max_evals=2000,
    trials=trials_rnd_1)

    
trials_tpe_2 = Trials()
trials_rnd_2 = Trials()

best_tpe_2 = fmin(objective2,
    space=hp.uniform('x', -5, 5),
    algo=tpe.suggest,
    max_evals=2000,
    trials=trials_tpe_2)

best_rnd_2 = fmin(objective2,
    space=hp.uniform('x', -5, 5),
    algo=rand.suggest,
    max_evals=2000,
    trials=trials_rnd_2)



#general comparison


#focus on evolution














