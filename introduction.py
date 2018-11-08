# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:39:51 2018

@author: Alexis

CODE CORRESPONDING TO THE INTRODUCTION NOTEBOOK
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from models import piecewise
import time

from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials


from matplotlib import animation
import matplotlib.patches as patches
import matplotlib.path as path
# =============================================================================
# Simple cases
# =============================================================================


#gaussian mixture
f, g, h = norm(loc=-3, scale = 1), norm(loc=0, scale = 1), norm(loc=4, scale = 1)
def gaussian_mixture(x):
    return(0.49*(1-f.pdf(x))+0.1*(1-g.pdf(x))+0.51*(1-h.pdf(x)))
  
#random piecewise linear function
def piecewise_linear(x):
    return(piecewise(breaks=15)(x))

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
    max_evals=1000,
    trials=trials_tpe_1)

best_rnd_1 = fmin(objective1,
    space=hp.uniform('x', -5, 5),
    algo=rand.suggest,
    max_evals=1000,
    trials=trials_rnd_1)

    
trials_tpe_2 = Trials()
trials_rnd_2 = Trials()

best_tpe_2 = fmin(objective2,
    space=hp.uniform('x', -5, 5),
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials_tpe_2)

best_rnd_2 = fmin(objective2,
    space=hp.uniform('x', -5, 5),
    algo=rand.suggest,
    max_evals=1000,
    trials=trials_rnd_2)

#general 

#best scores
print(best_tpe_1)
print(best_rnd_1)

print(best_tpe_2)
print(best_rnd_2)
#in both cases we reach the global min! rnd is a little bit closer from the expected result

#execution time
print('Average time for an iteration of TPE on function 1: {:.4f} sec'.format((trials_tpe_1.results[-1]['eval_time']-trials_tpe_1.results[0]['eval_time'])/len(trials_tpe_1)))
print('Average time for an iteration of random search on function 1: {:.4f} sec'.format((trials_rnd_1.results[-1]['eval_time']-trials_rnd_1.results[0]['eval_time'])/len(trials_rnd_1)))

print('Average time for an iteration of TPE on function 2: {:.4f} sec'.format((trials_tpe_2.results[-1]['eval_time']-trials_tpe_2.results[0]['eval_time'])/len(trials_tpe_2)))
print('Average time for an iteration of random search on function 2: {:.4f} sec'.format((trials_rnd_2.results[-1]['eval_time']-trials_rnd_2.results[0]['eval_time'])/len(trials_rnd_2)))
#rnd is 6 times faster

#iterations to find best trial. trial id corresponds to the trial number whenexecuted seqentially
print(trials_tpe_1.best_trial['tid'])
print(trials_rnd_1.best_trial['tid'])

print(trials_tpe_2.best_trial['tid'])
print(trials_rnd_2.best_trial['tid'])

#focus on evolution: how does the q=algorithm sample their values?



def animation_plot(trial_object,generative_function,x_range=(-5,5),stop_at=1000,anim_len=10):
    
    x_axis = np.linspace(*x_range,5000)
    interval=anim_len/stop_at*1000
    
    n_bins=50
    
    max_,min_=max(generative_function(x_axis)), min(generative_function(x_axis))
    y_range=min_-0.1*abs(min_), max_+0.1*abs(max_)
    
    samples = list(map(lambda x: x['x'], trial_object.results))[:stop_at]


    fig = plt.figure(figsize=(16,9))
    ax1=fig.add_subplot(121)
    ax1.set_xlim(x_range[0],x_range[1])
    ax1.set_ylim(y_range[0],y_range[1])
    ax1.plot(x_axis,generative_function(x_axis),color='b',linestyle='dashed')
    scats, = ax1.plot([], [],'ro',markersize=5)
    scats.set_data([], [])
    plt.title('Sampled values evolution')
    
    n, bins = np.histogram(samples, n_bins)
    
    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + 0.001
    nrects = len(left)
    
    # here comes the tricky part -- we have to set up the vertex and path
    # codes arrays using moveto, lineto and closepoly
    
    # for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
    # CLOSEPOLY; the vert for the closepoly is ignored but we still need
    # it to keep the codes aligned with the vertices
    nverts = nrects*(1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom
    
    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(
        barpath, facecolor='blue', edgecolor='yellow', alpha=0.5)
    
    ax2=fig.add_subplot(122)
    ax2.add_patch(patch)
    ax2.set_xlim(*x_range)
    ax2.set_ylim(0, 1)
    plt.title('Sampled values histogram (normed)')

    # animation function.  This is called sequentially
    def animate(i):
        x =samples[:i]
        y = generative_function(x)
        scats.set_data(x, y)
        
        n, bins = np.histogram(x, n_bins)
        n= n/n.max()
    
        top = bottom + n
        verts[1::5, 1] = top
        verts[2::5, 1] = top
        return scats,patch
    
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=stop_at, interval=interval, blit=True, repeat=False)

    plt.show()
    return(anim)

#Random search on the gaussian mixture loss
animation_plot(trials_rnd_1,gaussian_mixture)

#TPE algorithm on the gaussian mixture loss
animation_plot(trials_tpe_1,gaussian_mixture)

#Rnd algorithm on the piecewise linear loss
animation_plot(trials_rnd_2,piecewise_linear)

#TPE algorithm on the piecewise linear loss
animation_plot(trials_tpe_2,piecewise_linear)
