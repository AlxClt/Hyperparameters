# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:50:19 2018

@author: Alexis
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import matplotlib.path as path

# =============================================================================
# Animated plot (intro)
# =============================================================================

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