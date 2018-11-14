# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:39:51 2018

@author: Alexis

CODE CORRESPONDING TO THE INTRODUCTION NOTEBOOK
"""

import numpy as np
from scipy.stats import norm
from models import piecewise
import time
import pandas as pd
#Bayesian optimisation
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials

#VIsualisation
import matplotlib.pyplot as plt



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
#Gaussian mixture
def objective1(x):
    return {
        'loss': gaussian_mixture(x),
        'x':x,
        'status': STATUS_OK,
        'eval_time': time.time()
        }

#Piecewise linear
def objective2(x):
    return {
        'loss': piecewise_linear(x),
        'x':x,
        'status': STATUS_OK,
        'eval_time': time.time()
        }


start = time.time()

#Gaussian mixture function optimization with tpe algorithm
trials_tpe_1 = Trials()

best_tpe_1 = fmin(objective1,
    space=hp.uniform('x', -5, 5),
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials_tpe_1)


#Gaussian mixture function optimization with random search
trials_rnd_1 = Trials()

best_rnd_1 = fmin(objective1,
    space=hp.uniform('x', -5, 5),
    algo=rand.suggest,
    max_evals=1000,
    trials=trials_rnd_1)


#Piecewise linear function optimization with tpe algorithm
trials_tpe_2 = Trials()

best_tpe_2 = fmin(objective2,
    space=hp.uniform('x', -5, 5),
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials_tpe_2)

#Piecewise linear function optimization with random search
trials_rnd_2 = Trials()

best_rnd_2 = fmin(objective2,
    space=hp.uniform('x', -5, 5),
    algo=rand.suggest,
    max_evals=1000,
    trials=trials_rnd_2)

print('Time taken : {:.2f}s'.format(time.time() - start))
#general 

result = pd.DataFrame(index=['GM + TPE', 'GM + Random search','PL + TPE','PL + Random search'],columns=['trial'])
result['trial'][0]=trials_tpe_1
result['trial'][1]=trials_rnd_1
result['trial'][2]=trials_tpe_2
result['trial'][3]=trials_rnd_2

result['best loss']=result['trial'].apply(lambda x: x.best_trial['result']['loss'])
result['best x']=result['trial'].apply(lambda x: x.best_trial['result']['x'])
result['time taken (s)']=result['trial'].apply(lambda x: round(x.results[-1]['eval_time']-x.results[0]['eval_time'],2))
#trial id corresponds to the trial number when executed sequentially
result['best iteration']=result['trial'].apply(lambda x: x.best_trial['tid'])
result['time to best iteration']=result['trial'].apply(lambda x: round(x.best_trial['result']['eval_time']-x.results[0]['eval_time'],2))

result['real min loss']=gaussian_mixture(np.linspace(-5,5,50000)).min()
result['real min loss'][2:4]=min(piecewise_linear(np.linspace(-5,5,50000)))
result['real best x']=np.linspace(-5,5,50000)[gaussian_mixture(np.linspace(-5,5,50000)).argmin()]
result['real best x'][2:4]=np.linspace(-5,5,50000)[np.argmin(piecewise_linear(np.linspace(-5,5,50000)))]

result=result.drop('trial',axis=1)
result.head(4)
#statistically signifiant difference between tpe and random search
def get_best_loss(function,algorithm):
    def obj(x):
        return {
            'loss': function(x),
            'x':x,
            'status': STATUS_OK,
            'eval_time': time.time()
            }
    t=Trials()
    best = fmin(obj,
        space=hp.uniform('x', -5, 5),
        algo=algorithm,
        max_evals=1000,
        trials=t)
    return(t.best_trial['result']['loss'])

tpe_=[]
rnd_=[]
start=time.time()
n=100
for i in range(n):
    tpe_.append(get_best_loss(piecewise_linear,tpe.suggest))
    rnd_.append(get_best_loss(piecewise_linear,rand.suggest))
print('Time taken: {:.2f}min'.format((time.time()-start)/60))

s_tpe=n/(n-1)*np.var(tpe_)
s_rnd=n/(n-1)*np.var(rnd_)

avg_tpe=np.mean(tpe_)
avg_rnd=np.mean(rnd_)

conf_tpe=(avg_tpe-3.390*s_tpe/np.sqrt(n),avg_tpe+3.390*s_tpe/np.sqrt(n))
conf_rnd=(avg_rnd-3.390*s_rnd/np.sqrt(n),avg_rnd+3.390*s_rnd/np.sqrt(n))

plt.figure(figsize=(9,6))
xrange=min([min(rnd_),min(tpe_)]),max([max(rnd_),max(tpe_)])
plt.hist(rnd_,bins=100,range=xrange,label='random search losses',alpha=0.8)
plt.hist(tpe_,bins=100,range=xrange,label='TPE losses',alpha=0.8)
plt.legend()
plt.title('Losses distribution')
plt.show()

print('Best loss 99.9% confidence interval of TPE algorithm: [{:.7f} , {:.7f}]'.format(*conf_tpe))
print('Best loss 99.9% confidence interval of Random search algorithm: [{:.7f} , {:.7f}]'.format(*conf_rnd))


#focus on evolution: how does the algorithm sample their values?

from utils import animation_plot

#Random search on the gaussian mixture loss
animation_plot(trials_rnd_1,gaussian_mixture)

#TPE algorithm on the gaussian mixture loss
animation_plot(trials_tpe_1,gaussian_mixture)

#Rnd algorithm on the piecewise linear loss
ani=animation_plot(trials_rnd_2,piecewise_linear)
ani.save('videos/random_search_pl.mp4')
#TPE algorithm on the piecewise linear loss
ani=animation_plot(trials_tpe_2,piecewise_linear)
ani.save('videos/tpe_pl.mp4')


# =============================================================================
# Classical ML
# We use one of scikit-learn examples -> the digits dataset
# =============================================================================
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import GridSearchCV

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.


def apply_logistic(hparams):
     
     apply_pca = hparams.pop('apply_pca')
     pca_n_components = int(hparams.pop('pca_n_components'))
     logistic = SGDClassifier(loss='log',max_iter=10000, tol=1e-5, random_state=0, **hparams)
     
     if(apply_pca):
          pca = PCA(n_components = pca_n_components)
          pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
     else: 
          pipe = Pipeline(steps=[('logistic', logistic)])
     
     digits = datasets.load_digits()
     X_digits = digits.data
     y_digits = digits.target
     
     scores = cross_val_score(pipe, X_digits, y_digits, cv=5, scoring='accuracy')
     
     return(1-np.mean(scores))

def objective(hparams):
    return {
        'loss': apply_logistic(hparams),
        'x':hparams,
        'status': STATUS_OK,
        'eval_time': time.time()
        }
    
#search space definition. We keep log loss otherwise the algorithm is no longer a logistic regression classifier anymore
space = {'alpha':hp.uniform('alpha',0.0001,5),
        'l1_ratio': hp.uniform('l1_ratio',0,1),
        'apply_pca':hp.choice('apply_pca',[0,1]),
        'pca_n_components':hp.quniform('pca_n_components',1,64,2)}
#tpe
trials_tpe=Trials()

start=time.time()
best = fmin(objective,
   space=space,
   algo=tpe.suggest,
   max_evals=500,
   trials=trials_tpe)

print('Time taken by TPE: {:.2f}min'.format((time.time()-start)/60))
print('Best TPE score : {:.2f}'.format(100*(1-trials_tpe.best_trial['result']['loss'])))
print('Best TPE hparams :')
print(best)

#random search
space = {'alpha':hp.uniform('alpha',0.0001,5),
        'l1_ratio': hp.uniform('l1_ratio',0,1),
        'apply_pca':hp.choice('apply_pca',[0,1]),
        'pca_n_components':hp.quniform('pca_n_components',1,64,2)}

trials_rnd=Trials()

start=time.time()
best = fmin(objective,
   space=space,
   algo=rand.suggest,
   max_evals=500,
   trials=trials_rnd)

print('Time taken by Random search: {:.2f}min'.format((time.time()-start)/60))
print('Best Random search score : {:.2f}'.format(100*(1-trials_rnd.best_trial['result']['loss'])))
print('Best Random search hparams :')
print(best)


# Bonus, grid search: Parameters of pipelines can be set using ‘__’ separated parameter names:
# We will try to be realistic about grid search by not putting as much choice as within the hyperopt framework 

#1. with pca

start=time.time()

param_grid = {
    'pca__n_components':[5,10,20,30,40,50,64], #reasonable assumption
    'logistic__alpha': np.logspace(-4, 1, 20), #discrete with 20 values instead of continuous
    'logistic__l1_ratio': np.linspace(0,1,20)  #discrete with 20 values instead of continuous
}

logistic = SGDClassifier(loss='log',max_iter=10000, tol=1e-5, random_state=0)
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

search = GridSearchCV(pipe, param_grid, iid=False, cv=5,
                      return_train_score=False)
search.fit(X_digits, y_digits)

best_score = search.best_score_
best_params = search.best_params_

#2. without pca
param_grid = {
    'logistic__alpha': np.logspace(-4, 1, 20), #discrete with 20 values instead of continuous
    'logistic__l1_ratio': np.linspace(0,1,20)  #discrete with 20 values instead of continuous
}

logistic = SGDClassifier(loss='log', max_iter=10000, tol=1e-5, random_state=0)
pipe = Pipeline(steps=[('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

search = GridSearchCV(pipe, param_grid, iid=False, cv=5,
                      return_train_score=False)
search.fit(X_digits, y_digits)

if(search.best_score_<best_score):
     best_params=search.best_params_
     best_score=search.best_score_

print('Time taken by grid search: {:.2f}min'.format((time.time()-start)/60))
print('Best grid search score : {:.2f}'.format(100*best_score))
print('Best grid search hparams :')
print(best)

# =============================================================================
# Analysis of the scores
# =============================================================================
def get_values(trial,*keys):
     res={}
     for key in keys:
          res[key]=[]
     #res={'alpha':[],'l1_ratio':[],'apply_pca':[],'pca_n_components':[],'score':[]}
     for t in trial:
          vals=t['misc']['vals']
          for key in keys:
               if key=='score':
                    res['score'].append(1-t['result']['loss'])
               else:
                    res[key].append(vals[key][0])
     return(res)

vals_tpe=get_values(trials_tpe,'alpha','l1_ratio','apply_pca','pca_n_components','score')
vals_rnd=get_values(trials_rnd,'alpha','l1_ratio','apply_pca','pca_n_components','score')

vals_tpe=pd.DataFrame(vals_tpe)
vals_tpe.iloc[:,0:4].plot(kind='kde',subplots=True, sharex=False, layout=(2,2))
plt.title('Sampling distribution of the TPE algorithm')
plt.tight_layout()

vals_rnd=pd.DataFrame(vals_rnd)
vals_rnd.iloc[:,0:4].plot(kind='kde',subplots=True, sharex=False,layout=(2,2))
plt.title('Sampling distribution of the random search algorithm')
plt.tight_layout()

highscores_tpe=vals_tpe[vals_tpe['score']>=0.99*vals_tpe.score.max()].reset_index()
highscores_tpe=highscores_tpe.rename({'index':'iteration'},axis=1)
highscores_rnd=vals_rnd[vals_rnd['score']>=0.99*vals_rnd.score.max()].reset_index()
highscores_rnd=highscores_rnd.rename({'index':'iteration'},axis=1)

plt.hist(highscores_tpe.score,bins=15, label='TPE high scores')
plt.hist(highscores_rnd.score,bins=15, label='Random search high scores')
plt.legend()
plt.title('Distribution of high scores')
