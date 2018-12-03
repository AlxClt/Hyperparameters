# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:23:54 2018

@author: Alexis
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from MNIST_Models import MNISTModel, MoreFlexibleMNISTModel
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, STATUS_FAIL, Trials
import pickle
import pandas as pd
import pprint
# =============================================================================
# First step: only tuning the optimisation hyperparameters + dropout
# =============================================================================

N_EPOCH=5
BATCH_SIZE=64
MAX_STEPS = int(N_EPOCH*60000//BATCH_SIZE)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

"""
Reference parameters
"""
def apply_reference_mnist_model():

    batch = tf.placeholder(tf.float32, [None, 784])
    target = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    batch_size = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    
    pkeep_=0.75
    
    model = MNISTModel(batch, target, step, batch_size,pkeep)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(MAX_STEPS):
        images, labels = mnist.train.next_batch(BATCH_SIZE)
        _,loss = sess.run(model.optimize, {batch: images, target: labels,
                                           step:i, batch_size:BATCH_SIZE, 
                                           pkeep:pkeep_})
        if i%100==0:
            images, labels = mnist.test.images, mnist.test.labels
            accuracy = sess.run(model.accuracy, {batch: images, target: labels, 
                                         step:i, batch_size:BATCH_SIZE, 
                                         pkeep:1.0})
    
            print('Progress: {:.2f}% - Test accuracy {:6.2f}%'.format(100*i/MAX_STEPS ,100 * accuracy))
    
    images, labels = mnist.test.images, mnist.test.labels
    accuracy = sess.run(model.accuracy, {batch: images, target: labels, 
                                         step:i, batch_size:BATCH_SIZE, 
                                         pkeep:1.0})
    
    print('Progress: 100% - Test accuracy {:6.2f}%'.format( 100 * accuracy))

    sess.close()
    
    return(accuracy)

target_accuracy=apply_reference_mnist_model()


"""
Bayesian optimisation
"""

def apply_mnist_model(hparams):

    batch = tf.placeholder(tf.float32, [None, 784])
    target = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    batch_size = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    
    pkeep_=hparams.pop('pkeep')
    
    model = MNISTModel(batch, target, step, batch_size,pkeep, **hparams['optimizer_hparams'])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(MAX_STEPS):
        images, labels = mnist.train.next_batch(BATCH_SIZE)
        _,loss = sess.run(model.optimize, {batch: images, target: labels,
                                           step:i, batch_size:BATCH_SIZE, 
                                           pkeep:pkeep_})
              
    images, labels = mnist.test.images, mnist.test.labels
    accuracy = sess.run(model.accuracy, {batch: images, target: labels, 
                                         step:i, batch_size:BATCH_SIZE, 
                                         pkeep:1.0})
    
    print('Progress: 100% - Test accuracy {:6.2f}%'.format( 100 * accuracy))

    sess.close()
    
    #be careful to return 1-accuracy since we wish to minimize the score
    return(1-accuracy)

    
space = {'optimizer_hparams':
         hp.choice('optimizer',[
                        {'optimizer_algo':
                             {'optimizer':tf.train.AdamOptimizer,
                              'beta1':hp.uniform('beta1',0.8,0.99),'beta2':hp.uniform('beta2',0.9,0.9999),'epsilon':hp.uniform('epsilon',0.01,1)},
               
                         'max_lr':hp.loguniform('learning_rate_adam', np.log(0.001), np.log(0.2)),
                         'min_lr_ratio':hp.choice('min_lr_ratio_adam',[1,10,20,30,40,50,75,100,200,500,1000]),
                         'decay_step':hp.quniform('decay_step_adam',200,MAX_STEPS,50), 
                         'decay_rate':hp.uniform('decay_rate_adam',0,1)},
                        
                        {'optimizer_algo':
                             {'optimizer':tf.train.RMSPropOptimizer,
                              'decay':hp.uniform('decay',0.1,0.99),'momentum':hp.uniform('momentum',0.1,0.99),'epsilon':1e-10,},
                
                         'max_lr':hp.loguniform('learning_rate_rmsprop', np.log(0.001), np.log(0.2)),
                         'min_lr_ratio':hp.choice('min_lr_ratio_rmsprop',[1,10,20,30,40,50,75,100,200,500,1000]),
                         'decay_step':hp.quniform('decay_step_rmsprop',200,MAX_STEPS,50), 
                         'decay_rate':hp.uniform('decay_rate_rmsprop',0,1)},
                              
                        {'optimizer_algo':
                             {'optimizer':tf.train.GradientDescentOptimizer},
                             
                         'max_lr':hp.loguniform('learning_rate_gdopt', np.log(0.001), np.log(0.2)),
                         'min_lr_ratio':hp.choice('min_lr_ratio_gdopt',[1,10,20,30,40,50,75,100,200,500,1000]),
                         'decay_step':hp.quniform('decay_step_gdopt',200,MAX_STEPS,50), 
                         'decay_rate':hp.uniform('decay_rate_gdopt',0,1)}
                   ]),
          'pkeep':hp.uniform('pkeep',0.3,0.9)}

def objective(hparams):
    return({
        'loss': apply_mnist_model(hparams),
        'hparams':hparams,
        'status': STATUS_OK,
        'eval_time': time.time()
        })
    
trials_tpe = Trials()

best_tpe = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=500,
                trials=trials_tpe)

#results analysis -> loading from data/trials_tpe_mnis_step1 because we made it run
#on google collab notebook

with open('data/trials_tpe_mnist_step1','rb') as f:
    trials_tpe=pickle.load(f)

print('Best Accuracy: {:.2f}%'.format(100*(1-trials_tpe.best_trial['result']['loss'])))

def get_values(trial,*keys):
    res_={}
    for key in keys:
        res_[key]=[]
    for t in trial:
        vals=t['misc']['vals']
        for key in keys:
            if key=='score':
                res_['score'].append(1-t['result']['loss'])
            else:
                if len(vals[key])>0:
                    res_[key].append(vals[key][0])
                else:
                    res_[key].append(np.nan)
        
    return(res_)
    
#Optimizer
algorithm_choice = pd.DataFrame(get_values(trials_tpe,'optimizer','score'))

#We need to set the labels, since hp.choice returns the index of its choice, not the name of the algorithm

mapping = {0:'Adam',1:'RMSProp',2:'SGD'}
algorithm_choice.optimizer=algorithm_choice.optimizer.apply(lambda x : mapping[x])
algorithm_choice=algorithm_choice.groupby('optimizer').agg({'optimizer':'count','score':max})
algorithm_choice.rename({'optimizer':'count','score':'max score reached'})
print(algorithm_choice)

#optimizer's parameters
optimizers_params=pd.DataFrame(get_values(trials,'learning_rate_adam', 'min_lr_ratio_adam', 'decay_rate_adam', 'decay_step_adam', 'beta1', 'beta2',
                                         'pkeep','score'))

optimizers_params.plot(kind='kde',subplots=True, sharex=False, layout=(4,4),title='Hyperparameters for Adam optimizer',figsize=(16,16))
plt.show()

print('Best configuration: \n')
pprint.PrettyPrinter(indent=4).pprint(trials.best_trial['misc']['vals'])

#lr vs score
plt.figure(figsize=(16,9))
plt.scatter(optimizers_params[optimizers_params.score>0.9].learning_rate_adam,optimizers_params[optimizers_params.score>0.9].score)
plt.title('Learning rate vs score')
plt.xlabel('Initial learning rate')
plt.ylabel('Score')
plt.show()

#dropout
dpout = pd.DataFrame(get_values(trials_tpe,'pkeep','score'))

dpout['pkeep'].plot(kind='kde',title='Dropout probability sampling distribution',figsize=(9,6))
plt.show()


# =============================================================================
# Tuning the whole structure
# =============================================================================


N_EPOCH=5
BATCH_SIZE=64
MAX_STEPS = int(N_EPOCH*60000//BATCH_SIZE)

#Bounds for the search space


space = {
        'conv_layers':hp.choice('conv_layers',[
                {'conv_depth':1, 
                 'channels':{
                         'conv_channels_1':hp.quniform('conv_channels_1_1',2,12,2)
                         }, 
                 'filters': {
                         'conv_filter_1':hp.quniform('conv_filter_1_1',2,8,2)
                         }},
                {'conv_depth':2, 
                 'channels':{
                         'conv_channels_1':hp.quniform('conv_channels_1_2',2,12,2),
                         'conv_channels_mult_2':hp.uniform('conv_channels_mult_2_2',1,3)
                         }, 
                 'filters': {
                         'conv_filter_1':hp.quniform('conv_filter_1_2',2,8,2),
                         'conv_filter_mult_2':hp.uniform('conv_filter_mult_2_2',1,3)
                         }},
                {'conv_depth':3, 
                 'channels':{
                         'conv_channels_1':hp.quniform('conv_channels_1_3',2,12,2),
                         'conv_channels_mult_2':hp.uniform('conv_channels_mult_2_3',1,3),
                         'conv_channels_mult_3':hp.uniform('conv_channels_mult_3_3',1,3)
                         }, 
                 'filters': {
                         'conv_filter_1':hp.quniform('conv_filter_1_3',2,8,2),
                         'conv_filter_mult_2':hp.uniform('conv_filter_mult_2_3',1,3),
                         'conv_filter_mult_3':hp.uniform('conv_filter_mult_3_3',1,3)
                         }},
                {'conv_depth':4, 
                 'channels':{
                         'conv_channels_1':hp.quniform('conv_channels_1_4',2,12,2),
                         'conv_channels_mult_2':hp.uniform('conv_channels_mult_2_4',1,3),
                         'conv_channels_mult_3':hp.uniform('conv_channels_mult_3_4',1,3),
                         'conv_channels_mult_4':hp.uniform('conv_channels_mult_4_4',1,3)
                         }, 
                 'filters': {
                         'conv_filter_1':hp.quniform('conv_filter_1_4',2,8,2),
                         'conv_filter_mult_2':hp.uniform('conv_filter_mult_2_4',1,3),
                         'conv_filter_mult_3':hp.uniform('conv_filter_mult_3_4',1,3),
                         'conv_filter_mult_4':hp.uniform('conv_filter_mult_4_4',1,3)
                         }}
                ]),
        
        'dense_layers':hp.choice('dense_layers',[
                {'dense_depth':1, 
                 'dense_units_1':hp.quniform('dense_units_1_1',50,500,50)},
                {'dense_depth':2, 
                 'dense_units_1':hp.quniform('dense_units_1_2',10,500,10), 
                                 'dense_units_mult_2': hp.uniform('dense_units_mult_2_2',1,3)},
                {'dense_depth':3, 
                 'dense_units_1':hp.quniform('dense_units_1_3',50,500,50), 
                                 'dense_units_mult_2':hp.uniform('dense_units_mult_2_3',1,3),
                                 'dense_units_mult_3':hp.uniform('dense_units_mult_3_3',1,3)},
                ]),
        
        'optimizer_hparams':{
                'optimizer_algo':{'optimizer':tf.train.AdamOptimizer,
                                  'optimizer_params':{'beta1':0.9,
                                                      'beta2':0.99,
                                                      'epsilon':1e-08}},
                        
                'max_lr':hp.loguniform('learning_rate_adam', np.log(0.001), np.log(0.2)),
                'min_lr_ratio':hp.choice('min_lr_ratio_adam',[1,10,20,30,40,50,75,100,200,500,1000]),
                'decay_step':hp.quniform('decay_step_adam',200,MAX_STEPS,50), 
                'decay_rate':hp.uniform('decay_rate_adam',0,1)},

        'pkeep':hp.uniform('pkeep',0.3,1)
        }
                
                
from hyperopt.pyll.stochastic import sample
testspace=sample(space)
print(sample(testspace))

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


def apply_mnist_model(hparams):

    batch = tf.placeholder(tf.float32, [None, 784])
    target = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    batch_size = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    
    pkeep_=hparams.pop('pkeep')
    
    model = MoreFlexibleMNISTModel(batch, target, step, batch_size,pkeep, **hparams)

    if model.go_flag:
    
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
            
        for i in range(MAX_STEPS):
                        
            images, labels = mnist.train.next_batch(BATCH_SIZE)
            _,loss = sess.run(model.optimize, {batch: images, target: labels,
                                               step:i, batch_size:BATCH_SIZE, 
                                               pkeep:pkeep_})
       
            if i %100==0:
                images, labels = mnist.test.images, mnist.test.labels
                accuracy = sess.run(model.accuracy, {batch: images, target: labels, 
                                             step:i, batch_size:BATCH_SIZE, 
                                             pkeep:1.0})
        
                print('Progress: {:.2f}% - Test accuracy {:6.2f}%'.format(100*i/MAX_STEPS, 100 * accuracy))
    
        images, labels = mnist.test.images, mnist.test.labels
        accuracy = sess.run(model.accuracy, {batch: images, target: labels, 
                                             step:i, batch_size:BATCH_SIZE, 
                                             pkeep:1.0})
        
        print('Progress: 100% - Test accuracy {:6.2f}%'.format( 100 * accuracy))
    
        sess.close()
        
        #be careful to return 1-accuracy since we wish to minimize the score
        return(1-accuracy, STATUS_OK)
        
    else:
        print('invalid configuration')
        return(0, STATUS_FAIL)

def objective(hparams):
    
    loss, status = apply_mnist_model(hparams)
    
    return({
        'loss':loss ,
        'hparams':hparams,
        'status': status,
        'eval_time': time.time()
        })
    
trials_tpe = Trials()

best_tpe = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=200,
                trials=trials_tpe)

with open('data/trials_tpe_mnist_step2','rb') as f:
    trials=pickle.load(f)

print('Best Accuracy: {:.2f}%'.format(100*(1-trials.best_trial['result']['loss'])))

def get_values(trial,*keys):
    res_={}
    for key in keys:
        res_[key]=[]
    for t in trial:
        vals=t['misc']['vals']
        for key in keys:
            if key=='score':
                res_['score'].append(1-t['result']['loss'])
            else:
                if len(vals[key])>0:
                    res_[key].append(vals[key][0])
                else:
                    res_[key].append(np.nan)
        
    return(res_)


#Conv layers
conv_layers = pd.DataFrame(get_values(trials_tpe,'conv_layers','score'))

#We need to set the labels, since hp.choice returns the index of its choice, not the name of the algorithm
conv_layers.conv_layers=conv_layers.conv_layers+1

#visualize top score distribution
plt.figure(figsize=(9,6))
plt.scatter(conv_layers[conv_layers.score>0.95].conv_layers,conv_layers[conv_layers.score>0.95].score)
plt.title('Top scores (>0.95) per convolutional layers number')
plt.xlabel('Convolutional layers number')
plt.ylabel('Accuracy')
plt.show()

conv_layers=conv_layers.groupby('conv_layers').agg({'conv_layers':'count','score':max})
conv_layers=conv_layers.rename({'conv_layers':'count','score':'max score reached'},axis=1)
print(conv_layers)

#Dense layers
dense_layers = pd.DataFrame(get_values(trials_tpe,'dense_layers','score'))

#We need to set the labels, since hp.choice returns the index of its choice, not the name of the algorithm
dense_layers.dense_layers=dense_layers.dense_layers+1

#visualize top score distribution
plt.figure(figsize=(9,6))
plt.scatter(dense_layers[dense_layers.score>0.95].dense_layers,dense_layers[dense_layers.score>0.95].score)
plt.title('Top scores (>0.95) per dense layers number')
plt.xlabel('Dense layers number')
plt.ylabel('Accuracy')
plt.show()

dense_layers=dense_layers.groupby('dense_layers').agg({'dense_layers':'count','score':max})
dense_layers=dense_layers.rename({'dense_layers':'count','score':'max score reached'},axis=1)
print(dense_layers)


#optimizer's parameters
optimizers_params=pd.DataFrame(get_values(trials,'learning_rate_adam', 'min_lr_ratio_adam', 'decay_rate_adam', 'decay_step_adam',
                                         'pkeep','score'))

optimizers_params.plot(kind='kde',subplots=True, sharex=False, layout=(3,2),title='Hyperparameters for Adam optimizer',figsize=(16,16))
plt.show()

print('Best configuration: \n')
pprint.PrettyPrinter(indent=4).pprint(trials.best_trial['misc']['vals'])



# =============================================================================
# Trying to rach 99.5% accuracy (in progress)
# =============================================================================


#best config -> not reproducible because of the random initialzation of weghts and biais?

best_conf = {
        'conv_layers':
                {'conv_depth':3, 
                 'channels':{
                         'conv_channels_1':10.0,
                         'conv_channels_mult_2':2.1726153573086577,
                         'conv_channels_mult_3':2.277478274537393
                         }, 
                 'filters': {
                         'conv_filter_1':4.0,
                         'conv_filter_mult_2':1.9016499455362226,
                         'conv_filter_mult_3':1.126609590344377
                         }},
        
        'dense_layers':
                {'dense_depth':1, 
                 'dense_units_1':400.0}
                ,
        
        'optimizer_hparams':{
                'optimizer_algo':{'optimizer':tf.train.AdamOptimizer,
                                  'optimizer_params':{'beta1':0.9,
                                                      'beta2':0.99,
                                                      'epsilon':1e-08}},
                        
                'max_lr':0.001005214548988152,
                'min_lr_ratio':10, #careful, hp.choice
                'decay_step':2100, 
                'decay_rate':0.5484523903846402},

        'pkeep':0.49376913873300154
        }
           
plt.hist(np.random.normal(0.49376913873300154,0.1,1000))