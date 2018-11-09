# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:58:55 2018

@author: Alexis
"""

import tensorflow as tf
import math
import numpy as np
import functools



# =============================================================================
# Piecewise linear function
# =============================================================================
class piecewise:
    
    def __init__(self,domain=(-5,5),breaks=20,values=(0,1),seed=37):
        np.random.seed(seed)
        self.steps = np.linspace(domain[0]-0.01,domain[1]+0.05,breaks)
        self.values = np.random.uniform(values[0],values[1],breaks)
    
    def __call_one(self,x):
        idx=np.where(self.steps < x)[0][-1]
        return(self.__step(x,self.values[idx],self.values[idx+1],self.steps[idx],self.steps[idx+1]))
     
    def __step(self,x,y0,y1,x0,x1):
        a = (y1-y0)/(x1-x0)
        b = y0-a*x0
        return(a*x+b)
        
    def __call__(self,x):
        try:
            return([self.__call_one(y) for y in x])
        except TypeError:
            return(self.__call_one(x))




# =============================================================================
# decorator function wrapping the property in order to evaluate it in a lazy way: 
# see https://danijar.com/structuring-your-tensorflow-models/
# =============================================================================


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# =============================================================================
# Model for MNIST Dataset. Structure : conv + conv (+ conv omitted for runtime performance)
#                                       +dropout + dense
#
# Default parameters reach 96.36% in accuracy in one epoch (12min)
# =============================================================================

# MNIST with fixed structure conv + conv + conv + relu + dpout + softmax 
class MNISTModel:

    def __init__(self,batch,target,step,batch_size,pkeep,
                 min_lr_ratio = 30,max_lr = 0.003,decay_step = 2000,decay_rate = 1/math.e,
                 optimizer_hparams={'optimizer':tf.train.AdamOptimizer,'beta1':0.9,'beta2':0.999,'epsilon':1e-08},
                 K=6,L=12,M=24,N=200,C1=6,C2=5,C3=4):           
        
        #data
        self.batch=batch
        self.target=target
        self.step=step
        self.batch_size=batch_size
        #Structure parameters
        
        #Weights 
        self.W1 = tf.Variable(tf.truncated_normal([C1, C1, 1, K], stddev=0.1))  
        self.B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        
        self.W2 = tf.Variable(tf.truncated_normal([C2, C2, K, L], stddev=0.1))
        self.B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
        
        self.W3 = tf.Variable(tf.truncated_normal([C3, C3, L, M], stddev=0.1))
        self.B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
        
        self.W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
        self.B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        
        self.W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
        self.B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        
        self.W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
        self.B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
        
        #keep the last size for the reshape operation
        self.M = M
        
        #Dropout
        self.pkeep = pkeep # Dropout probability

        #Optimization parameters
        self.min_lr = max_lr/min_lr_ratio
        self.max_lr = max_lr
        self.decay_step = int(decay_step)
        self.decay_rate = decay_rate
        self.optimizer=optimizer_hparams.pop('optimizer')
        self.optimizer_hparams=optimizer_hparams
        #Graph initialization
        self.predict
        self.optimize
        self.accuracy
        self.prediction_with_logits

    @lazy_property
    def prediction_with_logits(self):
        
        X = tf.reshape(self.batch, [-1,28,28,1]) #[batch_size] + spatial_format + [in_channels]
                
        # The model
        stride = 1  # output is 28x28
        Y1 = tf.nn.relu(tf.nn.conv2d(X, self.W1, strides=[1, stride, stride, 1], padding='SAME') + self.B1)
        
        stride = 2  # output is 14x14
        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, self.W2, strides=[1, stride, stride, 1], padding='SAME') + self.B2)
        
        stride = 2  # output is 7x7
        Y3 = tf.nn.relu(tf.nn.conv2d(Y2, self.W3, strides=[1, stride, stride, 1], padding='SAME') + self.B3)
        
        #reshape the output from the third convolution for the fully connected layer
        YY = tf.reshape(Y3, shape=[-1, 7 * 7 * self.M])       
        
       
        Y4 = tf.nn.relu(tf.matmul(YY, self.W4) + self.B4)
        YY4 = tf.nn.dropout(Y4, self.pkeep)
        
        Ylogits = tf.matmul(YY4, self.W5) + self.B5
        
        return(Ylogits)
     
    @lazy_property
    def predict(self):
        return(tf.nn.softmax(self.prediction_with_logits))
        
    @lazy_property
    def optimize(self):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= self.prediction_with_logits,
                                                                labels=self.target)     
        cross_entropy = tf.reduce_mean(cross_entropy)*tf.cast(self.batch_size,tf.float32)
        
        lr = self.min_lr +  tf.train.exponential_decay(self.max_lr, self.step, self.decay_step, self.decay_rate)
        train_step = self.optimizer(lr,**self.optimizer_hparams).minimize(cross_entropy)
        
        return(train_step, cross_entropy)
    
    @lazy_property
    def accuracy(self):
        Y = self.predict
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.target, 1))
        return(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
       