# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:58:55 2018

@author: Alexis
"""

import tensorflow as tf
import math
import functools


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
# Model for MNIST Dataset. Structure : conv + conv + conv 
#                                       +dropout + dense
#
# Default parameters consistently reach more than 99% in accuracy in 5 epochs 
# =============================================================================

# MNIST with fixed structure conv + conv + conv + relu + dpout + softmax 
class MNISTModel:

    def __init__(self,batch,target,step,batch_size,pkeep,
                 min_lr_ratio = 30,max_lr = 0.003,decay_step = 2000,decay_rate = 1/math.e,
                 optimizer_algo={'optimizer':tf.train.AdamOptimizer,'beta1':0.9,'beta2':0.99,'epsilon':1e-08},
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
        self.optimizer=optimizer_algo['optimizer']
        self.optimizer_algo={key: value for key, value in optimizer_algo.items() if key!='optimizer'}
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
        train_step = self.optimizer(lr,**self.optimizer_algo).minimize(cross_entropy)
        
        return(train_step, cross_entropy)
    
    @lazy_property
    def accuracy(self):
        Y = self.predict
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.target, 1))
        return(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
      

# =============================================================================
# More flexible model for MNIST dataset : tuning the number of layers and their size
# 
# =============================================================================


class MoreFlexibleMNISTModel:

    def __init__(self,batch,
                 target,
                 step,
                 batch_size,
                 pkeep,
                 conv_layers,
                 dense_layers,
                 optimizer_hparams,
                 **kwargs):           
        
        self.go_flag=True
        
        #data
        self.batch=batch
        self.target=target
        self.step=step
        self.batch_size=batch_size
        #Structure parameters
        
        #Weights 
        self.conv_depth=conv_layers.get('conv_depth')
        self.conv_weights={}
        
        #building conv layers stack 
        for i in range(1,self.conv_depth+1):
          if i == 1 :
            K_init = int(conv_layers.get('channels').get('conv_channels_1'))
            C_init = int(conv_layers.get('filters').get('conv_filter_1'))

            self.conv_weights['C'+str(i)] = (tf.Variable(tf.truncated_normal([C_init, C_init, 1, K_init], stddev=0.1)),
                                             tf.Variable(tf.constant(0.1, tf.float32, [K_init])))
            last_n_channels = K_init 
            last_n_filters = C_init 
          else:
            K = int(last_n_channels*conv_layers.get('channels').get('conv_channels_mult_'+str(i)))
            C = int(last_n_filters*conv_layers.get('filters').get('conv_filter_mult_'+str(i)))

            if (K<1 or C<1 or C>100):
                self.go_flag=False
                break
            else:
              self.conv_weights['C'+str(i)] = (tf.Variable(tf.truncated_normal([C, C, last_n_channels, K], stddev=0.1)),
                                          tf.Variable(tf.constant(0.1, tf.float32, [K])))
              last_n_channels = K
              last_n_filters = C
              
        self.dense_depth=dense_layers.get('dense_depth')
        self.dense_weights={}
        
        #building dense layers stack      
        for i in range(1,self.dense_depth+1):
            if i == 1 :
                N_init = int(dense_layers.get('dense_units_1'))
                self.dense_weights['D'+str(i)]=(tf.Variable(tf.truncated_normal([7 * 7 * last_n_channels, N_init], stddev=0.1)),
                                                tf.Variable(tf.constant(0.1, tf.float32, [N_init])))
                last_n_dense = N_init

            else:
                N = int(last_n_dense*dense_layers.get('dense_units_mult_'+str(i)))

                if (N<1) or (N>2000): #N>2000 is way too big 
                     self.go_flag=False
                     break
                else:
                    self.dense_weights['D'+str(i)]=(tf.Variable(tf.truncated_normal([last_n_dense, N], stddev=0.1)),
                                                tf.Variable(tf.constant(0.1, tf.float32, [N])))
                    last_n_dense = N

        if self.go_flag:
            self.Wout = tf.Variable(tf.truncated_normal([last_n_dense, 10], stddev=0.1))
            self.Bout = tf.Variable(tf.constant(0.1, tf.float32, [10]))

            #keep the last size for the reshape operation
            self.M = last_n_channels

            #Dropout
            self.pkeep = pkeep # Dropout probability

            #Optimization parameters
            self.max_lr = optimizer_hparams.get('max_lr')
            self.min_lr = self.max_lr/optimizer_hparams.get('min_lr_ratio')
            self.decay_step = int(optimizer_hparams.get('decay_step'))
            self.decay_rate = optimizer_hparams.get('decay_rate')
            self.optimizer=optimizer_hparams.get('optimizer_algo').get('optimizer')
            self.optimizer_hparams=optimizer_hparams.get('optimizer_algo').get('optimizer_params')

            #Graph initialization
            self.predict
            self.optimize
            self.accuracy
            self.prediction_with_logits
            
    @lazy_property
    def prediction_with_logits(self):
        
        X = tf.reshape(self.batch, [-1,28,28,1]) #[batch_size] + spatial_format + [in_channels]
        
        #Conv layers stack
        for i in range(1,self.conv_depth+1):
            #stride is defined to reduce the size to 7*7 at last layer
            if self.conv_depth==1:
              stride = 4
            if (self.conv_depth>1) & (self.conv_depth-i<2):
              stride = 2
            if (self.conv_depth>1) & (self.conv_depth-i>1):
              stride=1

            X = tf.nn.relu(tf.nn.conv2d(X, self.conv_weights['C'+str(i)][0], strides=[1, stride, stride, 1], padding='SAME') + self.conv_weights['C'+str(i)][1])

        #reshape the output from the third convolution for the fully connected layer
        Y = tf.reshape(X, shape=[-1, 7 * 7 * self.M])       

        #Dense layers stack
        
        for i in range(1,self.dense_depth+1):
            Y = tf.nn.relu(tf.matmul(Y, self.dense_weights['D'+str(i)][0]) + self.dense_weights['D'+str(i)][1])
            Y = tf.nn.dropout(Y, self.pkeep)
 
        Ylogits = tf.matmul(Y, self.Wout) + self.Bout
       
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
       