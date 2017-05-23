# Copyright 2015 Matthieu Courbariaux

# This file is part of BinaryConnect.

# BinaryConnect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryConnect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryConnect.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import batch_norm
import binary_connect

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict
from utils.inputdata import *
from utils.Gates import Gates


if __name__ == "__main__":
    
    # BN parameters
    batch_size = 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .15
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 2048
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 120
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = 0. # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = 0.
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = True
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    LR_start = .001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    print('Loading MNIST dataset...')

    gates = Gates(force = True, image_size = 28, inputsize1 = 916, inputsize2=916, mask = False)
    gates.train.stack()
    gates.test.stack()
    gates.valid.stack()

    train_labels=gates.train.Y  
    train_images=gates.train.X
    test_labels=gates.test.Y   
    test_images=gates.test.X
    valid_labels=gates.valid.Y
    valid_images=gates.valid.X
    train_images = train_images.reshape(-1, 1, 28, 28)
    #np.putmask(train_images,train_images>0.35,1)
    #np.putmask(train_images,train_images<=0.35,0)
    valid_images = valid_images.reshape(-1, 1, 28, 28)
    #np.putmask(valid_images,valid_images>0.35,1)
    #np.putmask(valid_images,valid_images<=0.35,0)
    test_images = test_images.reshape(-1, 1, 28, 28)
    #np.putmask(test_images,test_images>0.35,1)
    #np.putmask(test_images,test_images<=0.35,0)  
    # for hinge loss
    train_labels = 2* train_labels - 1.
    valid_labels = 2* valid_labels - 1.
    test_labels = 2* test_labels - 1.

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)

    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = binary_connect.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = batch_norm.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
    
    mlp = binary_connect.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=7)      
                  
    mlp = batch_norm.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.identity)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_connect.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_connect.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    binary_connect.train(
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            200,
            train_images,train_labels,
            valid_images,valid_labels,
            test_images,test_labels)
    
    # store the weights value and bias value of each binaryDenselayer 
    from six.moves import cPickle as pickle
    print (lasagne.layers.get_all_params(mlp))
    para = lasagne.layers.get_all_param_values(mlp)
    weights = []
    means = []
    stds = []
    for i in range(0, 24, 6):
        weights.append(para[i])
        means.append(para[i+2])
        stds.append(para[i+3])


    # save the values to a pickle file in order to construct thr model of tensorflow framework
    save = {"weights": weights, "means": means, "invars": stds}
    pickle_files = "mnist_bnn_paras.pkl"
    with open(pickle_files, 'wb') as f:
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
