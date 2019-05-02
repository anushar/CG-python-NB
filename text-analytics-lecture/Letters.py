#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#==============================================================================
# Convolutional Neural Network on MNIST digits data
#==============================================================================

#from keras.models import Sequential

import pickle as cPickle
import random
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import confusion_matrix



#==============================================================================
# Misc functions
#==============================================================================

# Load data
def load_dataset(dataset_path = '/Users/Mia/Downloads/gzip/letters/emnist.pkl.gz'):
    import pickle
    import gzip
    file                = gzip.open(dataset_path, 'rb')
    data                = pickle.load(file, encoding='latin1')
    X_train, y_train    = data[0]
    X_test, y_test      = data[1]
	
    # Reshape				
    X_train             = X_train.reshape((-1, 1, 28, 28))
    X_test              = X_test.reshape((-1, 1, 28, 28))
    
    # Change dataformat
    y_train             = y_train.astype(np.uint8)
    y_test              = y_test.astype(np.uint8)
    return X_train, y_train, X_test, y_test

# Plot confusion matrix
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def plot_confusion_matrix(y_test, preds, cmap=plt.cm.Blues):
    
    # Setup    
    classes     = np.unique(y_test)
    tick_marks  = np.arange(len(classes))
    cmap        = plt.cm.Blues
    
    # Confusion matrix
    cm          = confusion_matrix(y_test, preds)    
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh      = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return cm
#==============================================================================
# 
#==============================================================================

# Load data    
X_train, y_train, X_test, y_test = load_dataset()

y_train = np.array(y_train - 1)
y_test = np.array(y_test - 1)

small_train_idx = random.sample(range(len(X_train)), 1000)
small_test_idx = random.sample(range(len(X_test)), 1000)

X_train_s 	= X_train[small_train_idx]
y_train_s 	= y_train[small_train_idx]
X_test_s 	= X_test[small_test_idx]
y_test_s 	= y_test[small_test_idx]

# Example image
plt.imshow(X_train[0][0], cmap=cm.binary)

# Convolutional NN
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 28, 28),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=26,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.02,
    update_momentum=0.9,
    max_epochs=50,
    verbose=1,
    )
    
# Train the network
nn      = net1.fit(X_train_s, y_train_s)
#nn      = net1.fit(X_train, y_train)

# Test CNN
preds   = net1.predict(X_test)

# Confusion matrix
plot_confusion_matrix(y_test, preds)


nn      = net1.fit(X_train, y_train)
# Test CNN
preds   = net1.predict(X_test)

# Confusion matrix
plot_confusion_matrix(y_test, preds)


def save_data(obj, save_file):
    cPickle.dump(obj, open(save_file, 'wb'))
    

visualize.plot_conv_weights(net1.layers_['conv2d1'])
visualize.plot_conv_weights(net1.layers_['conv2d2'])


# =============================================================================
# Architecture 2
# =============================================================================
# Convolutional NN
net2 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 28, 28),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(4, 4),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),   
    # layer conv2d2
    conv2d2_num_filters=64,
    conv2d2_filter_size=(4, 4),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2), 
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.05,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )
    
# Train the network
nn2      = net2.fit(X_train_s, y_train_s)

# Test CNN
preds2   = net2.predict(X_test)

# Confusion matrix
plot_confusion_matrix(y_test, preds2)

visualize.plot_conv_weights(net2.layers_['conv2d1'])
visualize.plot_conv_weights(net2.layers_['conv2d2'])