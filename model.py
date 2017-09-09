# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:20:15 2017

@author: Chathuranga
"""

import tensorflow as tf;
import os as os
import struct as struct;
import numpy as np;
#set current working directory
os.chdir('E:\Data\Projects\mnist-tensorflow')

def getImageData(path='./data/t10k-images.idx3-ubyte'):
    '''
    interprets mnist image files and returns a 3 - dimensional vector.
    '''

    arr = [];
    with open(path, "rb") as file:
        magic, size, rows, cols = struct.unpack('>iiii', file.read(16))

        arr = np.fromfile(file, dtype=np.dtype(np.uint8))    
        return np.reshape(arr, (size, rows, cols));

def getLabelData(path='./data/t10k-labels.idx1-ubyte'):
    '''
    interprets mnist label data and returns a 1 - dimensional array
    '''
    
    arr = [];
    with open(path, "rb") as file:
        magic, size = struct.unpack('>ii', file.read(8))
        arr = np.fromfile(file, dtype=np.dtype(np.uint8))    
        return np.reshape(arr,(size, 1));



def prepareDataSets():
    '''
    load & prepare mnist datasets. returns train and test sets.   
    '''
    
    
    train_X = getImageData('./data/train-images.idx3-ubyte');
    train_Y = getLabelData('./data/train-labels.idx1-ubyte');
    test_X = getImageData('./data/t10k-images.idx3-ubyte');
    test_Y = getLabelData('./data/t10k-labels.idx1-ubyte');


       
    train_X_prep = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]);
    test_X_prep = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]);

    with tf.Session() as sess:
        
        train_Y_prep = sess.run(tf.one_hot(train_Y.reshape(-1), 10, axis=1));
        test_X_prep = sess.run(tf.one_hot(test_Y.reshape(-1), 10, axis=1));

    return train_X, train_Y, test_X, test_Y;

def initializeWeights():
    '''
    initialize the weights for the 3 layer neueral network
    '''
    
    tf.reset_default_graph();
    W1 = tf.get_variable("W1", shape=[10, 784], initializer=tf.contrib.layers.xavier_initializer(seed=1));
    b1 = tf.get_variable("b1", shape=[10, 1], initializer=tf.zeros_initializer());
    
    W2 = tf.get_variable("W2", shape=[10, 10], initializer=tf.contrib.layers.xavier_initializer(seed=1));
    b2 = tf.get_variable("b2", shape=[10, 1], initializer=tf.zeros_initializer());
    
    return {"W1" : W1, "b1" : b1,
            "W2" : W2, "b2" : b2};

def forwardPropagate(X, weights):
    '''
    implements forward propation of the neural network
    '''
    
    W1 = weights["W1"];
    b1 = weights["b1"];
    W2 = weights["W2"];
    b2 = weights["b2"];
    
    A1 = tf.sigmoid(tf.matmul(W1, X.T) + b1);
    A2 = tf.sigmoid(tf.matmul(W2, A1) + b2);
    
    return A2;
    
def trainNN():
    pass;
    
    



