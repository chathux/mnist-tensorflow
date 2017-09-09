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
    
    #(m, n_x)
    train_X_prep = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]);
    test_X_prep = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]);

    with tf.Session() as sess:
        
        #(m, 10)
        train_Y_prep = sess.run(tf.one_hot(train_Y.reshape(-1), 10, axis=1));
        test_Y_prep = sess.run(tf.one_hot(test_Y.reshape(-1), 10, axis=1));

    return train_X_prep, train_Y_prep, test_X_prep, test_Y_prep;

def initializeWeights():
    '''
    initialize the weights for the 3 layer neueral network
    '''
    
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
    
    Z1 = tf.matmul(W1, X) + b1;
    A1 = tf.sigmoid(Z1);
    
    Z2 = tf.matmul(W2, A1) + b2;
    
    return Z2;
    
def trainNN(train_X, train_Y, learning_rate=0.001, iterations=20):
    '''
    trains the neural network.
    returns the optimized weights as a dictionary.
    '''
    
    tf.reset_default_graph();
    

    
    X = tf.placeholder(dtype=tf.float32, shape=[train_X.shape[1], None])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, train_Y.shape[1]])
    
    
    weights = initializeWeights();
    logits = forwardPropagate(X, weights); 
    
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(logits),  labels=Y));
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost);
    
    with tf.Session() as sess:
            
        sess.run(tf.global_variables_initializer());        
        
        #transpose X to eliminate transpose operation at each epoch
        train_X_T = sess.run(tf.transpose(train_X));
        
        for i in range(0,iterations):

            epoch_cost, _ = sess.run([cost, optimizer], feed_dict={X:train_X_T, Y:train_Y });
            print("epoch %i cost : %f "%(i, epoch_cost));
            
        weights = sess.run(weights);    
    return weights;
    
def predict(test_X, weights):
    '''
    predicts the results for test_X based on the given weights.
    '''
    
    X = tf.placeholder(dtype=tf.float32, shape=[test_X.shape[1], None])    
    
    pred_Y = tf.argmax(forwardPropagate(X, weights), axis=0);
    
    with tf.Session() as sess:
        test_X_T = sess.run(tf.transpose(test_X));
        pred_Y = sess.run(pred_Y, feed_dict={X: test_X_T})

    return pred_Y;

train_X, train_Y, test_X, test_Y = prepareDataSets();

weights = trainNN(train_X, train_Y, iterations=1500);

pred_train_Y = predict(train_X, weights);
pred_test_Y = predict(test_X, weights);


print("Train Accuracy :%f, Test Accuracy : %f"%
      (
      tf.Session().run(tf.reduce_mean(tf.cast(tf.equal(pred_train_Y, tf.argmax(train_Y, axis=-1)), "float"))),
      tf.Session().run(tf.reduce_mean(tf.cast(tf.equal(pred_test_Y, tf.argmax(test_Y, axis=-1)), "float")))
      )
)
#20 - 0.079599999
#100 - 0.1185
#200 - 0.1603
#1500- train : 0.445450 test - 0.451300