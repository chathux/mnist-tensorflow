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

#interprets mnist image files and returns a 3 - dimensional vector.
def getImageData(path='./data/t10k-images.idx3-ubyte'):
    

    arr = [];
    with open(path, "rb") as file:
        magic, size, rows, cols = struct.unpack('>iiii', file.read(16))

        arr = np.fromfile(file, dtype=np.dtype(np.uint8))    
        return np.reshape(arr, (size, rows, cols));

#interprets mnist label data and returns a 1 - dimensional array
def getLabelData(path='./data/t10k-labels.idx1-ubyte'):
    
    arr = [];
    with open(path, "rb") as file:
        magic, size = struct.unpack('>ii', file.read(8))
        arr = np.fromfile(file, dtype=np.dtype(np.uint8))    
        return np.reshape(arr,(size, 1));


train_X = getImageData('./data/train-images.idx3-ubyte');
train_Y = getLabelData('./data/train-labels.idx1-ubyte');

test_X = getImageData('./data/t10k-images.idx3-ubyte');
test_Y = getLabelData('./data/t10k-labels.idx1-ubyte');


