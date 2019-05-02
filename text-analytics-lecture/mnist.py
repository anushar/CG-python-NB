#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:48:03 2018

@author: mia
"""

import gzip
import pickle
import numpy as np



IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_SCALE = 255



def extract_image(filename, IMAGE_SIZE, num_samples):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_samples)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_samples, IMAGE_SIZE * IMAGE_SIZE)/(PIXEL_SCALE*NUM_CHANNELS)
    return data


def extract_labels(filename, num_samples):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_samples)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels



#==============================================================================
# Process test data
#==============================================================================

test_image_filename = '/Users/mia/Downloads/t10k-images-idx3-ubyte.gz'
test_image_arrays = extract_image(test_image_filename, IMAGE_SIZE, 10000)

test_label_filename = '/Users/mia/Downloads/t10k-labels-idx1-ubyte.gz'
test_label_arrays = extract_labels(test_label_filename, 10000)

test = (test_image_arrays, test_label_arrays)

#==============================================================================
# Process training data
#==============================================================================

train_image_filename = '/Users/mia/Downloads/train-images-idx3-ubyte.gz'
train_image_arrays = extract_image(train_image_filename, IMAGE_SIZE, 60000)

train_label_filename = '/Users/mia/Downloads/train-labels-idx1-ubyte.gz'
train_label_arrays = extract_labels(train_label_filename, 60000)

train = (train_image_arrays, train_label_arrays)

#==============================================================================
# Save
#==============================================================================

output_file = gzip.open("/Users/mia/Downloads/gzip/letters/mnist.pkl.gz", "wb")
pickle.dump((train,test), output_file)
output_file.close()


