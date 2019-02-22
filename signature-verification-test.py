from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="Path to the deep ranking model")

ap.add_argument("-i1", "--image1", required=True,
    help="Path to the first image")

ap.add_argument("-i2", "--image2", required=True,
    help="Path to the second image")

args = vars(ap.parse_args())

if not os.path.exists(args['model']):
    print("The model path doesn't exist!")
    exit()

if not os.path.exists(args['image1']):
    print("The image 1 path doesn't exist!")
    exit()

if not os.path.exists(args['image2']):
    print("The image 2 path doesn't exist!")
    exit()

args = vars(ap.parse_args())

import random
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, GlobalAveragePooling2D, MaxPooling2D, Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from PIL import Image
from keras.preprocessing import image                  
from tqdm import tqdm
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from PIL import ImageFile      
from keras.regularizers import l2  
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.callbacks import ModelCheckpoint

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_base_network_signet(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(filters=96, kernel_size=11, activation='relu')(input)
    x = MaxPooling2D(pool_size=(3,3))(x)

    x = Conv2D(filters=256, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3))(x)

    x = Conv2D(filters=384, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)

img_height = 155
img_width = 220

image1 = load_img(args['image1'])
image1 = img_to_array(image1).astype("float32")
image1 = transform.resize(image1, (img_height, img_width))
image1 *= 1./255
image1 = np.expand_dims(image1, axis = 0)
input_shape = image1.shape[1:]

image2 = load_img(args['image2'])
image2 = img_to_array(image2).astype("float32")
image2 = transform.resize(image2, (img_height, img_width))
image2 *= 1./255
image2 = np.expand_dims(image2, axis = 0)

print(input_shape)

base_network = create_base_network_signet(input_shape)

#for layer in base_network.layers[:-3]:
#     layer.trainable = False

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
model.load_weights(args['model'])
print(image1.shape)
print(image2.shape)
print(model.predict([image1, image2]))