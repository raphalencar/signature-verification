from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os

#from keras.utils.visualize_util import plot

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras import backend as K
import sys
import glob
from tqdm import tqdm
from keras.regularizers import l2 
import logging
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
random.seed(1337)

# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger()

def euclidean_distance(vects):
    assert len(vects) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    # return (shape1[1], 0)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    # y_pred --> Dw, euclidean distance computed in the embedded space

    """ What you want is double the distance if the pairs are equal - 
    this is the loss for pairs should be be with "zero" distance. 
    But if the pairs are distinct from each other you want to calculate
    their distance from the margin and double it.
    """

    margin = 1

    """ Dw to be close to 0 when y_true is 1 (for positive pairs) and 
        Dw close or bigger than 1 when y_true is 0 (for negative pairs)
        Distance low for similar pairs and high for diff images
    """    
    
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
    """In the case below, similar pairs: y=0 and dissimilar y=1
       Dw = 1 for similar pairs 
    """
    # return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    # corresponding label += [0,1] instead of label += [1,0]


# def create_base_network_signet(input_shape):
    
#     seq = Sequential()
#     seq.add(Conv2D(96, (11, 11), activation='relu', name='conv1_1', strides=(4,4), input_shape= input_shape, 
#                         kernel_initializer='glorot_uniform', data_format='channels_last'))
#     seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
#     seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
#     seq.add(ZeroPadding2D((2, 2),))
    
#     seq.add(Conv2D(256, (5, 5), activation='relu', name='conv2_1', strides=(1, 1), kernel_initializer='glorot_uniform'))
#     seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
#     seq.add(MaxPooling2D((3,3), strides=(2, 2)))
#     seq.add(Dropout(0.3))# added extra
#     seq.add(ZeroPadding2D((1, 1)))
    
#     seq.add(Conv2D(384, (3, 3), activation='relu', name='conv3_1', strides=(1, 1), kernel_initializer='glorot_uniform'))
#     seq.add(ZeroPadding2D((1, 1)))
    
#     seq.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2', strides=(1, 1), kernel_initializer='glorot_uniform'))    
#     seq.add(MaxPooling2D((3,3), strides=(2, 2)))
#     seq.add(Dropout(0.3))# added extra
# #    model.add(SpatialPyramidPooling([1, 2, 4]))
#     seq.add(Flatten(name='flatten'))
#     seq.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
#     seq.add(Dropout(0.5))
    
#     seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) # softmax changed to relu
    
#     print (seq.summary())
#     return seq

def create_base_network_signet(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(filters=96, kernel_size=11, activation='relu', kernel_regularizer=l2(2e-4))(input)
    x = MaxPooling2D(pool_size=(3,3))(x)

    x = Conv2D(filters=256, kernel_size=5, activation='relu', kernel_regularizer=l2(2e-4))(x)
    x = MaxPooling2D(pool_size=(3,3))(x)

    x = Conv2D(filters=384, kernel_size=3, activation='relu', kernel_regularizer=l2(2e-4))(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)

def compute_accuracy_roc(predictions, labels):
   # Compute ROC accuracy with a range of thresholds on distances.
   dmax = np.max(predictions)
   dmin = np.min(predictions)
   nsame = np.sum(labels == 1)
   ndiff = np.sum(labels == 0)
   
   step = 0.01
   max_acc = 0
   
   for d in np.arange(dmin, dmax+step, step):
       idx1 = predictions.ravel() <= d
       idx2 = predictions.ravel() > d
       
       tpr = float(np.sum(labels[idx1] == 1)) / nsame       
       tnr = float(np.sum(labels[idx2] == 0)) / ndiff
       acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
       if (acc > max_acc):
           max_acc = acc
           
   return max_acc

def compute_accuracy(predictions, labels):
    """ Compute classification accuracy with a fixed threshold on distances.
    """
    return labels[predictions.ravel() < 0.5].mean()
    # return np.mean(labels==(predictions.ravel() > 0.5))

# def create_data(height,width,dir):
#     files = glob.glob(dir + '/*.PNG')
#     nb_classes = 16 # sigcomp 2011
#     # filename from path: file.rsplit('/', 1)[-1]
#     # cedar data specific, will get 1_1; original from start and .png from end removed
#     img_names = [file.rsplit('/')[1] for file in files]
#     data = []
#     labels = []
#     for i in range(len(img_names)):
#         class_index = int(img_names[i].split('_')[0])
#         label = class_index - 1
#         img = image.load_img(files[i], grayscale = True, target_size=(height, width))
#         img = image.img_to_array(img)#, dim_ordering='tf')
#         data.append(img)
#         labels.append(label)

#     return np.array(data), np.array(labels), nb_classes
 
# def create_data_test(height,width,dir):
#     files = glob.glob(dir + '/*.png')
#     nb_classes = 16 # sigcomp 2011
#     # filename from path: file.rsplit('/', 1)[-1]
#     # cedar data specific, will get 1_1; original from start and .png from end removed
#     img_names = [file.rsplit('/')[1] for file in files]
#     data = []
#     labels = []
#     for i in range(len(img_names)):
#         class_index = int(img_names[i].split('_')[0])
#         label = class_index - 1
#         img = image.load_img(files[i], grayscale = True, target_size=(height, width))
#         img = image.img_to_array(img)#, dim_ordering='tf')
#         data.append(img)
#         labels.append(label)

    return np.array(data), np.array(labels), nb_classes

def create_pairs(x, digit_indices, nb_classes):
    """      x:         X_train, array of array of all train samples.
        digit_indices:  List of array, length = no of classes; each sublist consists of train sample indices
                        belonging to that particular class index/class
    """
    """ Positive and negative pair creation.
        Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    
    print ('\n\n')
    print ('X_train shape:  ', x.shape)
    print ('Digit_indices shape:    ', np.array(digit_indices).shape)
    print ('Length of digit indices:    ', len(digit_indices))
    print ('No of classes:  ', nb_classes)
    print ('\n\n')

    n = min([len(digit_indices[d]) for d in range(nb_classes)]) - 1
    for d in range(nb_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, nb_classes)
            dn = (d + inc) % nb_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]    #check label based on similarity = 0 or 1
            # labels += [0,1]     #similar pairs = 0 in this case
    return np.array(pairs), np.array(labels)

def get_valid_dn(d, digit_indices):
    inc = random.randrange(1, nb_classes)
    dn = (d + inc) % nb_classes
    while len(digit_indices[dn]) == 0:
        inc = random.randrange(1, nb_classes)
        dn = (d + inc) % nb_classes
    return dn

def format_input(X):
    if not(type(X) is np.ndarray):   
        X = np.array(X)
        print ('Formatted input by converting to array')
    try:
        print ('Input shape before reshaping:   ', X.shape)
        # X = X.reshape(X.shape[0], X.shape[1], img_height, img_width, 1)
        X = X.reshape(X.shape[0], img_height, img_width, 1)
        X = X.astype('float32')
        X /= 255

        return X
    except Exception as e:
        print ('EXCEPTION while reshaping and formatting input data X:  \n', e)
        print (log.exception('ERROR MESSAGE'))

def path_to_tensor(img_path):
    img_height = 155
    img_width = 220
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_height, img_width, 1))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    return files, targets

x, y = load_dataset('data')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# parameters
featurewise_std_normalization = True
nb_classes = 64
epochs = 20
batch_size = 64    
# input_shape=(img_height, img_width, 1)

# pre-process the data for Keras
X_train = paths_to_tensor(X_train).astype('float32')/255
X_test = paths_to_tensor(X_test).astype('float32')/255
input_shape = X_train.shape[1:]

# load train data
# data_dir_train = 'train'
# X_train, y_train, nb_classes = create_data(img_height, img_width, data_dir_train)    
# X_train = format_input(X_train)

# create training positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices, nb_classes)

# for d in range(len(digit_indices)):    
#         # 16 instances of each class in sigcomp2011 data    
#         print('No. of instances for class %d:   %d'%(d,len(list(digit_indices[d]))))

# load test data
# data_dir_test = 'test'
# X_test, y_test, nb_classes = create_data_test(img_height, img_width, data_dir_test)
# X_test = format_input(X_test) 

# create testing positive and negative pairs
digit_indices = [np.where(y_test == i)[0] for i in range(nb_classes)]

count = 0
aux = np.array([])
for i in range(len(digit_indices)):
    if (len(digit_indices[i]) <= 1):
        print('{}'.format(i))
        aux = np.append(aux, [i])
        count += 1

digit_indices = np.delete(digit_indices, aux)
        
te_pairs, te_y = create_pairs(X_test, digit_indices, nb_classes - count)
    
# network definition
base_network = create_base_network_signet(input_shape)
base_network.summary()

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)
    
# compile model
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
adadelta = Adadelta()
model.compile(loss=contrastive_loss, optimizer=rms)

checkpointer = ModelCheckpoint(filepath='best_model/model_weights.hdf5', 
                               verbose=1, save_best_only=True)

history_callback = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, 
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
    callbacks=[checkpointer],
    verbose=1)

train_loss_hist = history_callback.history['loss']
train_loss = np.array(train_loss_hist)
np.savetxt("train_loss.csv", train_loss, delimiter=",")

val_loss_hist = history_callback.history['val_loss']
val_loss = np.array(val_loss_hist)
np.savetxt("val_loss.csv", val_loss, delimiter=",")
# model.save('best_model/model.h5')
# display model
# plot(model, show_shapes=True)
# sys.exit()     
    
# # callbacks
# fname = 'model_weights.hdf5'

# checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
# tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)

# compute final accuracy on training and test sets
# pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy_roc(pred, tr_y)
# tr_acc = compute_accuracy(pred, tr_y)
# pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(pred, te_y)
# compute final accuracy on training and test sets
model.load_weights('best_model/model_weights.hdf5')
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(y_pred, tr_y)

y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(y_pred, te_y)


# # model.fit_generator(generator=datagen.next_train(), samples_per_epoch=datagen.samples_per_train, nb_epoch=nb_epoch,
# #                     validation_data=datagen.next_valid(), nb_val_samples=int(datagen.samples_per_valid))   # KERAS 1

# model.fit_generator(generator=datagen.next_train(), steps_per_epoch=960, epochs=nb_epoch,
# #                     validation_data=datagen.next_valid(), validation_steps=120, callbacks=[checkpointer])  # KERAS 2
    

# # tr_pred = model.predict_generator(generator=datagen.next_train(), val_samples=int(datagen.samples_per_train))
# te_pred = model.predict_generator(generator=datagen.next_test(), steps=120)

# #tr_acc = compute_accuracy_roc(tr_pred, datagen.train_labels)
# te_acc = compute_accuracy_roc(te_pred, datagen.test_labels)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))